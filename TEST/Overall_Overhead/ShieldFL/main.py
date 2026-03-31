import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import random
import math
from phe import paillier

from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def set_model_vector(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())

# ==========================================
# 通信量理论计算引擎 (ShieldFL)
# ==========================================
def calc_shieldfl_comm(d, num_clients, key_bits=512):
    """
    计算 ShieldFL 的精确通信量 (MB)
    N 的位数为 key_bits，Paillier密文在 Z_{N^2}^* 中，占 2*key_bits 位
    """
    c_bytes = (2 * key_bits) // 8  # 128 Bytes (密文)
    p_bytes = 4                    # 4 Bytes (Float32 明文)
    
    # 1. 客户端平均通信量 (上传密文 + 下载明文)
    client_up = (d + 1) * c_bytes  # d 个梯度密文 + 1 个范数平方密文
    client_down = d * p_bytes      # d 个全局模型明文
    client_avg_mb = (client_up + client_down) / (1024 * 1024)
    
    # 2. 服务器端总通信量 (S1 <-> S2 交互 + S1 下发)
    # [SecJudge]: 每客户端 S1 发送1个盲化求和密文，S2 返回1个明文
    secjudge_comm = num_clients * (c_bytes + p_bytes)
    # [SecCos]: 每客户端 S1 发送2个盲化密文(内积, 范数), S2 返回2个明文
    seccos_comm = num_clients * (2 * c_bytes + 2 * p_bytes)
    # [全局解密]: S1 发送 d 个聚合密文，S2 返回 d 个明文
    dec_comm = d * c_bytes + d * p_bytes
    # [广播下发]: S1 向所有客户端广播明文模型
    broadcast_comm = num_clients * d * p_bytes
    
    server_total_mb = (secjudge_comm + seccos_comm + dec_comm + broadcast_comm) / (1024 * 1024)
    return client_avg_mb, server_total_mb

# ==========================================
# 密码服务提供商 (CSP) / 模拟服务器 S2
# ==========================================
class CryptographicServiceProvider:
    def __init__(self, key_length=512):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)

    def get_public_key(self):
        return self.public_key

    def secure_judge_verify(self, enc_sum_blinded, r_sq_sum):
        sum_blinded = self.private_key.decrypt(enc_sum_blinded)
        return sum_blinded - r_sq_sum

    def secure_verify_cosine_similarity(self, enc_dot_blinded, enc_norm_A_blinded, norm_B_blinded_pt):
        X = self.private_key.decrypt(enc_dot_blinded)
        Y = self.private_key.decrypt(enc_norm_A_blinded)
        if Y <= 0 or norm_B_blinded_pt <= 0:
            return False, -1.0
        cos_sim = X / math.sqrt(Y * norm_B_blinded_pt)
        return (cos_sim >= 0.0), cos_sim

    def decrypt_global_model(self, enc_global_update, deg, num_benign):
        plaintext_update = [(self.private_key.decrypt(enc_x) / (deg * num_benign)) for enc_x in enc_global_update]
        return torch.tensor(plaintext_update, dtype=torch.float32)

# ==========================================
# 客户端类 (ShieldFL)
# ==========================================
class ShieldFLClient:
    def __init__(self, client_data, model_class, public_key, device='cpu', kappa=1e-3, deg=100000):
        self.client_id = client_data['client_id']
        self.is_malicious = client_data['is_malicious']
        self.train_loader = client_data['host_loader']
        self.device = device
        self.model = model_class().to(self.device)
        self.public_key = public_key
        self.kappa = kappa
        self.deg = deg
        self.time_logs = {}

    def run_local_training_and_encrypt(self, global_weights, local_epochs):
        # 1. 本地训练
        start_train = time.time()
        set_model_vector(self.model, global_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()
        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        local_update = get_model_vector(self.model) - global_weights
        self.time_logs['t_train'] = time.time() - start_train
        
        # 2. 稀疏化与定点数放缩
        start_scale = time.time()
        mask = torch.abs(local_update) >= self.kappa
        sparse_update = local_update * mask
        norm = torch.norm(sparse_update, p=2)
        normalized_update = sparse_update / norm if norm > 0 else sparse_update
        scaled_update_np = np.round(normalized_update.cpu().numpy() * self.deg).astype(int)
        self.time_logs['t_scale'] = time.time() - start_scale
        
        # 3. Paillier 密文加密
        start_enc = time.time()
        enc_update = [self.public_key.encrypt(int(x)) for x in scaled_update_np]
        norm_A_sq = int(np.sum(scaled_update_np.astype(np.int64) ** 2))
        enc_norm_A_sq = self.public_key.encrypt(norm_A_sq)
        self.time_logs['t_enc'] = time.time() - start_enc
        
        return enc_update, enc_norm_A_sq

# ==========================================
# 云服务器系统 (ShieldFL Server)
# ==========================================
class CloudServer:
    def __init__(self, model_shape, csp, deg=100000, device='cpu'):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.csp = csp
        self.public_key = csp.get_public_key()
        self.device = device
        self.d = model_shape[0]
        self.deg = deg
        self.reference_gradient_int = np.ones(self.d, dtype=np.int64) * self.deg

    def init_global_weights(self, model):
        self.global_weights = get_model_vector(model).clone()

    def secure_cosine_similarity_aggregation(self, client_packages):
        benign_enc_updates, statuses = [], []
        time_logs = {'t_secjudge': 0.0, 't_seccos': 0.0, 't_agg': 0.0, 't_dec': 0.0}
        norm_B_sq = int(np.sum(self.reference_gradient_int ** 2))
        
        for i, (enc_A, enc_norm_A_sq) in enumerate(client_packages):
            # [SecJudge] 阶段
            t0 = time.time()
            r_sq_sum = random.randint(1, 1000)
            enc_sum_blinded = enc_norm_A_sq + r_sq_sum 
            real_sum = self.csp.secure_judge_verify(enc_sum_blinded, r_sq_sum)
            time_logs['t_secjudge'] += (time.time() - t0)
            
            target_sum = self.deg ** 2
            if abs(real_sum - target_sum) > (target_sum * 0.05):
                statuses.append(f"Rejected by SecJudge")
                continue

            # [SecCos] 阶段
            t0 = time.time()
            enc_dot_product = self.public_key.encrypt(0)
            for enc_a_i, b_i in zip(enc_A, self.reference_gradient_int):
                enc_dot_product += (enc_a_i * int(b_i))
                
            r = random.randint(1, 5)
            r_sq = r ** 2
            is_benign, cos_sim = self.csp.secure_verify_cosine_similarity(
                enc_dot_product * r_sq, enc_norm_A_sq * r_sq, norm_B_sq * r_sq 
            )
            time_logs['t_seccos'] += (time.time() - t0)
            
            if is_benign:
                benign_enc_updates.append(enc_A)
                statuses.append(f"Benign (Cos: {cos_sim:.4f})")
            else:
                statuses.append(f"Poisoned (Cos: {cos_sim:.4f})")
                
        # [密文聚合] 阶段
        t0 = time.time()
        num_benign = len(benign_enc_updates)
        if num_benign > 0:
            enc_global_update = benign_enc_updates[0]
            for j in range(1, num_benign):
                for k in range(self.d):
                    enc_global_update[k] += benign_enc_updates[j][k]
        else:
            enc_global_update = [self.public_key.encrypt(0) for _ in range(self.d)]
            num_benign = 1
        time_logs['t_agg'] += (time.time() - t0)
        
        # [全局解密] 阶段
        t0 = time.time()
        plaintext_tensor = self.csp.decrypt_global_model(enc_global_update, self.deg, num_benign)
        self.global_weights += plaintext_tensor.to(self.device)
        normalized_ref = plaintext_tensor / torch.norm(plaintext_tensor)
        self.reference_gradient_int = np.round(normalized_ref.cpu().numpy() * self.deg).astype(np.int64)
        time_logs['t_dec'] += (time.time() - t0)
        
        return self.global_weights, statuses, time_logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LeNet5', choices=['LeNet5', 'ResNet18'])
    parser.add_argument('--num_clients', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csp = CryptographicServiceProvider(key_length=512)
    model_class = LeNet5 if args.model == 'LeNet5' else ResNet18_CIFAR10
    dataset_name = 'MNIST' if args.model == 'LeNet5' else 'CIFAR10'

    client_dataloaders, _ = get_federated_dataloaders(dataset_name, args.num_clients, 0.33, 64)
    clients = [ShieldFLClient(data, model_class, csp.get_public_key(), device) for data in client_dataloaders]
    dummy_model = model_class().to(device)
    
    d = get_model_vector(dummy_model).size(0)
    server = CloudServer((d,), csp, device=device)
    server.init_global_weights(dummy_model)

    # 打印通信量分析
    c_avg_mb, s_total_mb = calc_shieldfl_comm(d, args.num_clients)
    print(f"\n[*] ShieldFL 通信量分析 (模型维度 {d}):")
    print(f"    - 客户端平均通信量 (上+下): {c_avg_mb:.2f} MB")
    print(f"    - 服务器端总通信量 (交互+下发): {s_total_mb:.2f} MB")

    print(f"\n>>> 开始 ShieldFL 单轮完整流程测试...")
    client_packages = []
    c_train_time, c_scale_time, c_enc_time = 0, 0, 0
    
    for client in clients:
        enc_A, enc_norm_A = client.run_local_training_and_encrypt(server.global_weights, 1)
        client_packages.append((enc_A, enc_norm_A))
        c_train_time += client.time_logs['t_train']
        c_scale_time += client.time_logs['t_scale']
        c_enc_time += client.time_logs['t_enc']
        
    _, statuses, s_time_logs = server.secure_cosine_similarity_aggregation(client_packages)
    
    print("\n  [客户端时间耗时 (平均)]")
    print(f"  - 本地模型训练 : {c_train_time/args.num_clients:.4f} s")
    print(f"  - 稀疏化与定点化 : {c_scale_time/args.num_clients:.4f} s")
    print(f"  - 全维 HE 加密 : {c_enc_time/args.num_clients:.4f} s")
    
    print("\n  [服务器端时间耗时 (总计)]")
    print(f"  - SecJudge 验证 : {s_time_logs['t_secjudge']:.4f} s")
    print(f"  - SecCos 相似度计算 : {s_time_logs['t_seccos']:.4f} s")
    print(f"  - 密文同态聚合 : {s_time_logs['t_agg']:.4f} s")
    print(f"  - 全局模型解密还原 : {s_time_logs['t_dec']:.4f} s")

if __name__ == "__main__":
    main()