import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import random
import math
from phe import paillier  # 真实的 Paillier 同态加密库

# 导入外部定义的模型与数据加载器
from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def set_model_vector(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())

# ==========================================
# 密码服务提供商 (CSP)
# ==========================================
class CryptographicServiceProvider:
    def __init__(self, key_length=512):
        print(f"[*] CSP 正在生成 {key_length} 位 Paillier 真实公私钥对...")
        # 记录密钥长度以计算精确通信量
        self.key_length = key_length 
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        
    def get_public_key(self):
        return self.public_key
        
    def secure_verify_cosine_similarity(self, enc_dot_blinded, enc_norm_A_blinded, norm_B_blinded_pt):
        """
        [ShieldFL 核心步骤 4]: CSP 解密盲化后的密文，计算余弦相似度
        X = r^2 (A·B),  Y = r^2 ||A||^2,  Z = r^2 ||B||^2
        CosSim = X / sqrt(Y * Z) = (r^2 A·B) / (r^2 ||A|| ||B||) -> 盲化因子 r^2 完美抵消
        """
        start_dec = time.time()
        X = self.private_key.decrypt(enc_dot_blinded)
        Y = self.private_key.decrypt(enc_norm_A_blinded)
        Z = norm_B_blinded_pt
        
        dec_time = time.time() - start_dec
        
        if Y <= 0 or Z <= 0:
            return False, -1.0, dec_time
            
        cos_sim = X / math.sqrt(Y * Z)
        is_benign = cos_sim >= 0.0
        return is_benign, cos_sim, dec_time

    def decrypt_global_model(self, enc_global_update):
        """CSP 协助解密最终的安全聚合结果"""
        print(f"    [CSP] 正在解密全局聚合模型 (维度: {len(enc_global_update)})...")
        start_dec = time.time()
        plaintext_update = [self.private_key.decrypt(enc_x) for enc_x in enc_global_update]
        dec_time = time.time() - start_dec
        return torch.tensor(plaintext_update, dtype=torch.float32), dec_time

# ==========================================
# 客户端类 (ShieldFL Client)
# ==========================================
class ShieldFLClient:
    def __init__(self, client_data, model_class, public_key, device='cpu'):
        self.client_id = client_data['client_id']
        self.is_malicious = client_data['is_malicious']
        self.train_loader = client_data['host_loader']
        self.device = device
        self.model = model_class().to(self.device)
        self.public_key = public_key
        self.time_logs = {}

    def run_local_training_and_encrypt(self, global_weights, local_epochs):
        start_train = time.time()
        set_model_vector(self.model, global_weights)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()
        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        local_update = get_model_vector(self.model) - global_weights
        self.time_logs['local_train'] = time.time() - start_train
        
        # [ShieldFL 核心步骤 1]: 客户端对全量参数进行逐元素公钥加密
        start_he = time.time()
        update_list = local_update.cpu().numpy().astype(np.float64).tolist()
        
        # 客户端计算并加密自己的范数平方 [[ ||A||^2 ]]
        norm_A_sq = sum(x * x for x in update_list)
        enc_norm_A_sq = self.public_key.encrypt(norm_A_sq)
        
        # 全维度逐元素加密 [[ A_i ]]
        print(f"    [Client {self.client_id}] 开始加密全维度梯度 ({len(update_list)} params)... 请耐心等待。")
        enc_update = [self.public_key.encrypt(x) for x in update_list]
        
        self.time_logs['encryption'] = time.time() - start_he
        return enc_update, enc_norm_A_sq

# ==========================================
# 云服务器系统 (Cloud Server)
# ==========================================
class CloudServer:
    def __init__(self, model_shape, csp, device='cpu'):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.csp = csp
        self.public_key = csp.get_public_key()
        self.device = device
        self.d = model_shape[0]
        # 参考梯度 B，初始化为 1
        self.reference_gradient = np.ones(self.d, dtype=np.float64) 

    def init_global_weights(self, model):
        self.global_weights = get_model_vector(model).clone()

    def secure_cosine_similarity_aggregation(self, client_packages):
        start_server = time.time()
        benign_enc_updates = []
        statuses = []
        
        total_he_mul_time = 0.0
        total_csp_dec_time = 0.0
        
        norm_B_sq = sum(x * x for x in self.reference_gradient)
        
        for i, (enc_A, enc_norm_A_sq) in enumerate(client_packages):
            op_start = time.time()
            
            # [ShieldFL 核心步骤 2]: 密文状态下的同态内积 [[ A·B ]] = \sum [[ A_i ]] * B_i
            enc_dot_product = self.public_key.encrypt(0.0)
            for enc_a_i, b_i in zip(enc_A, self.reference_gradient):
                enc_dot_product += (enc_a_i * b_i)
                
            # [ShieldFL 核心步骤 3]: 随机乘法盲化 (Multiplicative Blinding)
            r = random.uniform(1.0, 5.0)
            r_sq = r ** 2
            
            enc_dot_blinded = enc_dot_product * r_sq
            enc_norm_A_blinded = enc_norm_A_sq * r_sq
            norm_B_blinded_pt = norm_B_sq * r_sq 
            
            total_he_mul_time += (time.time() - op_start)
            
            # 交互：发送给 CSP
            is_benign, cos_sim, dec_time = self.csp.secure_verify_cosine_similarity(
                enc_dot_blinded, enc_norm_A_blinded, norm_B_blinded_pt
            )
            total_csp_dec_time += dec_time
            
            if is_benign:
                benign_enc_updates.append(enc_A)
                statuses.append(f"Benign (CosSim: {cos_sim:.4f})")
            else:
                statuses.append(f"Poisoned (CosSim: {cos_sim:.4f})")
                
        # [ShieldFL 核心步骤 5]: 密文状态下的同态安全聚合 \sum [[ A_i ]]
        agg_start = time.time()
        if len(benign_enc_updates) > 0:
            num_benign = len(benign_enc_updates)
            # 逐元素同态相加
            enc_global_update = benign_enc_updates[0]
            for j in range(1, num_benign):
                for k in range(self.d):
                    enc_global_update[k] += benign_enc_updates[j][k]
            # 同态均值 (乘以 1/N)
            for k in range(self.d):
                enc_global_update[k] *= (1.0 / num_benign)
        else:
            enc_global_update = [self.public_key.encrypt(0.0) for _ in range(self.d)]
            
        total_he_mul_time += (time.time() - agg_start)
        
        # 请求 CSP 解密全局聚合结果
        plaintext_update_tensor, dec_time2 = self.csp.decrypt_global_model(enc_global_update)
        total_csp_dec_time += dec_time2
        
        self.global_weights += plaintext_update_tensor.to(self.device)
        self.reference_gradient = plaintext_update_tensor.numpy().astype(np.float64)
        
        server_total_time = time.time() - start_server
        return self.global_weights, statuses, server_total_time, total_he_mul_time, total_csp_dec_time

# ==========================================
# 理论通信量精确计算引擎
# ==========================================
def calculate_shieldfl_communication(d, num_clients, key_length_bits=512):
    """
    Paillier 密文膨胀计算: 
    N 的位数为 key_length_bits，密文存在于 Z_{N^2}^* 中，因此密文位数为 2 * key_length_bits
    """
    ciphertext_bytes = (2 * key_length_bits) // 8
    
    # 1. 客户端上传: d 个密文 (梯度) + 1 个密文 (范数平方)
    client_to_cs = num_clients * (d + 1) * ciphertext_bytes
    
    # 2. CS 发送盲化数据给 CSP: 每个客户端 2 个密文 (盲化内积, 盲化范数)
    cs_to_csp = num_clients * 2 * ciphertext_bytes
    
    # 3. CSP 返回判定给 CS: 每个客户端 1 byte (布尔值)
    csp_to_cs = num_clients * 1
    
    # 4. CS 发送聚合模型给 CSP 解密: d 个密文
    cs_to_csp_agg = d * ciphertext_bytes
    
    # 5. CSP 返回明文聚合模型给 CS: d 个 float32 (4 bytes)
    csp_to_cs_agg = d * 4
    
    # 6. CS 下发全局模型给客户端: num_clients * d * 4 bytes
    cs_to_client = num_clients * d * 4
    
    total_bytes = client_to_cs + cs_to_csp + csp_to_cs + cs_to_csp_agg + csp_to_cs_agg + cs_to_client
    return total_bytes / (1024 * 1024)  # 转换为 MB

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LeNet5', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--poison_rate', type=float, default=0.33)
    parser.add_argument('--num_rounds', type=int, default=1) # 真实全维度 HE 极耗时，建议设为 1 轮验证流程
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 启动 ShieldFL 全维度绝对真实复现 | Device: {device}")
    
    csp = CryptographicServiceProvider(key_length=512)
    
    model_class = LeNet5 if args.model == 'LeNet5' else (ResNet18_CIFAR10 if args.model == 'ResNet18' else resnet20)
    dataset_name = 'MNIST' if args.model == 'LeNet5' else 'CIFAR10'

    client_dataloaders, test_loader = get_federated_dataloaders(
        dataset_name=dataset_name, num_clients=args.num_clients, poison_rate=args.poison_rate, batch_size=args.batch_size
    )
    
    clients = [ShieldFLClient(data, model_class, csp.get_public_key(), device=device) for data in client_dataloaders]
    dummy_model = model_class().to(device)
    
    d = get_model_vector(dummy_model).size(0)
    server = CloudServer(model_shape=(d,), csp=csp, device=device)
    server.init_global_weights(dummy_model)
    
    comm_mb = calculate_shieldfl_communication(d, args.num_clients, key_length_bits=512)
    print(f"[*] 模型参数量: {d} | 单轮理论精确通信量: {comm_mb:.2f} MB")
    print("-" * 60)

    for round_idx in range(args.num_rounds):
        print(f"\n>>> [Round {round_idx+1}/{args.num_rounds}] 开始 (警告: 真实全维 HE 计算将持续较长时间)")
        
        client_packages = []
        round_local_train_time = 0.0
        round_encryption_time = 0.0
        
        for client in clients:
            enc_update, enc_norm_A_sq = client.run_local_training_and_encrypt(server.global_weights, args.local_epochs)
            client_packages.append((enc_update, enc_norm_A_sq))
            round_local_train_time += client.time_logs['local_train']
            round_encryption_time += client.time_logs['encryption']
            
        global_weights, statuses, server_time, he_mul_time, csp_dec_time = server.secure_cosine_similarity_aggregation(client_packages)
        
        for i, status in enumerate(statuses):
            mal_flag = "[MAL]" if clients[i].is_malicious else "[BEN]"
            print(f"    Client_{i:02d} {mal_flag} -> {status}")
            
        print(f"\n  [时间与通信指标 (单轮)]")
        print(f"  - 客户端本地明文训练耗时   : {round_local_train_time:.4f} 秒")
        print(f"  - 客户端全维 HE 加密耗时   : {round_encryption_time:.4f} 秒")
        print(f"  - 云服务器 HE 同态乘法耗时 : {he_mul_time:.4f} 秒")
        print(f"  - CSP 节点解密运算耗时     : {csp_dec_time:.4f} 秒")
        print(f"  - 双服务器系统总执行耗时   : {server_time:.4f} 秒")
        print(f"  - 系统精确通信量 (MB)      : {comm_mb:.2f} MB")

if __name__ == "__main__":
    main()