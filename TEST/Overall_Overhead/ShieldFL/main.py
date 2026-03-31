import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import random
import math
from phe import paillier
from tqdm import tqdm

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
        # 按照论文 80-bit 安全级别，对应 N=1024 bits (这里为加快演示默认设为 512, 您可改为 1024)
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)

    def get_public_key(self):
        return self.public_key

# ==========================================
# 客户端类 (ShieldFL Client)
# ==========================================
class ShieldFLClient:
    def __init__(self, client_data, model_class, public_key, device='cpu', deg=100000):
        self.train_loader = client_data['host_loader']
        self.device = device
        self.model = model_class().to(self.device)
        self.public_key = public_key
        self.deg = deg
        self.time_logs = {'t_train': 0.0, 't_scale': 0.0, 't_enc': 0.0}

    def run_local_training_and_encrypt(self, global_weights, local_epochs):
        # 1. 真实本地训练
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
        
        # 2. 稀疏化与放缩
        start_scale = time.time()
        kappa = torch.quantile(torch.abs(local_update), 0.9).item() 
        mask = torch.abs(local_update) >= kappa
        sparse_update = local_update * mask
        norm = torch.norm(sparse_update, p=2) + 1e-12 
        normalized_update = sparse_update / norm
        scaled_update_np = np.round(normalized_update.cpu().numpy() * self.deg).astype(np.int64)
        self.time_logs['t_scale'] = time.time() - start_scale
        
        # 3. 逐维度真实公钥加密 (完全遵循论文，不提前计算平方和)
        start_enc = time.time()
        enc_update = [self.public_key.encrypt(int(x)) for x in scaled_update_np]
        self.time_logs['t_enc'] = time.time() - start_enc
        
        return enc_update

# ==========================================
# 云服务器系统 (ShieldFL Server - 真实操作版)
# ==========================================
class CloudServer:
    def __init__(self, model_shape, csp, deg=100000, device='cpu', key_length=512):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.csp = csp
        self.public_key = csp.get_public_key()
        self.device = device
        self.d = model_shape[0]
        self.deg = deg
        
        # 通信量统计参数
        self.c_bytes = (2 * key_length) // 8  # 密文字节数
        self.p_bytes = 8                      # 明文数值字节数 (int64)
        self.server_comm_bytes = 0            # 记录服务器交互通信量
        
        # 初始的聚合梯度为全 0 的密文
        self.enc_global_gradient = [self.public_key.encrypt(0) for _ in range(self.d)]

    def sec_judge(self, enc_g_i):
        """严格复现论文 Figure 5: SecJudge 交互协议"""
        m = len(enc_g_i)
        r_list = [random.randint(1, 5) for _ in range(m)]

        # [S1 执行]: 盲化
        blinded_enc = [enc_x + r for enc_x, r in zip(enc_g_i, r_list)]
        self.server_comm_bytes += m * self.c_bytes  # S1 将 m 个密文发给 S2

        # [S2 执行]: 解密并求平方和，然后加密
        dec_x = [self.csp.private_key.decrypt(x) for x in blinded_enc]
        sum_sq = sum([x**2 for x in dec_x])
        enc_sum_sq = self.public_key.encrypt(sum_sq)
        self.server_comm_bytes += self.c_bytes  # S2 将 1 个密文发给 S1

        # [S1 执行]: 密文去盲化
        # 论文公式: sum = enc_sum_sq - sum(2*r_k*x_k + r_k^2)
        unblind_term = sum([(enc_x * (2 * r)) + (r ** 2) for r, enc_x in zip(r_list, enc_g_i)])
        enc_final_sum = enc_sum_sq - unblind_term
        self.server_comm_bytes += self.c_bytes  # S1 将 1 个去盲化后的密文发给 S2

        # [S2 执行]: 最终解密
        final_sum = self.csp.private_key.decrypt(enc_final_sum)
        self.server_comm_bytes += self.p_bytes  # S2 将 1 个明文判定结果发回给 S1

        return final_sum

    def sec_cos(self, enc_a, enc_b):
        """严格复现论文 Figure 6: SecCos 交互协议 (处理两个加密向量)"""
        m = len(enc_a)
        r_list = [random.randint(1, 3) for _ in range(m)]

        # [S1 执行]: 双重盲化
        blinded_a = [a + r for a, r in zip(enc_a, r_list)]
        blinded_b = [b + r for b, r in zip(enc_b, r_list)]
        self.server_comm_bytes += 2 * m * self.c_bytes  # S1 将 2m 个密文发给 S2

        # [S2 执行]: 解密并计算内积，然后加密
        dec_a = [self.csp.private_key.decrypt(a) for a in blinded_a]
        dec_b = [self.csp.private_key.decrypt(b) for b in blinded_b]
        dot_product = sum([a * b for a, b in zip(dec_a, dec_b)])
        enc_dot = self.public_key.encrypt(dot_product)
        self.server_comm_bytes += self.c_bytes  # S2 将 1 个密文发给 S1

        # [S1 执行]: 密文去盲化
        # 论文展开式: (a+r)(b+r) = ab + ar + br + r^2 => 剔除 ar + br + r^2
        unblind_term = sum([(a * r) + (b * r) + (r ** 2) for r, a, b in zip(r_list, enc_a, enc_b)])
        enc_final_dot = enc_dot - unblind_term
        self.server_comm_bytes += self.c_bytes  # S1 将 1 个密文发给 S2

        # [S2 执行]: 最终解密出 Cosine Similarity
        final_dot = self.csp.private_key.decrypt(enc_final_dot)
        self.server_comm_bytes += self.p_bytes  # S2 发回明文

        return final_dot

    def process_round(self, client_packages):
        """完整执行服务端一轮的判定、余弦相似度计算与拜占庭聚合"""
        time_logs = {'t_secjudge': 0.0, 't_seccos': 0.0, 't_agg': 0.0, 't_dec': 0.0}
        num_clients = len(client_packages)
        
        # 1. 真实执行所有客户端的 SecJudge
        print("    [Server] 正在执行全量 SecJudge...")
        t0 = time.time()
        valid_clients = []
        for i in tqdm(range(num_clients), desc="SecJudge"):
            real_sum = self.sec_judge(client_packages[i])
            # 允许合理的定点数量化误差容限
            if abs(real_sum - self.deg**2) <= (self.deg**2 * 0.1):
                valid_clients.append(client_packages[i])
        time_logs['t_secjudge'] = time.time() - t0

        if not valid_clients:
            print("    [Server] 警告：所有客户端梯度被过滤！")
            valid_clients = client_packages # 兜底保护
            
        num_valid = len(valid_clients)

        # 2. 真实执行与历史聚合梯度的 SecCos
        print("    [Server] 正在执行与全局梯度的 SecCos...")
        t0 = time.time()
        cos_values = []
        for i in tqdm(range(num_valid), desc="SecCos (vs Global)"):
            cos_val = self.sec_cos(valid_clients[i], self.enc_global_gradient)
            cos_values.append(cos_val)
            
        # 寻找恶意基线 g* (最低 cosine)
        min_idx = np.argmin(cos_values)
        enc_g_star = valid_clients[min_idx]

        # 3. 真实执行与恶意基线 g* 的 SecCos 以确定 Confidence
        print("    [Server] 正在执行与恶意基线的 SecCos 计算 Confidence...")
        confidences = []
        for i in tqdm(range(num_valid), desc="SecCos (vs g*)"):
            cos_star = self.sec_cos(valid_clients[i], enc_g_star)
            confidences.append(self.deg - cos_star) # 论文置信度公式
        time_logs['t_seccos'] = time.time() - t0

        # 4. 真实执行密文聚合 (加权)
        print("    [Server] 正在执行全局密文加权聚合...")
        t0 = time.time()
        total_conf = sum(confidences) + 1e-9
        norm_conf = [int((c / total_conf) * self.deg) for c in confidences]
        
        enc_global_update = [self.public_key.encrypt(0) for _ in range(self.d)]
        for j in range(num_valid):
            weight = norm_conf[j]
            for k in range(self.d):
                # 密文标量乘法与加法
                enc_global_update[k] += valid_clients[j][k] * weight
        time_logs['t_agg'] = time.time() - t0

        # 5. 真实执行全局解密
        print("    [Server] 正在执行全局模型解密与更新...")
        t0 = time.time()
        plaintext_update = [(self.csp.private_key.decrypt(enc_x) / (self.deg**2)) for enc_x in enc_global_update]
        self.enc_global_gradient = enc_global_update # 记录为下一轮基准
        time_logs['t_dec'] = time.time() - t0
        
        plaintext_tensor = torch.tensor(plaintext_update, dtype=torch.float32)
        self.global_weights += plaintext_tensor.to(self.device)
        
        return self.global_weights, time_logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet20', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--test_dim', type=int, default=0, help='为了真实跑通代码设置的测试维度，设为 0 则为全维度（将极其耗时）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 启动 ShieldFL [绝对真实执行版] | Device: {device} | Clients: {args.num_clients}")
    
    key_length = 512
    csp = CryptographicServiceProvider(key_length=key_length)
    model_class = resnet20 if args.model == 'ResNet20' else (LeNet5 if args.model == 'LeNet5' else ResNet18_CIFAR10)
    dataset_name = 'CIFAR10' if args.model != 'LeNet5' else 'MNIST'

    client_dataloaders, _ = get_federated_dataloaders(dataset_name, num_clients=args.num_clients, poison_rate=0.0, batch_size=64)
    dummy_model = model_class().to(device)
    
    # 维度处理
    full_d = get_model_vector(dummy_model).size(0)
    d = args.test_dim if args.test_dim > 0 else full_d
    if args.test_dim > 0:
        print(f"[!] 警告: 为避免您等待数小时，当前已截断测试维度至 {d}。如需测试全维度 {full_d} 耗时，请添加参数 --test_dim 0")
    
    server = CloudServer((d,), csp, device=device, key_length=key_length)
    
    # 初始化全局权重 (这里仅截断部分用于实验跑通)
    global_weights_full = get_model_vector(dummy_model).clone()
    server.global_weights = global_weights_full[:d]

    # --- 通信量严格核算 (依据您的需求) ---
    c_bytes = (2 * key_length) // 8
    client_upload_mb = (d * c_bytes) / (1024*1024)
    client_download_mb = (d * c_bytes) / (1024*1024)  # 下载初始/更新的加密模型
    
    print(f"\n>>> 阶段 1: 客户端本地训练与真实同态加密...")
    client_packages = []
    c_train_time, c_scale_time, c_enc_time = 0, 0, 0
    for i in range(args.num_clients):
        client = ShieldFLClient(client_dataloaders[i], model_class, csp.get_public_key(), device)
        # 仅取前 d 维投入实验
        enc_A = client.run_local_training_and_encrypt(global_weights_full, 1)[:d]
        client_packages.append(enc_A)
        c_train_time += client.time_logs['t_train']
        c_scale_time += client.time_logs['t_scale']
        c_enc_time += client.time_logs['t_enc']
        
    print(f">>> 阶段 2: 服务端执行全链路防御与安全聚合操作 (含 tqdm 进度指示)...")
    _, s_time_logs = server.process_round(client_packages)
    
    server_comm_mb = server.server_comm_bytes / (1024*1024)
    
    print("\n" + "="*55)
    print(f"    ShieldFL 全量真实操作测评报告 (Dim={d}, N={args.num_clients})")
    print("="*55)
    print(f" [精确通信量剖析]")
    print(f"  - 单客户端上传密文参数     : {client_upload_mb:.4f} MB")
    print(f"  - 单客户端收到(初始)模型   : {client_download_mb:.4f} MB")
    print(f"  - S1 与 S2 的真实交互通信量: {server_comm_mb:.4f} MB")
    print("-" * 55)
    print(f" [客户端单点耗时 (串行均值)]")
    print(f"  - 本地模型训练   : {c_train_time/args.num_clients:.4f} s")
    print(f"  - 动态稀疏定点化 : {c_scale_time/args.num_clients:.4f} s")
    print(f"  - 全维同态加密   : {c_enc_time/args.num_clients:.4f} s")
    print("-" * 55)
    print(f" [服务端全量真实计算耗时]")
    print(f"  - SecJudge 验证 (所有客户端)   : {s_time_logs['t_secjudge']:.4f} s")
    print(f"  - SecCos 相似度 (含两次遍历计算) : {s_time_logs['t_seccos']:.4f} s")
    print(f"  - 密文加法与标量乘法聚合       : {s_time_logs['t_agg']:.4f} s")
    print(f"  - 全局模型解密还原             : {s_time_logs['t_dec']:.4f} s")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()