import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import random
import math
from dataclasses import dataclass
from phe.util import getprimeover, invert
from tqdm import tqdm

from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def set_model_vector(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())


@dataclass(frozen=True)
class DoubleTrapdoorCiphertext:
    A: int
    B: int


class DoubleTrapdoorPublicKey:
    def __init__(self, n, h):
        self.n = n
        self.nsquare = n * n
        self.g = n + 1
        self.h = h

    def encode(self, value):
        return value % self.n

    def decode(self, value):
        if value > self.n // 2:
            return value - self.n
        return value

    def encrypt(self, value, r=None):
        encoded = self.encode(int(value))
        if r is None:
            r = random.SystemRandom().randrange(1, self.n)

        A = pow(self.g, r, self.nsquare)
        B = (pow(self.h, r, self.nsquare) * (1 + encoded * self.n)) % self.nsquare
        return DoubleTrapdoorCiphertext(A, B)

    def encrypt_zero(self):
        return self.encrypt(0)

    def add(self, left, right):
        return DoubleTrapdoorCiphertext(
            (left.A * right.A) % self.nsquare,
            (left.B * right.B) % self.nsquare,
        )

    def add_plain(self, cipher, value):
        encoded = self.encode(int(value))
        return DoubleTrapdoorCiphertext(
            cipher.A,
            (cipher.B * (1 + encoded * self.n)) % self.nsquare,
        )

    def negate(self, cipher):
        return DoubleTrapdoorCiphertext(
            invert(cipher.A, self.nsquare),
            invert(cipher.B, self.nsquare),
        )

    def sub(self, left, right):
        return self.add(left, self.negate(right))

    def scalar_mul(self, cipher, scalar):
        scalar = int(scalar)
        if scalar == 0:
            return DoubleTrapdoorCiphertext(1, 1)
        if scalar < 0:
            return self.scalar_mul(self.negate(cipher), -scalar)
        return DoubleTrapdoorCiphertext(
            pow(cipher.A, scalar, self.nsquare),
            pow(cipher.B, scalar, self.nsquare),
        )


class DoubleTrapdoorKeyShares:
    def __init__(self, key_length=1024):
        half_bits = key_length // 2
        p = getprimeover(half_bits)
        q = getprimeover(half_bits)
        while p == q:
            q = getprimeover(half_bits)

        n = p * q
        master_secret = random.SystemRandom().randrange(1, n)
        server_share = random.SystemRandom().randrange(1, n)
        csp_share = (master_secret - server_share) % n

        public_key = DoubleTrapdoorPublicKey(n=n, h=pow(n + 1, master_secret, n * n))

        self.public_key = public_key
        self.server_share = server_share
        self.csp_share = csp_share

    def partial_decrypt(self, cipher, share):
        factor = pow(cipher.A, share, self.public_key.nsquare)
        reduced = (cipher.B * invert(factor, self.public_key.nsquare)) % self.public_key.nsquare
        return DoubleTrapdoorCiphertext(cipher.A, reduced)

    def final_decrypt(self, partially_decrypted_cipher, share):
        factor = pow(partially_decrypted_cipher.A, share, self.public_key.nsquare)
        masked = (partially_decrypted_cipher.B * invert(factor, self.public_key.nsquare)) % self.public_key.nsquare
        message = ((masked - 1) // self.public_key.n) % self.public_key.n
        return self.public_key.decode(message)

# ==========================================
# 密码服务提供商 (CSP) 
# ==========================================
class CryptographicServiceProvider:
    def __init__(self, key_length=1024):
        # 按照论文思路，将主密钥拆分为 S1/S2 两份，所有解密都走双陷门串行链路。
        self.key_shares = DoubleTrapdoorKeyShares(key_length=key_length)
        self.public_key = self.key_shares.public_key
        self.server_key_share = self.key_shares.server_share
        self.csp_key_share = self.key_shares.csp_share

    def get_public_key(self):
        return self.public_key

    def get_server_key_share(self):
        return self.server_key_share

    def finalize_partial_cipher(self, partially_decrypted_cipher):
        return self.key_shares.final_decrypt(partially_decrypted_cipher, self.csp_key_share)

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
    def __init__(self, model_shape, csp, deg=100000, device='cpu', key_length=1024):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.csp = csp
        self.public_key = csp.get_public_key()
        self.server_key_share = csp.get_server_key_share()
        self.device = device
        self.d = model_shape[0]
        self.deg = deg
        
        # 通信量统计参数
        self.component_bytes = (2 * key_length) // 8
        self.cipher_bytes = 2 * self.component_bytes  # 双陷门密文由 (A, B) 两个模 N^2 分量组成
        self.p_bytes = 8
        self.server_comm_bytes = 0
        
        # 初始的聚合梯度为全 0 的密文
        self.enc_global_gradient = [self.public_key.encrypt_zero() for _ in range(self.d)]

    def _partial_decrypt_vector(self, enc_vector):
        return [
            self.csp.key_shares.partial_decrypt(cipher, self.server_key_share)
            for cipher in enc_vector
        ]

    def _serial_decrypt_vector(self, enc_vector):
        partially_decrypted = self._partial_decrypt_vector(enc_vector)
        self.server_comm_bytes += len(partially_decrypted) * self.cipher_bytes
        return [self.csp.finalize_partial_cipher(cipher) for cipher in partially_decrypted]

    def _serial_decrypt_cipher(self, enc_cipher):
        partially_decrypted = self.csp.key_shares.partial_decrypt(enc_cipher, self.server_key_share)
        self.server_comm_bytes += self.cipher_bytes
        return self.csp.finalize_partial_cipher(partially_decrypted)

    def sec_judge(self, enc_g_i):
        """严格复现论文 Figure 5: SecJudge 交互协议"""
        m = len(enc_g_i)
        r_list = [random.randint(1, 5) for _ in range(m)]

        # [S1 执行]: 盲化
        blinded_enc = [self.public_key.add_plain(enc_x, r) for enc_x, r in zip(enc_g_i, r_list)]

        # [S1 -> S2 串行双陷门]: S1 部分解密，S2 最终解密
        dec_x = self._serial_decrypt_vector(blinded_enc)
        sum_sq = sum(x ** 2 for x in dec_x)
        enc_sum_sq = self.public_key.encrypt(sum_sq)
        self.server_comm_bytes += self.cipher_bytes

        # [S1 执行]: 密文去盲化
        # 论文公式: sum = enc_sum_sq - sum(2*r_k*x_k + r_k^2)
        unblind_term = self.public_key.encrypt_zero()
        for r, enc_x in zip(r_list, enc_g_i):
            scaled = self.public_key.scalar_mul(enc_x, 2 * r)
            corrected = self.public_key.add_plain(scaled, r ** 2)
            unblind_term = self.public_key.add(unblind_term, corrected)
        enc_final_sum = self.public_key.sub(enc_sum_sq, unblind_term)

        # [S1 -> S2 串行双陷门]: 对最终结果再次走部分解密 + 最终解密
        final_sum = self._serial_decrypt_cipher(enc_final_sum)
        self.server_comm_bytes += self.p_bytes

        return final_sum

    def sec_cos(self, enc_a, enc_b):
        """严格复现论文 Figure 6: SecCos 交互协议 (处理两个加密向量)"""
        m = len(enc_a)
        r_list = [random.randint(1, 3) for _ in range(m)]

        # [S1 执行]: 双重盲化
        blinded_a = [self.public_key.add_plain(a, r) for a, r in zip(enc_a, r_list)]
        blinded_b = [self.public_key.add_plain(b, r) for b, r in zip(enc_b, r_list)]

        # [S1 -> S2 串行双陷门]: 两个向量都必须经过部分解密再最终解密
        dec_a = self._serial_decrypt_vector(blinded_a)
        dec_b = self._serial_decrypt_vector(blinded_b)
        dot_product = sum(a * b for a, b in zip(dec_a, dec_b))
        enc_dot = self.public_key.encrypt(dot_product)
        self.server_comm_bytes += self.cipher_bytes

        # [S1 执行]: 密文去盲化
        # 论文展开式: (a+r)(b+r) = ab + ar + br + r^2 => 剔除 ar + br + r^2
        unblind_term = self.public_key.encrypt_zero()
        for r, a, b in zip(r_list, enc_a, enc_b):
            ar_term = self.public_key.scalar_mul(a, r)
            br_term = self.public_key.scalar_mul(b, r)
            mixed = self.public_key.add(ar_term, br_term)
            corrected = self.public_key.add_plain(mixed, r ** 2)
            unblind_term = self.public_key.add(unblind_term, corrected)
        enc_final_dot = self.public_key.sub(enc_dot, unblind_term)

        # [S1 -> S2 串行双陷门]: 最终相似度结果也必须串行解密
        final_dot = self._serial_decrypt_cipher(enc_final_dot)
        self.server_comm_bytes += self.p_bytes

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
        
        enc_global_update = [self.public_key.encrypt_zero() for _ in range(self.d)]
        for j in range(num_valid):
            weight = norm_conf[j]
            for k in range(self.d):
                weighted_cipher = self.public_key.scalar_mul(valid_clients[j][k], weight)
                enc_global_update[k] = self.public_key.add(enc_global_update[k], weighted_cipher)
        time_logs['t_agg'] = time.time() - t0

        # 5. 真实执行全局解密
        print("    [Server] 正在执行全局模型解密与更新...")
        t0 = time.time()
        plaintext_update = [self._serial_decrypt_cipher(enc_x) / (self.deg ** 2) for enc_x in enc_global_update]
        self.server_comm_bytes += self.d * self.p_bytes
        self.enc_global_gradient = enc_global_update
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
    
    key_length = 1024    #这里应该会影响性能,密钥长度
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
    server.global_weights = global_weights_full[:d].clone()

    # --- 通信量严格核算 (依据您的需求) ---
    client_upload_mb = (d * server.cipher_bytes) / (1024 * 1024)
    client_download_mb = (d * 4) / (1024 * 1024)  # 当前实现中下发的是 float32 明文全局模型
    
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
    print(f"  - 单客户端收到(明文)模型   : {client_download_mb:.4f} MB")
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
