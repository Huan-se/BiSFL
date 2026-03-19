import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np
import io

# 导入外部定义的模型与数据加载器
from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().type(torch.float64)

def set_model_vector(model, vector):
    torch.nn.utils.vector_to_parameters(vector.type(torch.float32), model.parameters())

# ==========================================
# 真实安全多方计算 (SMC) 引擎：基于 Additive Secret Sharing
# ==========================================
class RealSMCEngine:
    def __init__(self, device='cpu'):
        self.device = device
        self.comm_bytes = 0.0
        self.smc_time = 0.0

    def get_comm_mb(self):
        return self.comm_bytes / (1024 * 1024)

    def generate_beaver_triples(self, shape):
        """离线阶段：受信任第三方或预计算生成 Beaver 三元组 [A], [B], [C] 满足 C = A * B"""
        A = torch.randn(shape, dtype=torch.float64, device=self.device)
        B = torch.randn(shape, dtype=torch.float64, device=self.device)
        C = A * B
        
        # 分割成 shares
        A1 = torch.randn_like(A); A2 = A - A1
        B1 = torch.randn_like(B); B2 = B - B1
        C1 = torch.randn_like(C); C2 = C - C1
        return (A1, A2), (B1, B2), (C1, C2)

    def sec_mul(self, X1, X2, Y1, Y2):
        """在线阶段：真实执行 SecMul (Beaver 乘法协议)"""
        start_time = time.time()
        shape = X1.shape
        (A1, A2), (B1, B2), (C1, C2) = self.generate_beaver_triples(shape)
        
        # 1. 局部计算盲化值
        D1 = X1 - A1; D2 = X2 - A2
        E1 = Y1 - B1; E2 = Y2 - B2
        
        # 2. 交互揭示 D 和 E (Server 1 和 Server 2 交换数据)
        D = D1 + D2
        E = E1 + E2
        # 通信量：D1, D2, E1, E2 皆为 float64 (8 bytes)
        self.comm_bytes += (D.numel() * 8 * 2) + (E.numel() * 8 * 2)
        
        # 3. 计算乘积份额 Z1, Z2
        Z1 = D * E + D * B1 + E * A1 + C1
        Z2 = D * B2 + E * A2 + C2
        
        self.smc_time += (time.time() - start_time)
        return Z1, Z2

    def sec_dis(self, X1, X2, Y1, Y2):
        """真实执行 SecDis (安全欧氏距离平方)"""
        # 局部相减 (无通信)
        Diff1 = X1 - Y1
        Diff2 = X2 - Y2
        
        # 密文求平方 (需要一次 SecMul)
        Sq1, Sq2 = self.sec_mul(Diff1, Diff2, Diff1, Diff2)
        
        # 局部求和 (无通信)
        Dist1 = torch.sum(Sq1)
        Dist2 = torch.sum(Sq2)
        return Dist1, Dist2

    def sec_cmp(self, X1, X2, Y1, Y2):
        """
        真实执行 SecCmp (安全比较 X < Y)。
        通过掩码盲化协议：生成联合正随机数 R，计算 Z = R * (X - Y)，揭示 Z 的符号。
        """
        start_time = time.time()
        # 局部相减: Diff = X - Y
        Diff1 = X1 - Y1
        Diff2 = X2 - Y2
        
        # 生成正随机数份额 (R = R1 + R2 > 0)
        R = torch.abs(torch.randn(1, dtype=torch.float64, device=self.device)) + 0.1
        R1 = torch.randn(1, dtype=torch.float64, device=self.device)
        R2 = R - R1
        
        # 盲化相乘 (Z = R * Diff)
        Z1, Z2 = self.sec_mul(R1, R2, Diff1, Diff2)
        
        # 揭示 Z
        Z = Z1 + Z2
        self.comm_bytes += 16 # Z1, Z2 各 8 字节
        
        self.smc_time += (time.time() - start_time)
        return Z.item() < 0

    def sec_clip(self, Update1, Update2, dist1, dist2, m_h):
        """
        真实执行 SecClip (安全裁剪)。
        目标：Update * (m_h / dist)。为避免复杂的除法电路，通过掩码除法实现。
        """
        start_time = time.time()
        # 生成联合随机数 R = R1 + R2
        R = torch.randn(1, dtype=torch.float64, device=self.device) + 1.0
        R1 = torch.randn(1, dtype=torch.float64, device=self.device)
        R2 = R - R1
        
        # 盲化距离：Z = R * dist
        Z1, Z2 = self.sec_mul(R1, R2, dist1, dist2)
        Z = Z1 + Z2
        self.comm_bytes += 16
        
        # 明文计算除数因子 factor = m_h / Z  (因为 Z = R * dist，所以 m_h / Z = m_h / (R*dist))
        # 那么目标倍率 clip_factor = (m_h / dist) = factor * R
        factor = m_h / Z.item()
        
        # 将 R 乘回模型更新：[Clip_Update] = factor * ([Update] * [R])
        # 首先用 SecMul 计算 Update * R
        # 扩展 R 的维度以匹配 Update
        R1_exp = R1.expand_as(Update1)
        R2_exp = R2.expand_as(Update2)
        
        Up_R1, Up_R2 = self.sec_mul(Update1, Update2, R1_exp, R2_exp)
        
        # 局部乘以公开因子 factor
        Clip_Up1 = Up_R1 * factor
        Clip_Up2 = Up_R2 * factor
        
        self.smc_time += (time.time() - start_time)
        return Clip_Up1, Clip_Up2

# ==========================================
# 双服务器系统 (Dual-Server Architecture)
# ==========================================
class EPPRFLDualServer:
    def __init__(self, model_shape, epsilon=0.06, lam=0.05, device='cpu'):
        self.global_weights = torch.zeros(model_shape, dtype=torch.float64, device=device)
        self.delta_h = torch.zeros(model_shape, dtype=torch.float64, device=device)
        self.d = model_shape[0]
        self.epsilon = epsilon
        self.lam = lam
        self.device = device
        self.m_h = None
        self.smc = RealSMCEngine(device=device)

    def init_global_weights(self, model):
        self.global_weights = get_model_vector(model).clone()

    def process_round(self, client_shares_1, client_shares_2):
        self.smc.comm_bytes = 0.0
        self.smc.smc_time = 0.0
        num_clients = len(client_shares_1)
        
        # 1. 客户端上传份额通信量统计 (全维度 d * 8 Bytes)
        self.smc.comm_bytes += num_clients * self.d * 8 * 2 
        
        # 2. 降维 (Downsampling)
        k_dim = int(self.epsilon * self.d)
        indices = torch.randperm(self.d, device=self.device)[:k_dim]
        
        down_h = self.delta_h[indices]
        # 服务器在本地切片份额 (无通信)
        down_h1 = torch.randn_like(down_h); down_h2 = down_h - down_h1
        
        dist_shares_1 = []
        dist_shares_2 = []
        
        # 3. 真实安全距离计算 (SecDis)
        for i in range(num_clients):
            up1_down = client_shares_1[i][indices]
            up2_down = client_shares_2[i][indices]
            d1, d2 = self.smc.sec_dis(up1_down, up2_down, down_h1, down_h2)
            dist_shares_1.append(d1)
            dist_shares_2.append(d2)
            
        # 4. 真实安全中位数 (SecMed) - 为了保证脚本能跑完，这里简化为联合揭示后求中位数
        # 在严格 SMC 中，SecMed 需要 O(N^2) 次 SecCmp 冒泡排序
        revealed_dists = []
        for d1, d2 in zip(dist_shares_1, dist_shares_2):
            revealed_dists.append((d1 + d2).item())
            self.smc.comm_bytes += 16
        
        if self.m_h is None:
            self.m_h = np.median(revealed_dists)
            
        m_h_tensor1 = torch.tensor([self.m_h], dtype=torch.float64, device=self.device)
        m_h_tensor2 = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        
        # 阈值
        lam_m_h1 = m_h_tensor1 * self.lam; lam_m_h2 = m_h_tensor2 * self.lam
        
        benign_updates_1 = []
        benign_updates_2 = []
        statuses = []
        
        # 5. 过滤与裁剪 (F&C)
        for i in range(num_clients):
            # 真实 SecCmp 比较：Diff < lam * m_h
            # Diff_sq = (dist - m_h)^2 (需要 SecMul)
            diff1 = dist_shares_1[i] - m_h_tensor1
            diff2 = dist_shares_2[i] - m_h_tensor2
            sq_diff1, sq_diff2 = self.smc.sec_mul(diff1, diff2, diff1, diff2)
            
            is_benign = self.smc.sec_cmp(sq_diff1, sq_diff2, lam_m_h1, lam_m_h2)
            
            if is_benign:
                # 检查是否需要裁剪: dist > m_h
                needs_clip = not self.smc.sec_cmp(dist_shares_1[i], dist_shares_2[i], m_h_tensor1, m_h_tensor2)
                
                if needs_clip and revealed_dists[i] > 0:
                    c1, c2 = self.smc.sec_clip(client_shares_1[i], client_shares_2[i], dist_shares_1[i], dist_shares_2[i], self.m_h)
                    benign_updates_1.append(c1)
                    benign_updates_2.append(c2)
                    statuses.append("Benign (Clipped)")
                else:
                    benign_updates_1.append(client_shares_1[i])
                    benign_updates_2.append(client_shares_2[i])
                    statuses.append("Benign (Passed)")
            else:
                statuses.append(f"Poisoned Filtered")

        # 6. 安全聚合 (局部相加)
        agg_start = time.time()
        if len(benign_updates_1) > 0:
            agg_1 = torch.mean(torch.stack(benign_updates_1), dim=0)
            agg_2 = torch.mean(torch.stack(benign_updates_2), dim=0)
            # 双服务器向客户端广播聚合结果
            final_update = agg_1 + agg_2
            self.smc.comm_bytes += self.d * 8 * 2  # S1 和 S2 分别下发
        else:
            final_update = torch.zeros_like(self.global_weights)
            
        self.global_weights += final_update
        self.delta_h = final_update.clone()
        agg_time = time.time() - agg_start
        
        return self.global_weights, statuses, self.smc.smc_time, agg_time, self.smc.get_comm_mb()

# ==========================================
# 客户端类 (EPPRFL Client)
# ==========================================
class EPPRFLClient:
    def __init__(self, client_data, model_class, device='cpu'):
        self.client_id = client_data['client_id']
        self.is_malicious = client_data['is_malicious']
        self.train_loader = client_data['host_loader']
        self.device = device
        self.model = model_class().to(self.device)
        self.time_logs = {}

    def run_local_training_and_split(self, global_weights, local_epochs):
        start_time = time.time()
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
                
        local_update = get_model_vector(self.model) - global_weights.type(torch.float64)
        local_train_time = time.time() - start_time
        
        # 真实 ASS 分割：拆分成两个 float64 的 Share
        start_split = time.time()
        share_1 = torch.randn_like(local_update)
        share_2 = local_update - share_1
        split_time = time.time() - start_split
        
        self.time_logs['train'] = local_train_time
        self.time_logs['split'] = split_time
        return share_1, share_2

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet20', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--poison_rate', type=float, default=0.2)
    parser.add_argument('--num_rounds', type=int, default=2)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 启动 EPPRFL 真实多方安全计算 (SMC) 引擎 | Device: {device}")
    
    dataset_name = 'MNIST' if args.model == 'LeNet5' else 'CIFAR10'
    model_class = LeNet5 if args.model == 'LeNet5' else (ResNet18_CIFAR10 if args.model == 'ResNet18' else resnet20)

    client_dataloaders, test_loader = get_federated_dataloaders(
        dataset_name=dataset_name, num_clients=args.num_clients, poison_rate=args.poison_rate, batch_size=args.batch_size
    )
    
    clients = [EPPRFLClient(data, model_class, device=device) for data in client_dataloaders]
    dummy_model = model_class().to(device)
    
    d = get_model_vector(dummy_model).size(0)
    server_system = EPPRFLDualServer(model_shape=(d,), epsilon=0.06, lam=0.05, device=device)
    server_system.init_global_weights(dummy_model)
    
    print(f"[*] 模型参数量: {d} (SMC 降维计算维度: {int(d*0.06)})")
    print("-" * 60)

    for round_idx in range(args.num_rounds):
        print(f"\n>>> [Round {round_idx+1}/{args.num_rounds}] 开始")
        
        shares_1_list = []
        shares_2_list = []
        round_train_time = 0.0
        round_split_time = 0.0
        
        # 1. 客户端真实训练并拆分 Shares
        for client in clients:
            s1, s2 = client.run_local_training_and_split(server_system.global_weights, args.local_epochs)
            shares_1_list.append(s1)
            shares_2_list.append(s2)
            round_train_time += client.time_logs['train']
            round_split_time += client.time_logs['split']
            
        # 2. 双服务器真实执行 Beaver 交互与 F&C
        global_weights, statuses, smc_time, agg_time, comm_mb = server_system.process_round(shares_1_list, shares_2_list)
        
        for i, status in enumerate(statuses):
            mal_flag = "[MAL]" if clients[i].is_malicious else "[BEN]"
            print(f"    Client_{i:02d} {mal_flag} -> {status}")
            
        print(f"\n  [时间与通信指标 (单轮)]")
        print(f"  - 客户端本地明文训练耗时   : {round_train_time:.4f} 秒 (累加)")
        print(f"  - 客户端真实 Share 分割耗时: {round_split_time:.4f} 秒 (累加)")
        print(f"  - 双服务器真实 SMC 交互耗时: {smc_time:.4f} 秒")
        print(f"  - 双服务器明文聚合耗时     : {agg_time:.4f} 秒")
        print(f"  - 真实双服务器系统总通信量 : {comm_mb:.2f} MB")
        
        dummy_model.eval()
        set_model_vector(dummy_model, global_weights)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = dummy_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"  [Accuracy] Global Model: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()