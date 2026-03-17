import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import numpy as np

# 导入外部定义的模型与数据加载器
from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

# ==========================================
# 辅助函数：模型参数与向量的转换
# ==========================================
def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def set_model_vector(model, vector):
    torch.nn.utils.vector_to_parameters(vector, model.parameters())

# ==========================================
# 密码学 SMC 开销与通信量模拟器
# 严格对照论文 Table I 和 Table II 的公式
# ==========================================
class SMCSimulator:
    def __init__(self, bit_length=64):
        self.l = bit_length # 秘密共享的位宽 [cite: 410]
        self.comm_bytes = 0.0
    
    def reset_comm(self):
        self.comm_bytes = 0.0

    def add_client_upload_comm(self, d, num_clients):
        # 客户端通信开销: O(d|w|) 
        self.comm_bytes += num_clients * (d * self.l / 8)
    
    def simulate_SecDis(self, k):
        # 计算距离的通信开销: 4kl bits 
        self.comm_bytes += (4 * k * self.l) / 8
        
    def simulate_SecMed(self, n):
        # 计算中位数的通信开销: 3k^2l - 3kl bits, 这里 k=n 
        self.comm_bytes += (3 * (n**2) * self.l - 3 * n * self.l) / 8
        
    def simulate_SecCmp(self, count):
        # 安全比较的通信开销: 6l bits 每次 
        self.comm_bytes += count * (6 * self.l) / 8
        
    def simulate_SecClip(self, k):
        # 安全裁剪的通信开销: 8kl + 4l bits 
        self.comm_bytes += (8 * k * self.l + 4 * self.l) / 8

# ==========================================
# 客户端类 (EPPRFL Client)
# ==========================================
class EPPRFLClient:
    def __init__(self, client_data, model_class, device='cpu'):
        self.client_id = client_data['client_id']
        self.is_malicious = client_data['is_malicious']
        self.train_loader = client_data['host_loader'] # 论文中客户端使用全量本地数据
        self.device = device
        self.model = model_class().to(self.device)
        self.time_logs = {}

    def run_local_training(self, global_weights, local_epochs):
        """执行本地训练并模拟加法秘密共享 (ASS) 的分割耗时"""
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
                
        local_update = get_model_vector(self.model) - global_weights
        
        # 模拟生成随机数掩码并拆分为两个 shares [cite: 617-620]
        torch.manual_seed(self.client_id)
        L_i = torch.randn_like(local_update)
        share_1 = L_i
        share_2 = local_update - L_i
        
        self.time_logs['local_train_and_split'] = time.time() - start_time
        return share_1, share_2, local_update # 仅供模拟器明文校验使用，真实服务端不可见

# ==========================================
# 双服务器模拟类 (Dual-Server F&C Logic)
# ==========================================
class EPPRFLServerSystem:
    def __init__(self, model_shape, epsilon=0.06, lam=0.05, device='cpu'):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.d = model_shape[0]
        self.epsilon = epsilon # 降维率 [cite: 481, 882]
        self.lam = lam         # 过滤超参数 [cite: 481, 882]
        self.device = device
        
        # 初始化历史状态 [cite: 481, 642]
        self.delta_h = torch.zeros(model_shape).to(device)
        self.m_h = None
        
        self.smc_sim = SMCSimulator()

    def init_global_weights(self, model):
        self.global_weights = get_model_vector(model).clone()
        
    def run_detection_and_aggregation(self, client_updates_plaintext, num_clients):
        """
        模拟双服务器联合执行 F&C 的逻辑与耗时。
        此处我们利用明文直接计算结果，但时间复杂度和通信开销通过 smc_sim 严格统计。
        """
        start_detect = time.time()
        self.smc_sim.reset_comm()
        
        # 1. Update Downsampling 降维 [cite: 484-485, 643-647]
        k_dim = int(self.epsilon * self.d)
        indices = torch.randperm(self.d, device=self.device)[:k_dim]
        
        distances = []
        # 2. Distance Computation 距离计算 [cite: 486-487, 648-653]
        for update in client_updates_plaintext:
            downsampled_update = update[indices]
            downsampled_h = self.delta_h[indices]
            
            # 模拟 SecDis [cite: 649]
            dist = torch.sum((downsampled_update - downsampled_h) ** 2).item()
            distances.append(dist)
            self.smc_sim.simulate_SecDis(k_dim)
            
        # 计算当前的中间信息用于历史更新 [cite: 487, 650]
        if self.m_h is None:
            self.m_h = np.median(distances)
            self.smc_sim.simulate_SecMed(num_clients)
            
        benign_updates = []
        statuses = []
        
        # 3. Benign Update Filtering 过滤 [cite: 488, 654-657]
        self.smc_sim.simulate_SecCmp(num_clients)
        for i, dist in enumerate(distances):
            diff = (dist - self.m_h) ** 2
            if diff < self.lam * self.m_h:
                # 4. Benign Update Clipping 裁剪 [cite: 489, 658-660]
                self.smc_sim.simulate_SecCmp(1)
                if dist > self.m_h and dist > 0:
                    clip_factor = self.m_h / dist
                    clipped_update = client_updates_plaintext[i] * clip_factor
                    self.smc_sim.simulate_SecClip(self.d)
                    benign_updates.append(clipped_update)
                    statuses.append("Benign (Clipped)")
                else:
                    benign_updates.append(client_updates_plaintext[i])
                    statuses.append("Benign (Passed)")
            else:
                statuses.append(f"Poisoned Filtered (Diff: {diff:.4f})")
                
        detect_time = time.time() - start_detect
        
        # 5. Benign Update Aggregation 聚合 [cite: 492-496, 661-663]
        start_agg = time.time()
        if len(benign_updates) > 0:
            agg_update = torch.mean(torch.stack(benign_updates), dim=0)
        else:
            agg_update = torch.zeros_like(self.global_weights)
            
        self.global_weights += agg_update
        agg_time = time.time() - start_agg
        
        # 6. History Updating 历史更新 [cite: 497-526, 664-672]
        self.delta_h = agg_update.clone() # 简化的历史聚合逻辑
        
        # 统计客户端上传与服务端下发通信 
        self.smc_sim.add_client_upload_comm(self.d, num_clients)
        total_comm_mb = self.smc_sim.comm_bytes / (1024 * 1024)
        
        return self.global_weights, statuses, detect_time, agg_time, total_comm_mb

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="EPPRFL Execution")
    parser.add_argument('--model', type=str, default='LeNet5', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--poison_rate', type=float, default=0.2)
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] SMC Platform Initializing... Device: {device}")
    init_start = time.time()

    # 模型映射与数据集绑定
    if args.model == 'LeNet5':
        dataset_name = 'MNIST'
        model_class = LeNet5
    else:
        dataset_name = 'CIFAR10'
        model_class = ResNet18_CIFAR10 if args.model == 'ResNet18' else resnet20
        
    print(f"[*] Task: {args.model} on {dataset_name} | Clients: {args.num_clients} | Poison Rate: {args.poison_rate*100}%")

    client_dataloaders, test_loader = get_federated_dataloaders(
        dataset_name=dataset_name, num_clients=args.num_clients, poison_rate=args.poison_rate, batch_size=args.batch_size
    )
    
    clients = [EPPRFLClient(data, model_class, device=device) for data in client_dataloaders]
    dummy_model = model_class().to(device)
    
    server_system = EPPRFLServerSystem(
        model_shape=get_model_vector(dummy_model).size(), 
        epsilon=0.06, # 论文推荐降维参数 [cite: 882]
        lam=0.05,     # 论文推荐过滤参数 [cite: 882]
        device=device
    )
    server_system.init_global_weights(dummy_model)
    
    print(f"[*] Initialization Time: {time.time() - init_start:.2f} s\n" + "-"*50)

    for round_idx in range(args.num_rounds):
        print(f"\n>>> [Round {round_idx+1}/{args.num_rounds}] Started")
        
        client_updates_plaintext = []
        round_local_train_time = 0.0
        
        # 1. 客户端训练与秘密分割 (ASS)
        for client in clients:
            _, _, local_update = client.run_local_training(server_system.global_weights, args.local_epochs)
            client_updates_plaintext.append(local_update)
            round_local_train_time += client.time_logs['local_train_and_split']
            
        # 2. 双服务器进行异常检测、裁剪与聚合
        global_weights, statuses, detect_time, agg_time, comm_mb = server_system.run_detection_and_aggregation(
            client_updates_plaintext, args.num_clients
        )
        
        for i, status in enumerate(statuses):
            mal_flag = "[MAL]" if clients[i].is_malicious else "[BEN]"
            print(f"    Client_{i:02d} {mal_flag} -> {status}")
            
        # 3. 指标汇总与验证
        print(f"  [Metrics] Total Local Train Time  : {round_local_train_time:.4f} s")
        print(f"  [Metrics] F&C Detect & Mask Time  : {detect_time:.4f} s")
        print(f"  [Metrics] SMC Aggregation Time    : {agg_time:.4f} s")
        print(f"  [Metrics] Server SMC Comm Payload : {comm_mb:.4f} MB")
        
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