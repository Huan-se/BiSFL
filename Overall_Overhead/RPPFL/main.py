import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

# 导入外部定义的模型与数据加载器
from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

# ==========================================
# 辅助函数：模型参数与向量的转换
# ==========================================
def get_model_vector(model):
    """将模型所有参数展平为一个一维向量"""
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def set_model_vector(model, vector):
    """将一维向量还原为模型参数"""
    torch.nn.utils.vector_to_parameters(vector, model.parameters())

def get_model_size_mb(model):
    """计算模型参数所占的通信量 (MB)"""
    param_size = sum(p.numel() for p in model.parameters())
    return param_size * 4 / (1024 * 1024) # 假设 float32 占 4 bytes

# ==========================================
# 客户端类 (RPPFL Client)
# ==========================================
class RPPFLClient:
    def __init__(self, client_data, model_class, device='cpu'):
        self.client_id = client_data['client_id']
        self.is_malicious = client_data['is_malicious']
        self.host_loader = client_data['host_loader']
        self.enclave_loader = client_data['enclave_loader']
        self.m_host = client_data['m_host']
        self.m_enclave = client_data['m_enclave']
        self.device = device
        
        # 实例化模型
        self.model_host = model_class().to(self.device)
        self.model_enclave = model_class().to(self.device)
        self.time_logs = {}

    def local_training(self, model, dataloader, epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        for _ in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def run_host_worker(self, global_weights, local_epochs):
        start_time = time.time()
        set_model_vector(self.model_host, global_weights)
        self.local_training(self.model_host, self.host_loader, epochs=local_epochs)
        theta_F = get_model_vector(self.model_host)
        self.time_logs['host_train'] = time.time() - start_time
        return theta_F

    def run_enclave_worker(self, global_weights, theta_F, round_idx, local_epochs, threshold):
        # 1. 可信子集轻量训练
        start_train = time.time()
        set_model_vector(self.model_enclave, global_weights)
        self.local_training(self.model_enclave, self.enclave_loader, epochs=local_epochs)
        theta_T = get_model_vector(self.model_enclave)
        self.time_logs['enclave_train'] = time.time() - start_train
        
        # 2. 投毒检测与混淆掩码 (合并计时为 detect_time)
        start_detect_mask = time.time()
        
        # 距离计算
        distance = torch.norm(theta_F - theta_T, p=2).item()
        
        # 异常检测：判定是否拦截
        if distance <= threshold:
            selected_theta = theta_F
            weight_m = self.m_host
            status = "Benign"
        else:
            selected_theta = theta_T
            weight_m = self.m_enclave
            status = f"Poisoned (Dist: {distance:.4f})"
            
        # 伪随机掩码混淆
        torch.manual_seed(round_idx * 100 + self.client_id)
        mask = torch.randn_like(selected_theta)
        masked_theta = selected_theta + (mask / weight_m)
        
        self.time_logs['detect_mask'] = time.time() - start_detect_mask
        return masked_theta, weight_m, mask, status

    def execute_round(self, global_weights, round_idx, local_epochs, threshold):
        theta_F = self.run_host_worker(global_weights, local_epochs)
        masked_theta, weight_m, mask, status = self.run_enclave_worker(global_weights, theta_F, round_idx, local_epochs, threshold)
        return masked_theta, weight_m, mask, status

# ==========================================
# 服务器类 (RPPFL Server)
# ==========================================
class RPPFLServer:
    def __init__(self, model_shape, device='cpu'):
        self.global_weights = torch.zeros(model_shape).to(device)
        self.device = device
        
    def init_global_weights(self, model):
        self.global_weights = get_model_vector(model).clone()
        
    def aggregate(self, client_updates, total_m, total_mask):
        aggregated_theta = torch.zeros_like(self.global_weights)
        for masked_theta, m_i in client_updates:
            aggregated_theta += (m_i / total_m) * masked_theta
        # 服务器端直接扣除聚合后的总掩码（利用伪随机同态相加抵消的性质）
        aggregated_theta -= (total_mask / total_m)
        self.global_weights = aggregated_theta
        return self.global_weights

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="RPPFL Execution")
    parser.add_argument('--model', type=str, default='LeNet5', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--poison_rate', type=float, default=0.2, help='Ratio of malicious clients')
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.006, help='L2 distance threshold for anomaly detection')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Platform Initializing... Device: {device}")
    
    init_start = time.time()

    # 模型映射与数据集绑定
    if args.model == 'LeNet5':
        dataset_name = 'MNIST'
        model_class = LeNet5
    elif args.model == 'ResNet18':
        dataset_name = 'CIFAR10'
        model_class = ResNet18_CIFAR10
    elif args.model == 'ResNet20':
        dataset_name = 'CIFAR10'
        model_class = resnet20
        
    print(f"[*] Task: {args.model} on {dataset_name} | Clients: {args.num_clients} | Poison Rate: {args.poison_rate*100}%")

    # 获取分离的数据加载器
    client_dataloaders, test_loader = get_federated_dataloaders(
        dataset_name=dataset_name, 
        num_clients=args.num_clients, 
        poison_rate=args.poison_rate, 
        batch_size=args.batch_size
    )
    
    # 实例化对象
    clients = [RPPFLClient(data, model_class, device=device) for data in client_dataloaders]
    dummy_model = model_class().to(device)
    server = RPPFLServer(get_model_vector(dummy_model).size(), device=device)
    server.init_global_weights(dummy_model)
    
    # 计算通信开销
    model_size_mb = get_model_size_mb(dummy_model)
    comm_volume_per_round = args.num_clients * model_size_mb * 2 # Client 上传 + Server 下发
    
    init_time = time.time() - init_start
    print(f"[*] Initialization Time: {init_time:.2f} s")
    print(f"[*] Comm Overhead per Round (Up+Down): {comm_volume_per_round:.2f} MB")
    print("--------------------------------------------------")

    # 开始联邦训练
    for round_idx in range(args.num_rounds):
        print(f"\n>>> [Round {round_idx+1}/{args.num_rounds}] Started")
        client_updates = []
        total_m = 0
        total_mask = torch.zeros_like(server.global_weights)
        
        round_host_train_time = 0.0
        round_enclave_train_time = 0.0
        round_detect_time = 0.0
        
        # 客户端并行阶段 (这里串行模拟以累加耗时)
        for client in clients:
            masked_theta, weight_m, mask, status = client.execute_round(
                server.global_weights, round_idx, args.local_epochs, args.threshold
            )
            client_updates.append((masked_theta, weight_m))
            total_m += weight_m
            total_mask += mask
            
            # 累加统计信息
            round_host_train_time += client.time_logs['host_train']
            round_enclave_train_time += client.time_logs['enclave_train']
            round_detect_time += client.time_logs['detect_mask']
            
            mal_flag = "[MAL]" if client.is_malicious else "[BEN]"
            print(f"    Client_{client.client_id:02d} {mal_flag} -> {status}")
            
        # 服务器聚合阶段
        agg_start = time.time()
        global_weights = server.aggregate(client_updates, total_m, total_mask)
        round_agg_time = time.time() - agg_start
        
        # 指标汇总报告
        print(f"  [Metrics] Total Host Train Time   : {round_host_train_time:.4f} s")
        print(f"  [Metrics] Total Enclave Train Time: {round_enclave_train_time:.4f} s")
        print(f"  [Metrics] Total Detect & Mask Time: {round_detect_time:.4f} s")
        print(f"  [Metrics] Server Aggregation Time : {round_agg_time:.4f} s")
        
        # 测试精度
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