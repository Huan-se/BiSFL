import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import csv

# --- 确保能正确导入 OURS 目录下的原生模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from _utils_.tee_adapter import get_tee_adapter_singleton
    from Defence.layers_proj_detect import Layers_Proj_Detector
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("👉 请确保你在 'Performance_Evaluation/OURS/' 目录下运行此脚本！")
    sys.exit(1)

# --- 定义模拟数据集以消除磁盘 I/O 干扰 ---
class DummyDataset(data.Dataset):
    def __init__(self, shape, num_classes, num_samples):
        self.data = torch.randn(num_samples, *shape)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        return self.data[idx], self.targets[idx]

def flatten_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).cpu().numpy())
    return np.concatenate(params).astype(np.float32)

def get_model(model_name):
    """动态加载模型，防止部分环境缺失特定模型文件"""
    try:
        if model_name == "LeNet5":
            from model.Lenet5 import LeNet5
            return LeNet5()
        elif model_name == "ResNet20":
            from model.Resnet20 import resnet20
            return resnet20()
        elif model_name == "ResNet18":
            from model.Resnet18 import ResNet18_CIFAR10
            return ResNet18_CIFAR10()
    except ImportError as e:
        print(f"\n❌ 缺少模型文件: {e}")
        print(f"👉 请确保从 Effect_Evaluation/model/ 中将 {model_name}.py 复制到了当前的 model/ 文件夹下！")
        sys.exit(1)

def run_benchmark():
    print("===================================================================")
    print("🚀 启动时间扩展因子评估 (Time Extension Factor Benchmark)")
    print("===================================================================\n")

    # --- 初始化 TEE ---
    tee_adapter = get_tee_adapter_singleton()
    try:
        tee_adapter.initialize_enclave()
    except Exception as e:
        print(f"❌ Enclave 初始化失败: {e}。请检查 libtee.so 是否已编译。")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] 本地训练仿真使用硬件: {device.type.upper()}\n")

    # --- 测试配置网格 ---
    # samples_per_client: 模拟每个客户端的本地数据集大小 (如 500 条)
    # N_clients: 模拟 Server 端并发检测的客户端总数
    configs = [
        {"name": "LeNet5", "shape": (1, 28, 28), "classes": 10, "samples_per_client": 500, "batch_size": 32, "N_clients": 100},
        {"name": "ResNet20", "shape": (3, 32, 32), "classes": 10, "samples_per_client": 500, "batch_size": 32, "N_clients": 100},
        {"name": "ResNet18", "shape": (3, 32, 32), "classes": 10, "samples_per_client": 500, "batch_size": 32, "N_clients": 100},
    ]

    results = []
    num_trials = 10  # 测时求平均的轮数

    for cfg in configs:
        model_name = cfg["name"]
        print(f"➤ 正在评估网络架构: {model_name} ...")
        
        model = get_model(model_name).to(device)
        dataset = DummyDataset(cfg["shape"], cfg["classes"], cfg["samples_per_client"])
        dataloader = data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # -----------------------------------------------------------------
        # 1. 测量本地训练时间 (T_train) - 1个 Epoch
        # -----------------------------------------------------------------
        model.train()
        # 预热 GPU
        for d, t in dataloader:
            d, t = d.to(device), t.to(device)
            optimizer.zero_grad()
            criterion(model(d), t).backward()
            optimizer.step()
            break
            
        total_train_time = 0.0
        for _ in range(num_trials):
            t_start = time.perf_counter()
            for d, t in dataloader:
                d, t = d.to(device), t.to(device)
                optimizer.zero_grad()
                criterion(model(d), t).backward()
                optimizer.step()
            total_train_time += (time.perf_counter() - t_start)
        
        t_train_ms = (total_train_time / num_trials) * 1000.0

        # -----------------------------------------------------------------
        # 2. 测量 TEE 投影时间 (T_proj)
        # -----------------------------------------------------------------
        w_new_flat = flatten_params(model)
        w_old_cache = np.zeros_like(w_new_flat)
        P_size = len(w_new_flat)
        
        # TEE 预热
        _, _ = tee_adapter.prepare_gradient(1, 42, w_new_flat, w_old_cache)
        
        total_proj_time = 0.0
        for _ in range(num_trials):
            t_start = time.perf_counter()
            output_proj, _ = tee_adapter.prepare_gradient(1, 42, w_new_flat, w_old_cache)
            total_proj_time += (time.perf_counter() - t_start)
            
        t_proj_ms = (total_proj_time / num_trials) * 1000.0
        K_dim = len(output_proj)

        # -----------------------------------------------------------------
        # 3. 测量 Server 检测时间 (T_detect)
        # -----------------------------------------------------------------
        N = cfg["N_clients"]
        client_projections = {}
        for cid in range(1, N + 1):
            noisy_proj = output_proj + np.random.normal(0, 0.001, K_dim).astype(np.float32)
            client_projections[cid] = {'full': torch.from_numpy(noisy_proj)}
            
        global_direction = {'full': torch.from_numpy(output_proj)}
        detector = Layers_Proj_Detector(config={'target_layers': []})
        
        total_detect_time = 0.0
        for _ in range(num_trials):
            t_start = time.perf_counter()
            detector.detect(client_projections, global_direction, {}, verbose=False)
            total_detect_time += (time.perf_counter() - t_start)
            
        t_detect_ms = (total_detect_time / num_trials) * 1000.0

        # -----------------------------------------------------------------
        # 4. 计算扩展因子 (Overhead Factor)
        # -----------------------------------------------------------------
        factor_percent = ((t_proj_ms + t_detect_ms) / t_train_ms) * 100.0
        
        results.append({
            "Network": model_name,
            "Params_P": P_size,
            "T_Train_ms": t_train_ms,
            "T_Proj_ms": t_proj_ms,
            "T_Detect_ms": t_detect_ms,
            "Overhead_Factor_%": factor_percent
        })

    # --- 打印报表与落盘 ---
    print("\n✅ 所有网络架构评估完毕！")
    csv_file = "time_extension_factor_benchmark.csv"
    headers = ["Network", "Params_P", "T_Train_ms", "T_Proj_ms", "T_Detect_ms", "Overhead_Factor_%"]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n| {'Network':<10} | {'Params (P)':<12} | {'T_train (ms)':<14} | {'T_proj (ms)':<13} | {'T_detect (ms)':<15} | {'Overhead (%)':<12} |")
    print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*16 + "|" + "-"*15 + "|" + "-"*17 + "|" + "-"*14 + "|")
    
    for r in results:
        print(f"| {r['Network']:<10} | {r['Params_P']:<12} | {r['T_Train_ms']:<14.2f} | {r['T_Proj_ms']:<13.2f} | {r['T_Detect_ms']:<15.2f} | {r['Overhead_Factor_%']:<11.4f}% |")

if __name__ == "__main__":
    run_benchmark()