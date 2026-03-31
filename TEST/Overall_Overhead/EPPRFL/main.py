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
# 辅助函数
# ==========================================
def get_model_vector(model):
    """将模型所有参数展平为一个一维向量"""
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# ==========================================
# 时间测试核心引擎 (EPPRFL 协议纯数学仿真)
# ==========================================
class EPPRFL_Timing_Simulator:
    def __init__(self, d, num_clients=10, epsilon=0.06, lam=0.05, device='cpu'):
        self.num_clients = num_clients
        self.d = d
        self.epsilon = epsilon
        self.sampled_d = int(d * epsilon)
        self.lam = lam
        self.device = device
        self.deg = 100000.0 # 定点数放缩因子

    def generate_beaver_triple(self, shape):
        """离线阶段：生成 Beaver 三元组（不计入在线耗时）"""
        A = torch.randn(shape, dtype=torch.float64, device=self.device)
        B = torch.randn(shape, dtype=torch.float64, device=self.device)
        C = A * B
        return (A/2, A/2), (B/2, B/2), (C/2, C/2)

    def sec_mul(self, X1, X2, Y1, Y2):
        """
        在线安全乘法 (SecMul)
        【修复】: 包含了 Beaver 三元组运算，并加入了真实的定点数截断 (TruncPr) 的 CPU 数学开销。
        """
        (A1, A2), (B1, B2), (C1, C2) = self.generate_beaver_triple(X1.shape)
        D1 = X1 - A1; D2 = X2 - A2
        E1 = Y1 - B1; E2 = Y2 - B2
        
        # 纯数学运算，不模拟网络延迟
        D = D1 + D2
        E = E1 + E2
        
        Z1 = C1 + D * B1 + E * A1
        Z2 = C2 + D * B2 + E * A2 + D * E
        
        # ---------------------------------------------------------
        # 【密码学修复】: 定点数乘法后产生的 TruncPr 截断纯数学开销
        # 在 ASS 中，两个定点数相乘结果会膨胀，必须生成随机掩码并进行除法截断
        # 这里使用 Tensor 运算模拟该过程耗费的 CPU 时钟周期
        # ---------------------------------------------------------
        mask1 = torch.randn_like(Z1)
        mask2 = torch.randn_like(Z2)
        Z1_trunc = (Z1 + mask1) / self.deg - mask1
        Z2_trunc = (Z2 + mask2) / self.deg - mask2
        
        return Z1_trunc, Z2_trunc

    def sec_cmp(self, X1, X2, Y1, Y2):
        """安全比较 (SecCmp)"""
        Diff1 = X1 - Y1
        Diff2 = X2 - Y2
        R = torch.abs(torch.randn(1, dtype=torch.float64, device=self.device)) + 0.1
        R1 = R / 2; R2 = R / 2
        Z1, Z2 = self.sec_mul(R1, R2, Diff1, Diff2)
        return (Z1 + Z2).item() < 0

    def sec_dis(self, X1, X2, Y1, Y2):
        """安全距离计算 (SecDis)"""
        Diff1 = X1 - Y1
        Diff2 = X2 - Y2
        Sq1, Sq2 = self.sec_mul(Diff1, Diff2, Diff1, Diff2)
        return torch.sum(Sq1), torch.sum(Sq2)

    def sec_clip(self, Up1, Up2, dist1, dist2, m_h1, m_h2):
        """
        安全裁剪 (SecClip)
        【修复】: 废弃占位符，严谨实现了论文提及的通过泰勒展开/牛顿迭代逼近计算除法的完整循环计算量。
        """
        # 使用牛顿-拉弗森迭代法 (Newton-Raphson) 逼近 1 / dist
        # 迭代公式: y_{n+1} = y_n * (2 - dist * y_n)
        # 一般在密文下需要迭代 6 次左右才能达到理想精度
        
        y1 = torch.ones_like(dist1) * 0.01
        y2 = torch.ones_like(dist2) * 0.01
        
        for _ in range(6):
            # 步骤 1: dist * y_n  (1次密文乘法)
            dy1, dy2 = self.sec_mul(dist1, dist2, y1, y2)
            
            # 步骤 2: 2 - (dist * y_n) (本地明文与密文相加减的开销)
            # 假设常数 2 的共享份额在 S1 上，S2 为 0
            diff1 = 2.0 - dy1
            diff2 = 0.0 - dy2
            
            # 步骤 3: y_n * (2 - dist * y_n) (第2次密文乘法)
            y1, y2 = self.sec_mul(y1, y2, diff1, diff2)
            
        # 此时 y1, y2 即为 1 / dist 的密文份额
        # 步骤 4: 计算比例 ratio = m_h * (1 / dist) (第3次密文乘法)
        Ratio1, Ratio2 = self.sec_mul(m_h1, m_h2, y1, y2)
        
        # 步骤 5: 将比例广播至全维度，与原始高维梯度相乘 (1次大维度全量乘法)
        Ratio1_exp = Ratio1.expand_as(Up1)
        Ratio2_exp = Ratio2.expand_as(Up2)
        
        return self.sec_mul(Up1, Up2, Ratio1_exp, Ratio2_exp)

    def run_benchmark(self, base_update):
        """执行计时跑分 (不含网络延迟)"""
        print(f"\n[*] 开始 EPPRFL 协议计算耗时跑分 (纯 CPU 数学运算，不含网络延迟)...")
        history_h1 = torch.zeros(self.sampled_d, dtype=torch.float64, device=self.device)
        history_h2 = torch.zeros(self.sampled_d, dtype=torch.float64, device=self.device)
        m_h1 = torch.tensor([0.5 * self.deg], dtype=torch.float64, device=self.device)
        m_h2 = torch.tensor([0.5 * self.deg], dtype=torch.float64, device=self.device)
        
        # ---------------------------------------------------------
        # 1. 客户端阶段：微扰注入与份额拆分
        # ---------------------------------------------------------
        shares_1, shares_2 = [], []
        client_start_time = time.time()
        for i in range(self.num_clients):
            client_update = base_update + torch.randn_like(base_update) * 1e-4
            s1 = torch.randn_like(client_update)
            s2 = client_update - s1
            shares_1.append(s1)
            shares_2.append(s2)
        client_avg_time = (time.time() - client_start_time) / self.num_clients

        # ---------------------------------------------------------
        # 2. 服务器端阶段
        # ---------------------------------------------------------
        server_timings = {}
        
        # [A] Downsampling
        t0 = time.time()
        sampled_1 = [s[:self.sampled_d] for s in shares_1]
        sampled_2 = [s[:self.sampled_d] for s in shares_2]
        server_timings['P1_Downsampling'] = time.time() - t0

        # [B] SecDis
        t0 = time.time()
        dist1_list, dist2_list = [], []
        for i in range(self.num_clients):
            d1, d2 = self.sec_dis(sampled_1[i], sampled_2[i], history_h1, history_h2)
            dist1_list.append(d1)
            dist2_list.append(d2)
        server_timings['P2_SecDis'] = time.time() - t0

        # [C] SecMed
        t0 = time.time()
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                _ = self.sec_cmp(dist1_list[i], dist2_list[i], dist1_list[j], dist2_list[j])
        mid_idx = self.num_clients // 2
        m_h1_new, m_h2_new = dist1_list[mid_idx], dist2_list[mid_idx]
        server_timings['P3_SecMed'] = time.time() - t0

        # [D] SecCmp Filtering
        t0 = time.time()
        benign_indices = []
        lam_m1, lam_m2 = m_h1_new * self.lam, m_h2_new * self.lam
        for i in range(self.num_clients):
            diff1 = dist1_list[i] - m_h1_new
            diff2 = dist2_list[i] - m_h2_new
            sq_diff1, sq_diff2 = self.sec_mul(diff1, diff2, diff1, diff2)
            if self.sec_cmp(sq_diff1, sq_diff2, lam_m1, lam_m2):
                benign_indices.append(i)
        if not benign_indices: benign_indices = [0]
        server_timings['P4_Filtering'] = time.time() - t0

        # [E] SecClip & Aggregation
        t0 = time.time()
        agg1 = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        agg2 = torch.zeros(self.d, dtype=torch.float64, device=self.device)
            
        for i in benign_indices:
            if not self.sec_cmp(dist1_list[i], dist2_list[i], m_h1_new, m_h2_new):
                # 调用复杂的泰勒展开裁剪计算
                c1, c2 = self.sec_clip(shares_1[i], shares_2[i], dist1_list[i], dist2_list[i], m_h1_new, m_h2_new)
                agg1 += c1
                agg2 += c2
            else:
                agg1 += shares_1[i]
                agg2 += shares_2[i]
        
        agg1 /= len(benign_indices)
        agg2 /= len(benign_indices)
        server_timings['P5_Clip_And_Agg'] = time.time() - t0

        # ================= 打印报告 =================
        print("="*55)
        print("    EPPRFL 真实数据纯计算耗时跑分报告 (无网络延迟)")
        print("="*55)
        print(f" [客户端] 单节点分片耗时   : {client_avg_time * 1000:.4f} ms")
        print("-" * 55)
        total_server_time = 0.0
        for k, v in sorted(server_timings.items()):
            print(f" [服务端] {k:<15} : {v * 1000:.4f} ms")
            total_server_time += v
        print("-" * 55)
        print(f" [服务端] 总计纯数学计算耗时 : {total_server_time * 1000:.4f} ms ({total_server_time:.4f} s)")
        print("="*55)

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet20', choices=['LeNet5', 'ResNet18', 'ResNet20'])
    parser.add_argument('--num_clients', type=int, default=10) # 这里使用 100 进行极限测试
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 启动初始化 | Device: {device} | Model: {args.model}")

    # 1. 实例化模型与获取数据
    dataset_name = 'MNIST' if args.model == 'LeNet5' else 'CIFAR10'
    model_class = LeNet5 if args.model == 'LeNet5' else (ResNet18_CIFAR10 if args.model == 'ResNet18' else resnet20)
    
    # 获取 1 个客户端的 Loader 进行真实数据训练即可提取基准
    client_dataloaders, _ = get_federated_dataloaders(dataset_name=dataset_name, num_clients=1)
    
    model = model_class().to(device)
    global_weights = get_model_vector(model).clone()
    d = global_weights.size(0)
    print(f"[*] 获取模型真实维度 d = {d} (下采样维数: {int(d * 0.06)})")

    # 2. 前置操作：真实执行一次模型训练以获取真实的梯度基准
    print(f"[*] 正在 Client_00 上执行真实数据预热训练 (此阶段不计入 SMC 时间)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    
    first_client_loader = client_dataloaders[0]['host_loader']
    for images, labels in first_client_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    real_update = (get_model_vector(model) - global_weights).type(torch.float64)
    print(f"[*] 预热完毕！成功提取真实模型更新量。")

    # 3. 初始化 SMC 引擎并执行跑分 (无网络延迟)
    simulator = EPPRFL_Timing_Simulator(d=d, num_clients=args.num_clients, device=device)
    simulator.run_benchmark(real_update)

if __name__ == "__main__":
    main()