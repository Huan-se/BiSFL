import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from model.Lenet5 import LeNet5
from model.Resnet18 import ResNet18_CIFAR10
from model.Resnet20 import resnet20
from data_loader import get_federated_dataloaders

def get_model_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# ==========================================
# 通信量理论计算引擎 (EPPRFL)
# ==========================================
def calc_epprfl_comm(d, sampled_d, num_clients):
    """
    计算 EPPRFL 的精确通信量 (MB)
    基于 Float64 加法秘密共享，每个元素占 8 Bytes
    """
    p_bytes = 8
    
    # 1. 客户端平均通信量 (上传两个 Share + 下载全局模型)
    client_up = 2 * d * p_bytes
    client_down = d * p_bytes
    client_avg_mb = (client_up + client_down) / (1024 * 1024)
    
    # 2. 服务器端总通信量 (S1 <-> S2 交互 + S1 下发)
    # [SecDis]: 交互恢复盲化值 D 和 E (维度为 sampled_d)。S1 和 S2 各发 2*sampled_d
    secdis_comm = num_clients * (sampled_d * p_bytes * 4)
    # [SecMed]: 两两比较，共 N(N-1)/2 次 SecCmp。每次 SecCmp = SecMul(标量) + Reveal(标量)
    num_cmp = num_clients * (num_clients - 1) // 2
    secmed_comm = num_cmp * (6 * p_bytes)
    # [SecCmp 过滤]: 每客户端 1次标量 SecMul + 1次 SecCmp
    filtering_comm = num_clients * (10 * p_bytes)
    # [SecClip]: 最坏情况所有良性节点都裁剪。1次标量 SecMul + 1次全维(d) SecMul
    secclip_comm = num_clients * (4 * p_bytes + d * p_bytes * 4)
    # [广播下发]: S1 重建全局模型后下发给所有客户端
    broadcast_comm = num_clients * d * p_bytes
    
    server_total_mb = (secdis_comm + secmed_comm + filtering_comm + secclip_comm + broadcast_comm) / (1024 * 1024)
    return client_avg_mb, server_total_mb

# ==========================================
# 时间测试核心引擎 (EPPRFL 协议仿真)
# ==========================================
class EPPRFL_Timing_Simulator:
    def __init__(self, d, num_clients=10, epsilon=0.06, lam=0.05, device='cpu'):
        self.num_clients = num_clients
        self.d = d
        self.epsilon = epsilon
        self.sampled_d = int(d * epsilon)
        self.lam = lam
        self.device = device

    def generate_beaver_triple(self, shape):
        A = torch.randn(shape, dtype=torch.float64, device=self.device)
        B = torch.randn(shape, dtype=torch.float64, device=self.device)
        C = A * B
        return (A/2, A/2), (B/2, B/2), (C/2, C/2)

    def sec_mul(self, X1, X2, Y1, Y2):
        (A1, A2), (B1, B2), (C1, C2) = self.generate_beaver_triple(X1.shape)
        D1 = X1 - A1; D2 = X2 - A2
        E1 = Y1 - B1; E2 = Y2 - B2
        D = D1 + D2
        E = E1 + E2
        Z1 = C1 + D * B1 + E * A1
        Z2 = C2 + D * B2 + E * A2 + D * E
        return Z1, Z2

    def sec_cmp(self, X1, X2, Y1, Y2):
        Diff1 = X1 - Y1; Diff2 = X2 - Y2
        R = torch.abs(torch.randn(1, dtype=torch.float64, device=self.device)) + 0.1
        R1 = R / 2; R2 = R / 2
        Z1, Z2 = self.sec_mul(R1, R2, Diff1, Diff2)
        return (Z1 + Z2).item() < 0

    def sec_dis(self, X1, X2, Y1, Y2):
        Diff1 = X1 - Y1; Diff2 = X2 - Y2
        Sq1, Sq2 = self.sec_mul(Diff1, Diff2, Diff1, Diff2)
        return torch.sum(Sq1), torch.sum(Sq2)

    def sec_clip(self, Up1, Up2, dist1, dist2, m_h1, m_h2):
        Ratio1, Ratio2 = self.sec_mul(m_h1, m_h2, torch.ones_like(dist1), torch.ones_like(dist2))
        Ratio1_exp = Ratio1.expand_as(Up1)
        Ratio2_exp = Ratio2.expand_as(Up2)
        return self.sec_mul(Up1, Up2, Ratio1_exp, Ratio2_exp)

    def run_benchmark(self, base_update):
        history_h1 = torch.zeros(self.sampled_d, dtype=torch.float64, device=self.device)
        history_h2 = torch.zeros(self.sampled_d, dtype=torch.float64, device=self.device)
        m_h1 = torch.tensor([0.5], dtype=torch.float64, device=self.device)
        m_h2 = torch.tensor([0.5], dtype=torch.float64, device=self.device)
        
        # 1. 客户端阶段 (微扰注入与秘密共享切分)
        shares_1, shares_2 = [], []
        t0 = time.time()
        for i in range(self.num_clients):
            client_update = base_update + torch.randn_like(base_update) * 1e-4
            s1 = torch.randn_like(client_update)
            s2 = client_update - s1
            shares_1.append(s1)
            shares_2.append(s2)
        client_avg_time = (time.time() - t0) / self.num_clients

        server_timings = {}
        
        # 2. [Downsampling]
        t0 = time.time()
        sampled_1 = [s[:self.sampled_d] for s in shares_1]
        sampled_2 = [s[:self.sampled_d] for s in shares_2]
        server_timings['t_downsampling'] = time.time() - t0

        # 3. [SecDis]
        t0 = time.time()
        dist1_list, dist2_list = [], []
        for i in range(self.num_clients):
            d1, d2 = self.sec_dis(sampled_1[i], sampled_2[i], history_h1, history_h2)
            dist1_list.append(d1)
            dist2_list.append(d2)
        server_timings['t_secdis'] = time.time() - t0

        # 4. [SecMed]
        t0 = time.time()
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                _ = self.sec_cmp(dist1_list[i], dist2_list[i], dist1_list[j], dist2_list[j])
        mid_idx = self.num_clients // 2
        m_h1_new, m_h2_new = dist1_list[mid_idx], dist2_list[mid_idx]
        server_timings['t_secmed'] = time.time() - t0

        # 5. [SecCmp Filtering]
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
        server_timings['t_filtering'] = time.time() - t0

        # 6. [SecClip & Aggregation]
        t0 = time.time()
        agg1 = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        agg2 = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        for i in benign_indices:
            if not self.sec_cmp(dist1_list[i], dist2_list[i], m_h1_new, m_h2_new):
                c1, c2 = self.sec_clip(shares_1[i], shares_2[i], dist1_list[i], dist2_list[i], m_h1_new, m_h2_new)
                agg1 += c1; agg2 += c2
            else:
                agg1 += shares_1[i]; agg2 += shares_2[i]
        server_timings['t_clip_agg'] = time.time() - t0
        
        return client_avg_time, server_timings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet20')
    parser.add_argument('--num_clients', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet20().to(device)
    d = get_model_vector(model).size(0)
    
    # 打印通信量分析
    epsilon = 0.06
    sampled_d = int(d * epsilon)
    c_avg_mb, s_total_mb = calc_epprfl_comm(d, sampled_d, args.num_clients)
    print(f"\n[*] EPPRFL 通信量分析 (模型维度 {d}, 采样维度 {sampled_d}):")
    print(f"    - 客户端平均通信量 (上+下): {c_avg_mb:.2f} MB")
    print(f"    - 服务器端总通信量 (交互+下发): {s_total_mb:.2f} MB")

    print(f"\n>>> 开始 EPPRFL 单轮完整流程测试...")
    simulator = EPPRFL_Timing_Simulator(d=d, num_clients=args.num_clients, epsilon=epsilon, device=device)
    
    # 生成基准真实维度更新量
    real_update = torch.randn(d, dtype=torch.float64, device=device)
    client_avg_time, s_logs = simulator.run_benchmark(real_update)
    
    print("\n  [客户端时间耗时 (平均)]")
    print(f"  - ASS 秘密共享切分 : {client_avg_time:.4f} s")
    
    print("\n  [服务器端时间耗时 (总计)]")
    print(f"  - 下采样截断 : {s_logs['t_downsampling']:.4f} s")
    print(f"  - SecDis 安全距离 : {s_logs['t_secdis']:.4f} s")
    print(f"  - SecMed 安全中位数 : {s_logs['t_secmed']:.4f} s")
    print(f"  - 恶意梯度过滤 : {s_logs['t_filtering']:.4f} s")
    print(f"  - 安全裁剪与聚合 : {s_logs['t_clip_agg']:.4f} s")

if __name__ == "__main__":
    main()