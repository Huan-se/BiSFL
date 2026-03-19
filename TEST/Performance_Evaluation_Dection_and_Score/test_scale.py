import sys
import os
import time
import pickle
import csv
import numpy as np
import torch  # [新增] 引入 torch 以修复类型错误

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

def run_benchmark():
    print("===================================================================")
    print("🚀 启动 OURS 核心机制 Benchmark (TEE 串行投影 + Server 异常检测)")
    print("===================================================================\n")

    # --- 1. 初始化 TEE Enclave ---
    print("[System] 正在初始化 SGX Enclave...")
    tee_adapter = get_tee_adapter_singleton()
    try:
        tee_adapter.initialize_enclave()
    except Exception as e:
        print(f"❌ Enclave 初始化失败: {e}。请检查 libtee.so 是否已编译。")
        sys.exit(1)
    
    # --- 2. 定义实验变量网格 ---
    PARAM_POWERS = [15 ,10, 10, 15]
    CLIENT_COUNTS = [10, 100, 1000, 10000]
    
    # 防御组件配置
    defense_config = {'target_layers': []} 
    
    results = []
    
    print("[Benchmark] 开始执行双重维度测试矩阵...\n")

    # 外层循环：遍历不同的模型参数维度 P
    for p_pow in PARAM_POWERS:
        P_size = 2 ** p_pow
        print(f"➤ 正在评估模型维度: P = 2^{p_pow} ({P_size} 参数)")
        
        # 准备 TEE 测试用假数据
        w_new_flat = np.random.randn(P_size).astype(np.float32)
        w_old_cache = np.zeros(P_size, dtype=np.float32)
        proj_seed = 42
        client_id = 1
        
        # -----------------------------------------------------------------
        # [Metric 1 & 2] TEE 投影时间 (T_proj) 与 上行通信量 (C_up)
        # -----------------------------------------------------------------
        # [修改] TEE 预热与串行求平均机制，防崩溃且结果更精确
        print("   >> 正在串行评估 TEE 投影时间...")
        num_trials = 10  # 串行测试 10 次取平均值
        total_proj_time = 0.0
        
        # 预热一次 (抵消 SGX 冷启动与内存页换入开销)
        _, _ = tee_adapter.prepare_gradient(client_id, proj_seed, w_new_flat, w_old_cache)
        
        for _ in range(num_trials):
            t_start = time.perf_counter()
            output_proj, ranges = tee_adapter.prepare_gradient(client_id, proj_seed, w_new_flat, w_old_cache)
            total_proj_time += (time.perf_counter() - t_start)
            
        t_proj_ms = (total_proj_time / num_trials) * 1000.0 # 计算单次平均毫秒
        
        # 计算上行网络载荷大小 (Bytes -> KB)
        feature_dict = {'full': output_proj}
        c_up_kb = len(pickle.dumps(feature_dict)) / 1024.0
        
        # 生成 Server 所需的 global_direction
        global_output, _ = tee_adapter.prepare_gradient(0, proj_seed, w_new_flat, w_old_cache)
        # [修复] 必须包装为 torch.Tensor 才能传入 Server 的防御检测算法
        global_direction = {'full': torch.from_numpy(global_output)}
        
        K_dim = len(output_proj)

        # 内层循环：遍历不同的客户端数量 N
        for N in CLIENT_COUNTS:
            # -----------------------------------------------------------------
            # 准备 Server 测试数据
            # -----------------------------------------------------------------
            client_projections = {}
            for cid in range(1, N + 1):
                # 注入极微小的高斯噪声以区分不同客户端，防止聚类算法退化
                noisy_proj = output_proj + np.random.normal(0, 0.001, K_dim).astype(np.float32)
                # [修复] 强制包装为 PyTorch Tensor
                client_projections[cid] = {'full': torch.from_numpy(noisy_proj)}
            
            suspect_counters = {}
            detector = Layers_Proj_Detector(config=defense_config)
            
            # -----------------------------------------------------------------
            # [Metric 3] Server 检测时间 (T_detect)
            # -----------------------------------------------------------------
            t_start_det = time.perf_counter()
            raw_weights, logs, global_stats = detector.detect(
                client_projections, global_direction, suspect_counters, verbose=False
            )
            t_detect_ms = (time.perf_counter() - t_start_det) * 1000.0
            
            # -----------------------------------------------------------------
            # [Metric 4] 下行通信量 (C_down)
            # -----------------------------------------------------------------
            weights_map = {cid: w for cid, w in raw_weights.items()}
            c_down_kb = len(pickle.dumps(weights_map)) / 1024.0
            
            # 记录结果
            results.append({
                "Param_Power": p_pow,
                "Param_Size": P_size,
                "Proj_Dim_K": K_dim,
                "Clients_N": N,
                "T_proj_ms": t_proj_ms,
                "C_up_KB": c_up_kb,
                "T_detect_ms": t_detect_ms,
                "C_down_KB": c_down_kb
            })
            
    # --- 3. 结果汇总与导出 ---
    print("\n✅ 所有 Benchmark 执行完毕！正在生成报表...")
    
    csv_file = "projection_detection_benchmark.csv"
    headers = ["Param_Power", "Param_Size", "Proj_Dim_K", "Clients_N", "T_proj_ms", "C_up_KB", "T_detect_ms", "C_down_KB"]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n| {'P (2^x)':<8} | {'P_Size':<10} | {'K_Dim':<6} | {'N':<6} || {'T_proj (ms)':<12} | {'C_up (KB)':<10} || {'T_detect (ms)':<13} | {'C_down (KB)':<11} |")
    print("|" + "-"*10 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*8 + "||" + "-"*14 + "|" + "-"*12 + "||" + "-"*15 + "|" + "-"*13 + "|")
    
    for r in results:
        print(f"| {r['Param_Power']:<8} | {r['Param_Size']:<10} | {r['Proj_Dim_K']:<6} | {r['Clients_N']:<6} || "
              f"{r['T_proj_ms']:<12.2f} | {r['C_up_KB']:<10.2f} || "
              f"{r['T_detect_ms']:<13.2f} | {r['C_down_KB']:<11.2f} |")
        
    print(f"\n💾 数据已成功落盘至: {os.path.abspath(csv_file)}")

if __name__ == "__main__":
    run_benchmark()