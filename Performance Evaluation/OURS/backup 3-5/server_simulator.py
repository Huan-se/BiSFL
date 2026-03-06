import sys
import os
import socket
import time
import random
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from _utils_.server_adapter import ServerAdapter
from Defence.layers_proj_detect import Layers_Proj_Detector
import network_utils as net

class ServerSimulator:
    def __init__(self, host='0.0.0.0', port=8888, num_clients=20, param_size=1000000, drop_rate=0.1, enable_projection=True):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.param_size = param_size
        self.drop_rate = drop_rate 
        self.enable_projection = enable_projection 
        
        self.client_sockets = {}
        
        print("[Server] 初始化 Server 端环境与检测器...")
        self.server_adapter = ServerAdapter()
        # 初始化检测器
        self.detector = Layers_Proj_Detector(config={})
        
        # [核心修正 1]：全局投影缓存，初始轮次设定为全 0 向量 (假设投影维度为 1024)
        self.global_proj_cache = np.zeros(1024, dtype=np.float32)
        
        self.seed_mask_root = 0x12345678 
        self.seed_global_0 = 0x87654321  
        self.seed_sss = 0x11223344
        
        # 统计变量
        self.t_init_total = 0
        self.comm_bytes_ratls = 0

    def _get_msg_size(self, msg_dict):
        import pickle
        return len(pickle.dumps(msg_dict)) + 4

    def wait_for_clients(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(self.num_clients)
        
        print(f"[Server] 监听端口 {self.port}，等待 {self.num_clients} 个客户端接入...")
        
        connected_count = 0
        while connected_count < self.num_clients:
            client_sock, addr = server_sock.accept()
            client_id = connected_count
            self.client_sockets[client_id] = client_sock
            net.send_msg(client_sock, {"action": "ASSIGN_ID", "client_id": client_id})
            connected_count += 1
            
        print("[Server] 所有客户端物理连接已就绪。")

    def perform_ratls_handshake(self):
        """执行严格的 RA-TLS 协议数据包交换与认证"""
        print("\n[Server] === 开始执行 RA-TLS 双向认证与安全种子协商 ===")
        t_start = time.time()
        active_ids = list(self.client_sockets.keys())

        # 1. 索要 TEE Quote (包含公钥)
        req_quote_msg = {"action": "REQ_RATLS_QUOTE"}
        self._send_to_targets(req_quote_msg, active_ids)
        quotes_results = self._recv_from_targets("RES_RATLS_QUOTE", active_ids)
        
        # 统计通信量
        self.comm_bytes_ratls += self._get_msg_size(req_quote_msg) * len(active_ids)
        for cid, payload in quotes_results.items():
            self.comm_bytes_ratls += 2 * self._get_msg_size({"action": "RES_RATLS_QUOTE", "data": payload})

        # 2. 模拟 Server 侧并发验证 Quote 签名的 CPU 耗时 (通常为几毫秒)
        time.sleep(0.005 * len(active_ids))

        # 3. Server 使用共享密钥加密种子并下发 (128 Bytes AES 密文负载)
        dummy_cipher_seeds = os.urandom(128) 
        msg_dicts = {cid: {"action": "SYNC_SEEDS", "data": dummy_cipher_seeds} for cid in active_ids}
        self._send_custom_to_targets(msg_dicts, active_ids)
        self._recv_from_targets("RES_SEEDS_ACK", active_ids)
        
        # 统计通信量
        for cid in active_ids:
            self.comm_bytes_ratls += self._get_msg_size(msg_dicts[cid])
            self.comm_bytes_ratls += self._get_msg_size({"action": "RES_SEEDS_ACK"})

        self.t_init_total = time.time() - t_start
        print(f"[Server] RA-TLS 握手完成！总耗时: {self.t_init_total:.4f} s\n")

    def _send_to_targets(self, msg_dict, target_ids):
        def _send(cid): net.send_msg(self.client_sockets[cid], msg_dict)
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_send, cid)

    def _send_custom_to_targets(self, msg_dicts, target_ids):
        def _send(cid): net.send_msg(self.client_sockets[cid], msg_dicts[cid])
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_send, cid)

    def _recv_from_targets(self, expected_action, target_ids):
        results = {}
        def _recv(cid):
            res = net.recv_msg(self.client_sockets[cid])
            if res and res.get("action") == expected_action:
                results[cid] = res.get("data")
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_recv, cid)
        return results
    def _get_msg_size(self, msg_dict):
        import pickle
        # 序列化后的字节长度 + 4字节的头部长度
        return len(pickle.dumps(msg_dict)) + 4

    def run_simulation_round(self):
        active_ids = list(self.client_sockets.keys())
        active_ids.sort()
        N = len(active_ids)
        
        # --- 通信量统计累加器 (以 Byte 为单位) ---
        comm_bytes_proj_upload = 0
        comm_bytes_weight_down = 0
        comm_bytes_shares_req = 0
        comm_bytes_shares_up = 0
        
        print(f"========== 完整通信与计算性能测试 (维度: {self.param_size}) ==========")
        
        # --- 准备阶段：下发模型 (此时间不计入您要求的五大核心耗时) ---
        dummy_global_model = np.random.randn(self.param_size).astype(np.float32)
        self._send_to_targets({"action": "SYNC_MODEL", "data": dummy_global_model}, active_ids)
        self._recv_from_targets("RES_TRAIN_DONE", active_ids)

        if self.enable_projection:
            # ==========================================================
            # 1. 投影时间：模型生成后 -> TEE 投影 -> 上传完毕
            # ==========================================================
            print("[1/4] 发起请求，客户端生成投影并回传...")
            t_start_proj = time.time()
            self._send_to_targets({"action": "REQ_PROJ"}, active_ids)
            proj_results = self._recv_from_targets("RES_PROJ", active_ids)
            t_proj_total = time.time() - t_start_proj
            
            # [通信量统计] 累加上传的投影大小
            for cid, feat in proj_results.items():
                comm_bytes_proj_upload += self._get_msg_size({"action": "RES_PROJ", "data": feat})

            # ==========================================================
            # 2. Server 异常检测：Server 端纯净计算与基准迭代
            # ==========================================================
            print("[2/4] 执行 Server 端防御检测算法...")
            t_start_det = time.time()
            
            global_update_direction = {'full': torch.from_numpy(self.global_proj_cache)}
            client_projections = {cid: {'full': torch.from_numpy(feat)} for cid, feat in proj_results.items()}
            suspect_counters = {}
            
            raw_weights, logs, global_stats = self.detector.detect(
                client_projections, global_update_direction, suspect_counters, verbose=False
            )
            total_score = sum(raw_weights.values())
            weights_map = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
            
            accepted_ids = [cid for cid, w in weights_map.items() if w > 1e-6]
            accepted_ids.sort()
            
            # 迭代更新全局基准投影
            new_global_proj = np.zeros(1024, dtype=np.float32)
            for cid in accepted_ids:
                new_global_proj += weights_map[cid] * proj_results[cid]
            self.global_proj_cache = new_global_proj
            t_det_total = time.time() - t_start_det

            if not accepted_ids:
                print("[Error] 没有客户端被接受！测试中止。")
                return
            
        else:
            print("[Skip] 跳过投影和检测阶段...")
            t_proj_total = 0.0
            t_det_total = 0.0
            
            # [新增] 默认所有客户端均存活，且权重平分
            accepted_ids = active_ids[:]
            weights_map = {cid: 1.0 / N for cid in active_ids}
            
        N_acc = len(accepted_ids) # 参与加密的活跃客户端数

        u1_ids = accepted_ids
        num_dropped = int(len(u1_ids) * self.drop_rate)
        if num_dropped > 0:
            dropped_ids = random.sample(u1_ids, num_dropped)
            u2_ids = [cid for cid in u1_ids if cid not in dropped_ids]
        else:
            u2_ids = u1_ids[:]
        u2_ids.sort()
        
        N_u1 = len(u1_ids)
        N_u2 = len(u2_ids)


        # ==========================================================
        # 3. 安全加扰时间：下发梯度掩码指令 -> 生成密文与分片 -> 上传完毕
        # ==========================================================
        print(f"[3/4] 安全加扰与收集: 期望 {N_u1} 人, 实际存活 {N_u2} 人 (掉线率 {self.drop_rate*100}%)...")
        t_start_mask = time.time()
        
        msg_dicts = {cid: {"action": "REQ_CIPHER", "u1_ids": u1_ids, "weight": weights_map[cid]} for cid in u2_ids}
        self._send_custom_to_targets(msg_dicts, u2_ids)
        
        for cid in u2_ids:
            comm_bytes_weight_down += self._get_msg_size(msg_dicts[cid])
            
        cipher_results = self._recv_from_targets("RES_CIPHER", u2_ids)
        
        # 传递 U1 和 U2 给存活的客户端
        req_share_msg = {"action": "REQ_SHARES", "u1_ids": u1_ids, "u2_ids": u2_ids}
        self._send_to_targets(req_share_msg, u2_ids)
        shares_results = self._recv_from_targets("RES_SHARES", u2_ids)
        
        t_mask_total = time.time() - t_start_mask
        
        comm_bytes_shares_req += self._get_msg_size(req_share_msg) * N_u2
        for cid, shares in shares_results.items():
            comm_bytes_shares_up += self._get_msg_size({"action": "RES_SHARES", "data": shares})
        
        t_mask_total = time.time() - t_start_mask

        # ==========================================================
        # 4. 安全聚合时间：Server 根据密文与分片进行聚合消去
        # ==========================================================
        print("[4/4] 启动 C++ ServerCore 安全聚合提取明文...")
        ciphers_list = [cipher_results[cid] for cid in u2_ids]
        shares_list = [shares_results[cid] for cid in u2_ids]
        
        t_start_agg = time.time()
        result_int = self.server_adapter.aggregate_and_unmask(
            self.seed_mask_root, self.seed_global_0, u1_ids, u2_ids, shares_list, ciphers_list
        )
        t_agg_total = time.time() - t_start_agg
        
        # ==========================================================
        # 通信量计算 (转换为 KB)
        # ==========================================================
        comm_init_kb = self.comm_bytes_ratls / 1024.0 / N
        comm_proj_detect_kb = (comm_bytes_proj_upload + comm_bytes_weight_down) / 1024.0 / N
        
        # 3. 安全加扰与聚合阶段通信量 (除开掩码向量本身)
        # 掩码生成通信量本方案为0 (本地 PRG)
        comm_mask_gen_kb = 0.0 
        # 掉线恢复通信量 (Req Shares + Res Shares)
        comm_recovery_kb = (comm_bytes_shares_req + comm_bytes_shares_up) / 1024.0 / N_u2 if N_u2 > 0 else 0
        comm_secure_agg_kb = comm_mask_gen_kb + comm_recovery_kb

        # ==========================================================
        # 打印全新的通信量与时间报告
        # ==========================================================
        print("\n============= Communication Report =============")
        print("【通信量评估 (单客户端均值)】")
        print(f" 1. 系统初始化通信量 : {comm_init_kb:.2f} KB (含完整 Quote 验证与密钥分发)")
        print(f" 2. 投影检测方案开销 : {comm_proj_detect_kb:.2f} KB (投影上传 + 权重下发)")
        print(f" 3. 安全加扰与恢复开销: {comm_secure_agg_kb:.2f} KB (本方案掩码生成 0KB + 分片恢复传输)")
        print(" -----------------------------------------------")

        
        # ==========================================================
        # 最终时间换算与汇总报告输出
        # ==========================================================
        avg_init = self.t_init_total / N
        avg_proj = t_proj_total / N
        avg_mask = t_mask_total / N_u2 if N_u2 > 0 else 0
        total_mask_pipeline = avg_init + avg_mask + t_agg_total
        total_proj_pipeline = avg_proj + t_det_total

        print("\n============= Time Report =============")
        print(f" [配置] 总客户端数: {N} | 参与聚合数: {N_u2} | 参数维度: {self.param_size}")
        print(" -----------------------------------------------")
        print(f" 1. 平均初始化时间     : {avg_init:.4f} s/client (TEE启动 + RATLS建立)")
        print(f" 2. 平均投影时间       : {avg_proj:.4f} s/client (参数生成 -> TEE投影 -> 传输)")
        print(f" 3. Server异常检测时间 : {t_det_total:.4f} s (纯 Server 端加权计算)")
        print(f" 4. 平均安全加扰时间   : {avg_mask:.4f} s/client (梯度加掩码 -> 分片生成 -> 传输)")
        print(f" 5. 安全聚合时间       : {t_agg_total:.4f} s (聚合求解明文更新，不含更新模型)")
        print(" ===============================================")
        print(f" 🚀 总时间 (掩码流程)  : {total_mask_pipeline:.4f} s  (公式: 1 + 4 + 5)")
        print(f" 🛡️ 投影检测总时间     : {total_proj_pipeline:.4f} s  (公式: 2 + 3)")
        print("================================================\n")
        
        for sock in self.client_sockets.values(): sock.close()
        
if __name__ == "__main__":
    # 实例化 Server 模拟器
    # 您可以在这里修改端口、客户端数量、参数维度等
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=40)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--enable_projection", type=bool, default=False)
    args = parser.parse_args()
    server = ServerSimulator(port=args.port, num_clients=args.num_clients, param_size=args.param_size, drop_rate=args.drop_rate, enable_projection=args.enable_projection)
        
    client_process = subprocess.Popen(
    ["python", "client_simulator.py", "--num_clients", f"{args.num_clients}", "--param_size", f"{args.param_size}", "--port", f"{args.port}"])

    # 第一步：挂起并等待客户端连接 (模拟 RATLS)
    server.wait_for_clients()
    server.perform_ratls_handshake() # 物理连接就绪后执行 RA-TLS
    # 第二步：所有客户端连齐后，开始执行一轮完整的通信与加密测试
    server.run_simulation_round()