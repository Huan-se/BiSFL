import sys
import os
import socket
import time
import random
import math
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import argparse
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from _utils_.server_adapter import ServerAdapter
from Defence.layers_proj_detect import Layers_Proj_Detector
import network_utils as net

class ServerSimulator:
    def __init__(self, host='0.0.0.0', port=8888, num_clients=20, param_size=1000000, drop_rate=0.1, enable_projection=False):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.param_size = param_size
        self.drop_rate = drop_rate 
        self.enable_projection = enable_projection 
        self.enable_model_broadcast = False
        
        self.client_sockets = {}
        
        print("[Server] 初始化 Server 端环境与检测器...")
        self.server_adapter = ServerAdapter()
        self.detector = Layers_Proj_Detector(config={})
        
        self.global_proj_cache = np.zeros(1024, dtype=np.float32)
        
        self.seed_mask_root = "12345678" 
        self.seed_global_0 = "87654321"  
        self.seed_sss = "11223344"
        
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
            
        print("[Server] 所有客户端物理连接已就绪。\n")

    def perform_ratls_handshake(self):
        print("[Server] === 开始执行 RA-TLS 双向认证与安全种子协商 ===")
        t_start = time.time()
        active_ids = list(self.client_sockets.keys())

        req_quote_msg = {"action": "REQ_RATLS_QUOTE"}
        self._send_to_targets(req_quote_msg, active_ids)
        quotes_results = self._recv_from_targets("RES_RATLS_QUOTE", active_ids)
        
        self.comm_bytes_ratls += self._get_msg_size(req_quote_msg) * len(active_ids)
        for cid, payload in quotes_results.items():
            self.comm_bytes_ratls += 2 * self._get_msg_size({"action": "RES_RATLS_QUOTE", "data": payload})

        time.sleep(0.005 * len(active_ids))

        dummy_cipher_seeds = os.urandom(128) 
        msg_dicts = {cid: {"action": "SYNC_SEEDS", "data": dummy_cipher_seeds} for cid in active_ids}
        self._send_custom_to_targets(msg_dicts, active_ids)
        self._recv_from_targets("RES_SEEDS_ACK", active_ids)
        
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

    def _generate_k_regular_graph(self, active_ids):
        N = len(active_ids)
        if N <= 1: return {}, 0
        K_degree = max(4, 2 * int(math.ceil(math.log2(N)))) 
        if K_degree >= N: K_degree = N - 1 if (N - 1) % 2 == 0 else N - 2
            
        neighbors = {i: [] for i in active_ids}
        for idx, cid in enumerate(active_ids):
            for d in range(1, (K_degree // 2) + 1):
                neighbors[cid].append(active_ids[(idx + d) % N])
                neighbors[cid].append(active_ids[(idx - d) % N])
        return neighbors, K_degree

    def run_simulation_round(self):
        active_ids = list(self.client_sockets.keys())
        active_ids.sort()
        N = len(active_ids)
        
        comm_bytes_proj_upload = 0
        comm_bytes_weight_down = 0
        comm_bytes_cipher_up = 0
        comm_bytes_shares_req = 0
        comm_bytes_shares_up = 0
        
        print(f"========== 完整通信与计算性能测试 (维度: {self.param_size}) ==========")
        
        dummy_global_model = np.random.randn(self.param_size).astype(np.float32)
        model_hash_str = str(int(hashlib.sha256(dummy_global_model.tobytes()).hexdigest()[:15], 16))

        if self.enable_model_broadcast:
            self._send_to_targets({"action": "SYNC_MODEL", "data": dummy_global_model}, active_ids)
        else:
            self._send_to_targets({"action": "SYNC_MODEL", "data": None}, active_ids)
        self._recv_from_targets("RES_TRAIN_DONE", active_ids)

        if self.enable_projection:
            print("[1/4] 发起请求，客户端生成投影并回传...")
            t_start_proj = time.time()
            self._send_to_targets({"action": "REQ_PROJ"}, active_ids)
            proj_results = self._recv_from_targets("RES_PROJ", active_ids)
            t_proj_total = time.time() - t_start_proj
            
            for cid, feat in proj_results.items():
                comm_bytes_proj_upload += self._get_msg_size({"action": "RES_PROJ", "data": feat})

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
            
            new_global_proj = np.zeros(1024, dtype=np.float32)
            for cid in accepted_ids:
                new_global_proj += weights_map[cid] * proj_results[cid]
            self.global_proj_cache = new_global_proj
            t_det_total = time.time() - t_start_det

            if not accepted_ids:
                return
        else:
            print("[Skip] 跳过投影和检测阶段...")
            t_proj_total = 0.0
            t_det_total = 0.0
            accepted_ids = active_ids[:]
            weights_map = {cid: 1.0 / N for cid in active_ids}
            self._send_to_targets({"action": "SKIP_PROJ"}, active_ids)
            self._recv_from_targets("RES_SKIP_PROJ", active_ids)

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

        graph_neighbors, K_degree = self._generate_k_regular_graph(u1_ids)
        threshold = K_degree // 2 + 1

        print(f"[3/5] 安全加扰与收集: 存活 {N_u2}/{N_u1}, K-正则图度数 K={K_degree}...")
        t_start_mask = time.time()
        
        msg_dicts = {}
        for cid in u2_ids:
            msg_dicts[cid] = {
                "action": "REQ_CIPHER", 
                "weight": weights_map[cid],
                "kappa_m": self.seed_mask_root,
                "t": 1,
                "model_hash": model_hash_str
            }
            
        self._send_custom_to_targets(msg_dicts, u2_ids)
        for cid in u2_ids:
            comm_bytes_weight_down += self._get_msg_size(msg_dicts[cid])
            
        cipher_results = self._recv_from_targets("RES_CIPHER", u2_ids)

        for cid, cipher_data in cipher_results.items():
            comm_bytes_cipher_up += self._get_msg_size({"action": "RES_CIPHER", "data": cipher_data})
        
        share_msg_dicts = {}
        view_hash_str = str(sum(u2_ids))
        for cid in u2_ids:
            dropped_neighbors = [n for n in graph_neighbors[cid] if n not in u2_ids]
            alive_neighbors = [n for n in graph_neighbors[cid] if n in u2_ids]
            share_msg_dicts[cid] = {
                "action": "REQ_SHARES", 
                "alive_neighbors": alive_neighbors,
                "dropped_neighbors": dropped_neighbors,
                "threshold": threshold,
                "kappa_s": self.seed_sss,
                "kappa_m": self.seed_mask_root,
                "t": 1,
                "view_hash": view_hash_str
            }
            
        self._send_custom_to_targets(share_msg_dicts, u2_ids)
        shares_results = self._recv_from_targets("RES_SHARES", u2_ids)
        
        for cid in u2_ids:
            comm_bytes_shares_req += self._get_msg_size(share_msg_dicts[cid])
            comm_bytes_shares_up += self._get_msg_size({"action": "RES_SHARES", "data": shares_results[cid]})
        
        t_mask_total = time.time() - t_start_mask

        # [精准修正！] 将 TA 离线计算与在线聚合时间剥离
        print("[4/5] TA (Trusted Authority) 离线预计算全局掩码流...")
        ta_s_alpha, ta_s_beta = self.server_adapter.ta_offline_compute(
            u1_ids, self.seed_mask_root, 1, self.param_size
        )

        print("[5/5] ServerCore 在线聚合与提取明文...")
        ciphers_list = [cipher_results[cid] for cid in u2_ids]
        shares_list = [shares_results[cid] for cid in u2_ids]
        
        # 此时只计入纯净的在线 Server 开销
        t_start_agg = time.time()
        result_float = self.server_adapter.aggregate_and_unmask_sparse(
            u1_ids, u2_ids, shares_list, ciphers_list, 
            self.seed_mask_root, 1, model_hash_str, threshold,
            ta_s_alpha, ta_s_beta
        )
        t_agg_total = time.time() - t_start_agg
        
        comm_init_kb = self.comm_bytes_ratls / 1024.0 / N
        comm_proj_detect_kb = (comm_bytes_proj_upload + comm_bytes_weight_down) / 1024.0 / N
        comm_cipher_up_kb = comm_bytes_cipher_up / 1024.0 / N_u2
        comm_recovery_kb = (comm_bytes_shares_req + comm_bytes_shares_up) / 1024.0 / N_u2
        total_comm_kb = comm_init_kb + comm_proj_detect_kb + comm_cipher_up_kb + comm_recovery_kb

        print("\n============= Communication Report =============")
        print("【通信量评估 (单客户端均值)】")
        print(f" 1. 系统初始化通信量 : {comm_init_kb:.2f} KB")
        print(f" 2. 投影检测方案开销 : {comm_proj_detect_kb:.2f} KB (极小化 O(1) 权重下发)")
        print(f" 3. 掩码模型上传开销 : {comm_cipher_up_kb:.2f} KB")
        print(f" 4. 标量分片恢复传输 : {comm_recovery_kb:.2f} KB (稀疏路由 O(log N))")
        print(" -----------------------------------------------")
        print(f" 📊 总物理网络开销   : {total_comm_kb:.2f} KB")
        
        avg_init = self.t_init_total / N 
        avg_proj = t_proj_total / N
        avg_mask = t_mask_total / N_u2 if N_u2 > 0 else 0
        total_mask_pipeline = avg_init + avg_mask + t_agg_total
        total_proj_pipeline = avg_proj + t_det_total

        print("\n============= Time Report =============")
        print(f" [配置] 客户端: {N} | 存活: {N_u2} | 维度: {self.param_size} | K度数: {K_degree}")
        print(" -----------------------------------------------")
        print(f" 1. 平均初始化时间     : {avg_init:.4f} s/client")
        print(f" 2. 平均投影时间       : {avg_proj:.4f} s/client")
        print(f" 3. Server异常检测时间 : {t_det_total:.4f} s")
        print(f" 4. 平均安全加扰时间   : {avg_mask:.4f} s/client")
        print(f" 5. 标量聚合还原时间   : {t_agg_total:.4f} s")
        print(" ===============================================")
        print(f" 🚀 总时间 (掩码流程)  : {total_mask_pipeline:.4f} s")
        print(f" 🛡️ 投影检测总时间     : {total_proj_pipeline:.4f} s")
        print("================================================\n")
        
        for sock in self.client_sockets.values(): sock.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8887)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--enable_projection", type=bool, default=False)
    args = parser.parse_args()
    server = ServerSimulator(port=args.port, num_clients=args.num_clients, param_size=args.param_size, drop_rate=args.drop_rate, enable_projection=args.enable_projection)
        
    client_process = subprocess.Popen(
    ["python", "client_simulator.py", "--num_clients", f"{args.num_clients}", "--param_size", f"{args.param_size}", "--port", f"{args.port}"])

    server.wait_for_clients()
    server.perform_ratls_handshake() 
    server.run_simulation_round()