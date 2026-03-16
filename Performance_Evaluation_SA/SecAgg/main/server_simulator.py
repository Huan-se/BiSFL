import sys
import os
import socket
import time
import random
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import network_utils as net
from _utils_.crypto_utils import CryptoUtils

MOD = 9223372036854775783
SCALE = 100000000.0

class ServerSimulator:
    def __init__(self, host='0.0.0.0', port=8888, num_clients=20, param_size=1000000, drop_rate=0.1):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.param_size = param_size
        self.drop_rate = drop_rate 
        
        self.client_sockets = {}
        self.public_keys = {}
        
        print("[Server] 初始化 Server 端环境 (NIST P-256 Double-Masking Version)...")
        self.comm_bytes_round0 = 0
        self.comm_bytes_round1 = 0
        self.comm_bytes_round2 = 0
        self.comm_bytes_round4 = 0

    def _get_msg_size(self, msg_dict):
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
            client_id = connected_count + 1 
            self.client_sockets[client_id] = client_sock
            net.send_msg(client_sock, {"action": "ASSIGN_ID", "client_id": client_id})
            connected_count += 1
        print("[Server] 所有客户端物理连接已就绪。")

    def _send_to_targets(self, msg_dict, target_ids):
        def _send(cid): net.send_msg(self.client_sockets[cid], msg_dict)
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_send, cid)

    def _send_custom_to_targets(self, msg_dicts, target_ids):
        def _send(cid): net.send_msg(self.client_sockets[cid], msg_dicts[cid])
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_send, cid)

    def _recv_from_targets(self, target_ids):
        results = {}
        def _recv(cid):
            res = net.recv_msg(self.client_sockets[cid])
            if res: results[cid] = res
        with ThreadPoolExecutor(max_workers=len(target_ids)) as executor:
            for cid in target_ids: executor.submit(_recv, cid)
        return results

    def run_simulation_round(self):
        active_ids = list(self.client_sockets.keys())
        active_ids.sort()
        N = len(active_ids)
        threshold = N // 2 + 1 
        
        print(f"\n========== 完整通信与计算性能测试 (维度: {self.param_size}) ==========")
        
        self._send_to_targets({"action": "SYNC_MODEL", "data": None}, active_ids)
        self._recv_from_targets(active_ids)

        # ==========================================================
        # Round 0: AdvertiseKeys
        # ==========================================
        print(f"[1/4] Round 0: 收集并广播 ECDH 公钥...")
        t_start_r0 = time.time()
        
        req_msg = {"action": "REQ_ROUND0"}
        self._send_to_targets(req_msg, active_ids)
        r0_results = self._recv_from_targets(active_ids)
        
        for cid, res in r0_results.items():
            self.public_keys[cid] = (res["c_pk"], res["s_pk"])
            self.comm_bytes_round0 += self._get_msg_size(res)
        self.comm_bytes_round0 += self._get_msg_size(req_msg) * N

        sync_msg = {"action": "SYNC_PUBKEYS", "public_keys_dict": self.public_keys}
        self._send_to_targets(sync_msg, active_ids)
        self._recv_from_targets(active_ids)
        self.comm_bytes_round0 += self._get_msg_size(sync_msg) * N
        
        t_r0 = time.time() - t_start_r0

        # ==========================================================
        # Round 1: ShareKeys
        # ==========================================
        print(f"[2/4] Round 1: 客户端生成掩码分片，Server 执行路由分发...")
        t_start_r1 = time.time()
        
        req_msg_r1 = {"action": "REQ_ROUND1", "threshold": threshold}
        self._send_to_targets(req_msg_r1, active_ids)
        r1_results = self._recv_from_targets(active_ids)
        
        routed_ciphertexts = {cid: {} for cid in active_ids}
        for sender_id, res in r1_results.items():
            self.comm_bytes_round1 += self._get_msg_size(res)
            for receiver_id, ct in res["ciphertexts"].items():
                routed_ciphertexts[receiver_id][sender_id] = ct
                
        t_r1 = time.time() - t_start_r1
        self.comm_bytes_round1 += self._get_msg_size(req_msg_r1) * N

        # ==========================================================
        # Round 2: MaskedInputCollection (同时模拟掉线)
        # ==========================================
        print(f"[3/4] Round 2: 收集加噪梯度 (设定掉线率 {self.drop_rate*100}%)...")
        t_start_r2 = time.time()
        
        msg_dicts_r2 = {
            cid: {"action": "REQ_ROUND2", "ciphertexts_dict": routed_ciphertexts[cid]}
            for cid in active_ids
        }
        self._send_custom_to_targets(msg_dicts_r2, active_ids)
        r2_results = self._recv_from_targets(active_ids)
        
        for cid in active_ids:
            self.comm_bytes_round2 += self._get_msg_size(msg_dicts_r2[cid])
        
        num_dropped = int(N * self.drop_rate)
        dropped_ids = random.sample(active_ids, num_dropped) if num_dropped > 0 else []
        u3_ids = [cid for cid in active_ids if cid not in dropped_ids]
        u3_ids.sort()
        
        y_u_list = []
        for cid in u3_ids:
            res = r2_results[cid]
            y_u_list.append(res["y_u"])
            self.comm_bytes_round2 += self._get_msg_size(res)
            
        t_r2 = time.time() - t_start_r2

        # ==========================================================
        # Round 4: Unmasking
        # ==========================================
        print(f"[4/4] Round 4: 收集交叉分片，执行拉格朗日插值解密与明文重构...")
        t_start_r4 = time.time()
        
        req_msg_r4 = {"action": "REQ_ROUND4", "u3_ids": u3_ids}
        self._send_to_targets(req_msg_r4, u3_ids)
        r4_results = self._recv_from_targets(u3_ids)
        
        b_u_shares = {u: {} for u in u3_ids}         
        s_sk_shares = {u: {} for u in dropped_ids}   
        
        for sender_id, res in r4_results.items():
            self.comm_bytes_round4 += self._get_msg_size(res)
            for target_id, share_info in res["shares"].items():
                x_coord = sender_id + 1 
                if share_info["type"] == "b_u":
                    b_u_shares[target_id][x_coord] = share_info["share"]
                elif share_info["type"] == "s_sk":
                    s_sk_shares[target_id][x_coord] = share_info["share"]
                    
        self.comm_bytes_round4 += self._get_msg_size(req_msg_r4) * len(u3_ids)

        print("      >> 正在消除噪声，还原明文...")
        t_start_agg = time.time()
        
        Z = np.zeros(self.param_size, dtype=np.uint64)
        for y in y_u_list:
            Z = (Z + y.astype(np.uint64)) % np.uint64(MOD)
        Z = Z.astype(np.int64)
        
        for u in u3_ids:
            if len(b_u_shares[u]) >= threshold:
                b_u_int = CryptoUtils.reconstruct_secret(b_u_shares[u])
                b_u_bytes = b_u_int.to_bytes(16, 'big')
                p_u = CryptoUtils.generate_mask(b_u_bytes, self.param_size, MOD)
                Z = (Z - p_u) % MOD
                
        for u in dropped_ids:
            if len(s_sk_shares[u]) >= threshold:
                s_sk_int = CryptoUtils.reconstruct_secret(s_sk_shares[u])
                
                for v in u3_ids:
                    s_uv = CryptoUtils.agree(s_sk_int, self.public_keys[v][1], length=16)
                    p_uv = CryptoUtils.generate_mask(s_uv, self.param_size, MOD)
                    
                    if u > v:
                        Z = (Z.astype(np.uint64) + p_uv.astype(np.uint64)) % np.uint64(MOD)
                        Z = Z.astype(np.int64)
                    elif u < v:
                        Z = (Z - p_uv) % MOD
                        
        threshold_neg = MOD // 2
        result_float = Z.astype(np.float64)
        result_float[Z > threshold_neg] -= float(MOD)
        result_float /= SCALE
        
        t_agg = time.time() - t_start_agg
        t_r4 = (time.time() - t_start_r4) - t_agg 

        comm_r0_kb = self.comm_bytes_round0 / 1024.0 / N
        comm_r1_kb = self.comm_bytes_round1 / 1024.0 / N
        comm_r2_kb = self.comm_bytes_round2 / 1024.0 / len(u3_ids) if u3_ids else 0
        comm_r4_kb = self.comm_bytes_round4 / 1024.0 / len(u3_ids) if u3_ids else 0
        total_comm_kb = comm_r0_kb + comm_r1_kb + comm_r2_kb + comm_r4_kb

        print("\n============= Communication Report (Double-Masking) =============")
        print("【通信量评估 (单客户端均值)】")
        print(f" 1. Round 0 (密钥广播)     : {comm_r0_kb:.2f} KB")
        print(f" 2. Round 1 (分片加密路由) : {comm_r1_kb:.2f} KB")
        print(f" 3. Round 2 (掩码模型上传) : {comm_r2_kb:.2f} KB")
        print(f" 4. Round 4 (掉线恢复收集) : {comm_r4_kb:.2f} KB")
        print(" -----------------------------------------------------------------")
        print(f" 📊 总通信开销             : {total_comm_kb:.2f} KB")

        print("\n============= Time Report (Double-Masking) =============")
        print(f" [配置] 总客户端: {N} | 参与聚合数: {len(u3_ids)} | 参数维度: {self.param_size}")
        print(" -----------------------------------------------------------------")
        print(f" 1. Round 0 耗时 (密钥交换)   : {t_r0 / N:.4f} s/client")
        print(f" 2. Round 1 耗时 (分片生成)   : {t_r1 / N:.4f} s/client")
        print(f" 3. Round 2 耗时 (掩码应用)   : {t_r2 / N:.4f} s/client")
        print(f" 4. Round 4 耗时 (分片收集)   : {t_r4 / len(u3_ids) if u3_ids else 0:.4f} s/client")
        print(f" 5. Server 聚合与解密计算耗时 : {t_agg:.4f} s")
        print(" =================================================================")
        print(f" 🚀 总耗时 (掩码闭环全流程)   : {(t_r0 + t_r1 + t_r2 + t_r4)/N + t_agg:.4f} s (对比 TEE)")
        print("==================================================================\n")
        
        for sock in self.client_sockets.values(): sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    args = parser.parse_args()
    
    server = ServerSimulator(port=args.port, num_clients=args.num_clients, param_size=args.param_size, drop_rate=args.drop_rate)
    
    import subprocess
    client_process = subprocess.Popen([
        "python", "client_simulator.py", 
        "--num_clients", f"{args.num_clients}", 
        "--param_size", f"{args.param_size}", 
        "--port", f"{args.port}"
    ])

    server.wait_for_clients()
    server.run_simulation_round()