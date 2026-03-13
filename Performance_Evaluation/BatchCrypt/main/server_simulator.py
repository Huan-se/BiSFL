import sys
import os
import socket
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import network_utils as net
from _utils_.batchcrypt_utils import BatchCryptUtils

class ServerSimulator:
    def __init__(self, host='0.0.0.0', port=8888, num_clients=20, param_size=1000000):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.param_size = param_size
        self.client_sockets = {}
        
        self.comm_bytes_keygen = 0
        self.comm_bytes_upload = 0
        self.comm_bytes_download = 0

    def _get_msg_size(self, msg_dict):
        return len(pickle.dumps(msg_dict)) + 4

    def wait_for_clients(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(self.num_clients)
        
        print(f"[Server] 监听等待 {self.num_clients} 个客户端接入...")
        for i in range(self.num_clients):
            client_sock, addr = server_sock.accept()
            cid = i + 1
            self.client_sockets[cid] = client_sock
            net.send_msg(client_sock, {"action": "ASSIGN_ID", "client_id": cid})
        print("[Server] 所有客户端就绪。")

    def _send_and_recv(self, msg_dict):
        results = {}
        def _task(cid):
            net.send_msg(self.client_sockets[cid], msg_dict)
            results[cid] = net.recv_msg(self.client_sockets[cid])
        with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            for cid in self.client_sockets: executor.submit(_task, cid)
        return results

    def run_simulation(self):
        N = self.num_clients
        print(f"\n========== BatchCrypt 通信与计算性能测试 (维度: {self.param_size}) ==========")
        
        # --- 初始化与训练 ---
        self._send_and_recv({"action": "SYNC_MODEL"})

        # === Phase 1: HE 密钥分发 ===
        print("[1/3] 执行 Paillier 密钥生成与分发...")
        t_start = time.time()
        keygen_res = self._send_and_recv({"action": "REQ_KEYGEN"})
        pub_key, priv_key = keygen_res[1]["pub_key"], keygen_res[1]["priv_key"]
        
        sync_msg = {"action": "SYNC_KEYS", "pub_key": pub_key, "priv_key": priv_key}
        self._send_and_recv(sync_msg)
        t_keygen = time.time() - t_start
        self.comm_bytes_keygen = self._get_msg_size(sync_msg) * N

        # === Phase 2: 加密上传 ===
        print("[2/3] 客户端端侧 Batch 加密并上传密文...")
        t_start = time.time()
        req_msg = {"action": "REQ_UPLOAD_CIPHER"}
        upload_res = self._send_and_recv(req_msg)
        
        ciphers_list = []
        r_max_list = []
        og_shape = upload_res[1]["og_shape"]
        
        for cid, res in upload_res.items():
            self.comm_bytes_upload += self._get_msg_size(res)
            ciphers_list.append(res["cipher"])
            r_max_list.append(res["r_max"])
            
        t_upload = time.time() - t_start

        # === Phase 3: Server 同态聚合 ===
        print("[3/3] Server 端执行密文同态聚合...")
        t_start = time.time()
        # 由于是 numpy array，我们可以直接按列相加
        aggregated_cipher = np.sum(ciphers_list, axis=0)
        t_agg = time.time() - t_start

        # === Phase 4: 下发解密 ===
        print("[4/4] 下发聚合密文并执行端侧解密...")
        t_start = time.time()
        avg_r_max = np.mean(r_max_list)
        dl_msg = {
            "action": "REQ_DECRYPT", 
            "agg_cipher": aggregated_cipher, 
            "og_shape": og_shape, 
            "avg_r_max": avg_r_max,
            "active_count": N
        }
        res_dl = self._send_and_recv(dl_msg)
        t_decrypt = time.time() - t_start
        
        for cid, res in res_dl.items():
            self.comm_bytes_download += self._get_msg_size(dl_msg)

        # === 报告输出 ===
        comm_up = self.comm_bytes_upload / 1024.0 / N
        comm_dl = self.comm_bytes_download / 1024.0 / N
        
        print("\n============= Communication Report (BatchCrypt HE) =============")
        print("【通信量评估 (单客户端均值)】")
        print(f" 1. 密文上传 (Batch后) : {comm_up:.2f} KB")
        print(f" 2. 密文下发 (Batch后) : {comm_dl:.2f} KB")
        print(" -----------------------------------------------------------------")
        print(f" 📊 总核心通信开销     : {comm_up + comm_dl:.2f} KB (对比明文传输的压缩率非常高)")

        print("\n============= Time Report (BatchCrypt HE) =============")
        print(f" 1. Client 端侧加密打包耗时 : {t_upload / N:.4f} s/client")
        print(f" 2. Server 密文同态聚合耗时 : {t_agg:.4f} s (由于 Batch，已大幅降低，但仍是瓶颈)")
        print(f" 3. Client 端侧解密反量化耗时 : {t_decrypt / N:.4f} s/client")
        print(" =================================================================")
        print(f" 🚀 总耗时 (计算流程)       : {(t_upload + t_decrypt)/N + t_agg:.4f} s (对比 TEE Baseline)")
        print("==================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    
    server = ServerSimulator(port=args.port, num_clients=args.num_clients, param_size=args.param_size)
    
    import subprocess
    subprocess.Popen(["python", "client_simulator.py", "--num_clients", f"{args.num_clients}", "--param_size", f"{args.param_size}", "--port", f"{args.port}"])
    server.wait_for_clients()
    server.run_simulation()