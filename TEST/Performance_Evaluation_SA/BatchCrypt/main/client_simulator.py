import sys
import os
import socket
import threading
import numpy as np
import time
import argparse
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import network_utils as net
from _utils_.batchcrypt_utils import BatchCryptUtils

class VirtualClient:
    def __init__(self, server_ip, port, param_size, num_clients):
        self.server_ip = server_ip
        self.port = port
        self.param_size = param_size
        self.client_id = None
        
        self.bc_utils = BatchCryptUtils(num_clients=num_clients, bit_width=16)
        self.pub_key = None
        self.priv_key = None
        self.w_new_cache = None

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))

        while True:
            msg = net.recv_msg(self.sock)
            if not msg: break
            
            action = msg.get("action")
            
            if action == "ASSIGN_ID":
                self.client_id = msg.get("client_id")
                
            elif action == "SYNC_MODEL":
                self.w_new_cache = np.random.randn(self.param_size).astype(np.float32)
                net.send_msg(self.sock, {"action": "RES_TRAIN_DONE"})

            # === Phase 1: 密钥协商 (选定 Client 1 生成全局 HE 密钥) ===
            elif action == "REQ_KEYGEN":
                if self.client_id == 1:
                    print(f"[Client 1] 正在生成 Paillier 2048 位密钥对 (这会花费一些时间)...")
                    pub, priv = BatchCryptUtils.generate_keypair()
                    net.send_msg(self.sock, {"action": "RES_KEYGEN", "pub_key": pub, "priv_key": priv})
                else:
                    net.send_msg(self.sock, {"action": "RES_KEYGEN"})
                    
            elif action == "SYNC_KEYS":
                self.pub_key = msg.get("pub_key")
                self.priv_key = msg.get("priv_key")
                net.send_msg(self.sock, {"action": "ACK"})

            # === Phase 2: BatchCrypt 梯度量化、打包与同态加密 ===
            elif action == "REQ_UPLOAD_CIPHER":
                # 计算本地梯度 (简化为自身的新模型)
                local_grad = self.w_new_cache
                
                # 调用 BatchCryptUtils 进行加密
                encrypted_batch, og_shape, r_max = self.bc_utils.encrypt_gradients(self.pub_key, local_grad)
                
                net.send_msg(self.sock, {
                    "action": "RES_UPLOAD_CIPHER", 
                    "cipher": encrypted_batch,
                    "og_shape": og_shape,
                    "r_max": r_max
                })

            # === Phase 3: 接收聚合密文并解密 ===
            elif action == "REQ_DECRYPT":
                agg_cipher = msg.get("agg_cipher")
                og_shape = msg.get("og_shape")
                avg_r_max = msg.get("avg_r_max")  # 近似使用平均 r_max
                active_count = msg.get("active_count")
                
                # 解密并反量化为明文梯度
                plain_grad = self.bc_utils.decrypt_and_unmask(
                    self.priv_key, agg_cipher, og_shape, avg_r_max, active_count
                )
                
                net.send_msg(self.sock, {"action": "RES_DECRYPT_DONE"})

def run_simulator(server_ip='127.0.0.1', port=8888, num_clients=20, param_size=1000000):
    print("==========================================")
    print("  Clients On (BatchCrypt HE Version)")
    print("==========================================\n")
    
    threads = []
    for i in range(num_clients):
        client = VirtualClient(server_ip, port, param_size, num_clients)
        t = threading.Thread(target=client.run, daemon=True)
        t.start()
        threads.append(t)
        
    for t in threads: t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    run_simulator(port=args.port, num_clients=args.num_clients, param_size=args.param_size)