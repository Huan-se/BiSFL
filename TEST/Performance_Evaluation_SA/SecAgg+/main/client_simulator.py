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
from _utils_.crypto_utils import CryptoUtils, PRIME

MOD = 9223372036854775783
SCALE = 100000000.0

class VirtualClient:
    def __init__(self, server_ip, port, param_size):
        self.server_ip = server_ip
        self.port = port
        self.param_size = param_size
        
        self.client_id = None
        self.sock = None
        self.w_new_cache = None
        
        self.c_sk, self.c_pk = None, None
        self.s_sk, self.s_pk = None, None
        self.b_u = None         
        
        self.neighbor_keys = {}         
        self.shares_from_others = {}  
        self.neighbors = []  

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.server_ip, self.port))
        except Exception as e:
            print(f"[VirtualClient] 连接失败: {e}")
            return

        while True:
            try:
                msg = net.recv_msg(self.sock)
                if not msg: break
                
                action = msg.get("action")
                
                if action == "ASSIGN_ID":
                    self.client_id = msg.get("client_id")
                    
                elif action == "SYNC_MODEL":
                    self.w_new_cache = np.random.randn(self.param_size).astype(np.float32)
                    net.send_msg(self.sock, {"action": "RES_TRAIN_DONE"})

                elif action == "REQ_ROUND0":
                    self.c_sk, self.c_pk = CryptoUtils.generate_key_pair()
                    self.s_sk, self.s_pk = CryptoUtils.generate_key_pair()
                    net.send_msg(self.sock, {"action": "RES_ROUND0", "c_pk": self.c_pk, "s_pk": self.s_pk})

                elif action == "REQ_ROUND1":
                    self.neighbor_keys = msg.get("neighbor_keys")
                    self.neighbors = list(self.neighbor_keys.keys())
                    t = msg.get("threshold")
                    
                    self.b_u = int.from_bytes(os.urandom(8), 'big') % PRIME
                    s_sk_int = CryptoUtils.bytes_to_int(self.s_sk)
                    
                    # SecAgg+: 仅对图邻居生成多项式分片
                    b_shares = CryptoUtils.share_secret(self.b_u, t, self.neighbors)
                    s_shares = CryptoUtils.share_secret(s_sk_int, t, self.neighbors)
                    
                    ciphertexts = {}
                    for v in self.neighbors:
                        c_uv = CryptoUtils.agree(self.c_sk, self.neighbor_keys[v][0])
                        pt = pickle.dumps((s_shares[v], b_shares[v]))
                        ct = CryptoUtils.encrypt(c_uv, pt)
                        ciphertexts[v] = ct
                        
                    net.send_msg(self.sock, {"action": "RES_ROUND1", "ciphertexts": ciphertexts})

                elif action == "REQ_ROUND2":
                    ciphertexts_from_others = msg.get("ciphertexts_dict")
                    self.active_neighbors_r2 = list(ciphertexts_from_others.keys())
                    
                    self.shares_from_others = {}
                    for u, ct in ciphertexts_from_others.items():
                        c_vu = CryptoUtils.agree(self.c_sk, self.neighbor_keys[u][0])
                        pt = CryptoUtils.decrypt(c_vu, ct)
                        self.shares_from_others[u] = pickle.loads(pt)
                        
                    b_u_bytes = CryptoUtils.int_to_bytes(self.b_u, 32)
                    p_u = CryptoUtils.generate_mask(b_u_bytes, self.param_size, MOD)
                    
                    for v in self.active_neighbors_r2:
                        s_uv = CryptoUtils.agree(self.s_sk, self.neighbor_keys[v][1])
                        p_uv = CryptoUtils.generate_mask(s_uv, self.param_size, MOD)
                        
                        if self.client_id > v: p_u = (p_u + p_uv) % MOD
                        else: p_u = (p_u - p_uv) % MOD
                            
                    w_int = (self.w_new_cache * SCALE).astype(np.int64) % MOD
                    y_u = (w_int + p_u) % MOD
                    net.send_msg(self.sock, {"action": "RES_ROUND2", "y_u": y_u})

                elif action == "REQ_ROUND4":
                    u3_ids = msg.get("u3_ids")
                    dropped_neighbors = [v for v in self.active_neighbors_r2 if v not in u3_ids]
                    
                    shares_to_send = {}
                    for v in self.active_neighbors_r2:
                        if v in dropped_neighbors:
                            shares_to_send[v] = {"type": "s_sk", "share": self.shares_from_others[v][0]}
                        elif v in u3_ids:
                            shares_to_send[v] = {"type": "b_u", "share": self.shares_from_others[v][1]}
                            
                    net.send_msg(self.sock, {"action": "RES_ROUND4", "shares": shares_to_send})

            except Exception as e:
                print(f"[VirtualClient {self.client_id}] 断开连接: {e}")
                break
                
        self.sock.close()

def run_simulator(server_ip='127.0.0.1', port=8888, num_clients=20, param_size=1000000):
    print("==========================================")
    print("  Clients On (SecAgg+ Sparse Version)")
    print("==========================================\n")
    
    threads = []
    for i in range(num_clients):
        client = VirtualClient(server_ip, port, param_size)
        t = threading.Thread(target=client.run, daemon=True)
        t.start()
        threads.append(t)
        
    for t in threads: t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8889)
    args = parser.parse_args()
    run_simulator(server_ip='127.0.0.1', port=args.port, num_clients=args.num_clients, param_size=args.param_size)