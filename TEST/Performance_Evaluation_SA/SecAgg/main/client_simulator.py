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
        
        self.c_sk = None        
        self.c_pk = None        
        self.s_sk = None        
        self.s_pk = None        
        self.b_u = None         
        
        self.others_keys = {}         
        self.shares_from_others = {}  
        self.u1_ids = []  
        self.u2_ids = []  

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.server_ip, self.port))
        except Exception as e:
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
                    net.send_msg(self.sock, {
                        "action": "RES_ROUND0",
                        "c_pk": self.c_pk,
                        "s_pk": self.s_pk
                    })

                elif action == "SYNC_PUBKEYS":
                    self.others_keys = msg.get("public_keys_dict")
                    self.u1_ids = list(self.others_keys.keys())
                    net.send_msg(self.sock, {"action": "ACK"})

                elif action == "REQ_ROUND1":
                    t = msg.get("threshold")
                    
                    # b_u 是作为 AES-CTR 种子的，指定为 128 位 (16 字节)
                    self.b_u = int.from_bytes(os.urandom(16), 'big')
                    # s_sk 已经是 int，直接使用
                    s_sk_int = self.s_sk
                    
                    max_id = max(self.u1_ids) + 1
                    b_shares = CryptoUtils.share_secret(self.b_u, t, max_id)
                    s_shares = CryptoUtils.share_secret(s_sk_int, t, max_id)
                    
                    ciphertexts = {}
                    for v in self.u1_ids:
                        if v == self.client_id: continue
                        # 导出 16 字节的密钥给 AES-128-GCM 使用
                        c_uv = CryptoUtils.agree(self.c_sk, self.others_keys[v][0], length=16)
                        pt = pickle.dumps((s_shares[v+1], b_shares[v+1]))
                        ct = CryptoUtils.encrypt(c_uv, pt)
                        ciphertexts[v] = ct
                        
                    net.send_msg(self.sock, {"action": "RES_ROUND1", "ciphertexts": ciphertexts})

                elif action == "REQ_ROUND2":
                    ciphertexts_from_others = msg.get("ciphertexts_dict")
                    self.u2_ids = list(ciphertexts_from_others.keys()) + [self.client_id]
                    self.u2_ids.sort()
                    
                    self.shares_from_others = {}
                    for u, ct in ciphertexts_from_others.items():
                        c_vu = CryptoUtils.agree(self.c_sk, self.others_keys[u][0], length=16)
                        pt = CryptoUtils.decrypt(c_vu, ct)
                        self.shares_from_others[u] = pickle.loads(pt)
                        
                    b_u_bytes = self.b_u.to_bytes(16, 'big')
                    p_u = CryptoUtils.generate_mask(b_u_bytes, self.param_size, MOD)
                    
                    for v in self.u2_ids:
                        if v == self.client_id: continue
                        # 导出 16 字节的种子给 AES-CTR 使用
                        s_uv = CryptoUtils.agree(self.s_sk, self.others_keys[v][1], length=16)
                        p_uv = CryptoUtils.generate_mask(s_uv, self.param_size, MOD)
                        
                        if self.client_id > v:
                            p_u = (p_u.astype(np.uint64) + p_uv.astype(np.uint64)) % np.uint64(MOD)
                            p_u = p_u.astype(np.int64)
                        else:
                            p_u = (p_u - p_uv) % MOD
                            
                    w_int = (self.w_new_cache * SCALE).astype(np.int64) % MOD
                    y_u = (w_int.astype(np.uint64) + p_u.astype(np.uint64)) % np.uint64(MOD)
                    
                    net.send_msg(self.sock, {"action": "RES_ROUND2", "y_u": y_u.astype(np.int64)})

                elif action == "REQ_ROUND4":
                    u3_ids = msg.get("u3_ids")
                    dropped_ids = [v for v in self.u2_ids if v not in u3_ids]
                    
                    shares_to_send = {}
                    for v in self.u2_ids:
                        if v == self.client_id: continue
                        if v in dropped_ids:
                            shares_to_send[v] = {"type": "s_sk", "share": self.shares_from_others[v][0]}
                        elif v in u3_ids:
                            shares_to_send[v] = {"type": "b_u", "share": self.shares_from_others[v][1]}
                            
                    net.send_msg(self.sock, {"action": "RES_ROUND4", "shares": shares_to_send})

            except Exception as e:
                break
        self.sock.close()

def run_simulator(server_ip='127.0.0.1', port=8888, num_clients=20, param_size=1000000):
    print("==========================================")
    print("  Clients On (NIST P-256 Double-Masking Version)")
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
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    run_simulator(server_ip='127.0.0.1', port=args.port, num_clients=args.num_clients, param_size=args.param_size)