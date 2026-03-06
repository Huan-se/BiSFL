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

# 系统常量，与服务端保持一致
MOD = 9223372036854775783
SCALE = 100000000.0

class VirtualClient:
    def __init__(self, server_ip, port, param_size):
        self.server_ip = server_ip
        self.port = port
        self.param_size = param_size
        
        self.client_id = None
        self.sock = None
        
        # 模拟训练缓存
        self.w_new_cache = None
        
        # === 密码学状态机参数 ===
        self.c_sk = None        # 用于加密通信的私钥
        self.c_pk = None        # 用于加密通信的公钥
        self.s_sk = None        # 用于掩码生成的私钥
        self.s_pk = None        # 用于掩码生成的公钥
        self.b_u = None         # 自身的私密随机种子 (自掩码)
        
        self.others_keys = {}         # 存储其他人的公钥 {uid: (c_pk, s_pk)}
        self.shares_from_others = {}  # 解密后收到的分片 {uid: (s_share, b_share)}
        
        self.u1_ids = []  # 参与 Round 1 的用户
        self.u2_ids = []  # 参与 Round 2 的用户

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
                    # 模拟本地训练过程，生成新模型参数
                    self.w_new_cache = np.random.randn(self.param_size).astype(np.float32)
                    net.send_msg(self.sock, {"action": "RES_TRAIN_DONE"})

                # ==========================================
                # Round 0: AdvertiseKeys
                # ==========================================
                elif action == "REQ_ROUND0":
                    # 生成两对 ECDH 密钥对
                    self.c_sk, self.c_pk = CryptoUtils.generate_key_pair()
                    self.s_sk, self.s_pk = CryptoUtils.generate_key_pair()
                    
                    net.send_msg(self.sock, {
                        "action": "RES_ROUND0",
                        "c_pk": self.c_pk,
                        "s_pk": self.s_pk
                    })

                # ==========================================
                # Round 1: ShareKeys
                # ==========================================
                elif action == "REQ_ROUND1":
                    self.others_keys = msg.get("public_keys_dict")
                    self.u1_ids = list(self.others_keys.keys())
                    t = msg.get("threshold")
                    
                    # 生成本地自掩码种子 b_u
                    self.b_u = int.from_bytes(os.urandom(16), 'big') % PRIME
                    s_sk_int = CryptoUtils.bytes_to_int(self.s_sk)
                    
                    # 秘密分片 (根据最大 ID 确定生成的分片数量)
                    max_id = max(self.u1_ids) + 1
                    b_shares = CryptoUtils.share_secret(self.b_u, t, max_id)
                    s_shares = CryptoUtils.share_secret(s_sk_int, t, max_id)
                    
                    ciphertexts = {}
                    for v in self.u1_ids:
                        if v == self.client_id: 
                            continue
                        
                        # 1. 计算与 v 的对称加密密钥 c_uv
                        c_uv = CryptoUtils.agree(self.c_sk, self.others_keys[v][0])
                        
                        # 2. 序列化需要发给 v 的分片数据 (注意分片索引用 v+1)
                        pt = pickle.dumps((s_shares[v+1], b_shares[v+1]))
                        
                        # 3. 认证加密
                        ct = CryptoUtils.encrypt(c_uv, pt)
                        ciphertexts[v] = ct
                        
                    net.send_msg(self.sock, {"action": "RES_ROUND1", "ciphertexts": ciphertexts})

                # ==========================================
                # Round 2: MaskedInputCollection
                # ==========================================
                elif action == "REQ_ROUND2":
                    ciphertexts_from_others = msg.get("ciphertexts_dict")
                    self.u2_ids = list(ciphertexts_from_others.keys()) + [self.client_id]
                    self.u2_ids.sort()
                    
                    # 1. 解密收到的分片
                    self.shares_from_others = {}
                    for u, ct in ciphertexts_from_others.items():
                        c_vu = CryptoUtils.agree(self.c_sk, self.others_keys[u][0])
                        pt = CryptoUtils.decrypt(c_vu, ct)
                        self.shares_from_others[u] = pickle.loads(pt)
                        
                    # 2. 利用 b_u 生成自身强干扰掩码 p_u
                    b_u_bytes = CryptoUtils.int_to_bytes(self.b_u, 32)
                    p_u = CryptoUtils.generate_mask(b_u_bytes, self.param_size, MOD)
                    
                    # 3. 利用协商密钥生成成对掩码 p_uv，并合并
                    for v in self.u2_ids:
                        if v == self.client_id: 
                            continue
                        
                        s_uv = CryptoUtils.agree(self.s_sk, self.others_keys[v][1])
                        p_uv = CryptoUtils.generate_mask(s_uv, self.param_size, MOD)
                        
                        # 根据大小关系决定加法还是减法，确保最终聚合时成对掩码相互抵消
                        if self.client_id > v:
                            p_u = (p_u + p_uv) % MOD
                        else:
                            p_u = (p_u - p_uv) % MOD
                            
                    # 4. 模型量化与加噪
                    w_int = (self.w_new_cache * SCALE).astype(np.int64) % MOD
                    y_u = (w_int + p_u) % MOD
                    
                    net.send_msg(self.sock, {"action": "RES_ROUND2", "y_u": y_u})

                # ==========================================
                # Round 4: Unmasking (跳过了可选的 Round 3)
                # ==========================================
                elif action == "REQ_ROUND4":
                    u3_ids = msg.get("u3_ids")
                    dropped_ids = [v for v in self.u2_ids if v not in u3_ids]
                    
                    shares_to_send = {}
                    for v in self.u2_ids:
                        if v == self.client_id: 
                            continue
                            
                        # 如果目标掉线，交出其 s_sk 的分片；如果目标存活，交出其 b_u 的分片
                        if v in dropped_ids:
                            shares_to_send[v] = {"type": "s_sk", "share": self.shares_from_others[v][0]}
                        elif v in u3_ids:
                            shares_to_send[v] = {"type": "b_u", "share": self.shares_from_others[v][1]}
                            
                    net.send_msg(self.sock, {"action": "RES_ROUND4", "shares": shares_to_send})

            except Exception as e:
                print(f"[VirtualClient {self.client_id}] 发生错误或断开连接: {e}")
                break
                
        self.sock.close()


def run_simulator(server_ip='127.0.0.1', port=8888, num_clients=20, param_size=1000000):
    print("==========================================")
    print("  Clients On (Double-Masking Version)")
    print("==========================================\n")
    
    threads = []
    for i in range(num_clients):
        client = VirtualClient(server_ip, port, param_size)
        t = threading.Thread(target=client.run, daemon=True)
        t.start()
        threads.append(t)
        
    for t in threads: 
        t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--param_size", type=int, default=1000000)
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    run_simulator(server_ip='127.0.0.1', port=args.port, num_clients=args.num_clients, param_size=args.param_size)