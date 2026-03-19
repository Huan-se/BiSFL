import sys
import os
import socket
import threading
import numpy as np
import time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from _utils_.tee_adapter import get_tee_adapter_singleton
import network_utils as net

sgx_semaphore = threading.Semaphore(10)

class VirtualClient:
    def __init__(self, server_ip, port, param_size):
        self.server_ip = server_ip
        self.port = port
        self.param_size = param_size
        
        self.client_id = None
        self.sock = None
        self.w_old_cache = np.zeros(param_size, dtype=np.float32)
        self.w_new_cache = None
        self.active_ids = []
        
        self.tee_adapter = get_tee_adapter_singleton()
        
        self.seed_mask_root = 0x12345678 
        self.seed_global_0 = 0x87654321  
        self.seed_sss = 0x11223344

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

                elif action == "REQ_RATLS_QUOTE":
                    # 模拟 TEE 内部调用 sgx_ecc256_create_key_pair 并生成 Quote
                    with sgx_semaphore:
                        time.sleep(0.02) # 硬件生成 Quote 的平均延迟
                        # 严格模拟 4384 字节的 DCAP Quote 和 64 字节的公钥
                        dummy_pub_key = os.urandom(64)
                        dummy_quote = os.urandom(4384)
                        
                    net.send_msg(self.sock, {
                        "action": "RES_RATLS_QUOTE",
                        "data": {"pub_key": dummy_pub_key, "quote": dummy_quote}
                    })

                elif action == "SYNC_SEEDS":
                    # 模拟 TEE 内部接收 Server 密文，并使用私钥 ECDH 解密
                    with sgx_semaphore:
                        time.sleep(0.002) # 极短的 AES 解密耗时
                    net.send_msg(self.sock, {"action": "RES_SEEDS_ACK"})
                    
                elif action == "SYNC_MODEL":
                    # [核心修正 2]：仅执行本地 Mock 训练，不包含任何 TEE 调用
                    self.w_new_cache = np.random.randn(self.param_size).astype(np.float32)
                    
                    # 告知 Server 训练完毕，触发下一步
                    net.send_msg(self.sock, {"action": "RES_TRAIN_DONE"})
                    
                elif action == "REQ_PROJ":
                    # [修复]：使用信号量保护，排队进入 SGX
                    with sgx_semaphore:
                        output, ranges = self.tee_adapter.prepare_gradient(
                            self.client_id, 12345, self.w_new_cache, self.w_old_cache
                        )
                    self.w_old_cache = self.w_new_cache.copy()
                    net.send_msg(self.sock, {"action": "RES_PROJ", "data": output})
                    
                elif action == "REQ_CIPHER":
                    # [接收 U1 列表用于生成掩码]
                    u1_ids = msg.get("u1_ids")
                    assigned_weight = msg.get("weight", 0.0)
                    
                    # [修复]：使用信号量保护
                    with sgx_semaphore:
                        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
                            self.seed_mask_root, self.seed_global_0, 
                            self.client_id, u1_ids, 
                            assigned_weight, self.param_size
                        )
                    net.send_msg(self.sock, {"action": "RES_CIPHER", "data": c_grad})
                    
                elif action == "REQ_SHARES":
                    # [接收 U1 和 U2 列表用于生成恢复多项式]
                    u1_ids = msg.get("u1_ids")
                    u2_ids = msg.get("u2_ids")
                    threshold = len(u1_ids) // 2 + 1
                    with sgx_semaphore:
                        shares = self.tee_adapter.get_vector_shares_dynamic(
                            self.seed_sss, self.seed_mask_root, 
                            u1_ids, u2_ids, 
                            self.client_id, threshold
                        )
                    net.send_msg(self.sock, {"action": "RES_SHARES", "data": shares})

            except Exception as e:
                print(f"[VirtualClient {self.client_id}] 发生错误: {e}")
                break
                
        self.sock.close()

def run_simulator(server_ip='127.0.0.1', port=8888, num_clients=20, param_size=1000000):
    print("==========================================")
    print("  Clients On")
    print("==========================================\n")
    
    tee_adapter = get_tee_adapter_singleton()
    tee_adapter.initialize_enclave()
    
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