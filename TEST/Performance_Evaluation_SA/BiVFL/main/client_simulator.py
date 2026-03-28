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
        
        self.tee_adapter = get_tee_adapter_singleton()

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
                    with sgx_semaphore:
                        time.sleep(0.02) 
                        dummy_pub_key = os.urandom(64)
                        dummy_quote = os.urandom(4384)
                    net.send_msg(self.sock, {"action": "RES_RATLS_QUOTE", "data": {"pub_key": dummy_pub_key, "quote": dummy_quote}})

                elif action == "SYNC_SEEDS":
                    with sgx_semaphore:
                        time.sleep(0.002) 
                    net.send_msg(self.sock, {"action": "RES_SEEDS_ACK"})
                    
                elif action == "SYNC_MODEL":
                    self.w_new_cache = np.random.randn(self.param_size).astype(np.float32)
                    net.send_msg(self.sock, {"action": "RES_TRAIN_DONE"})
                    
                elif action == "REQ_PROJ":
                    with sgx_semaphore:
                        output, ranges = self.tee_adapter.prepare_gradient(self.client_id, 12345, self.w_new_cache, self.w_old_cache)
                    self.w_old_cache = self.w_new_cache.copy()
                    output_np = np.array(output, dtype=np.float32)
                    net.send_msg(self.sock, {"action": "RES_PROJ", "data": output_np})
                
                elif action == "SKIP_PROJ":
                    self.w_old_cache = self.w_new_cache.copy()
                    net.send_msg(self.sock, {"action": "RES_SKIP_PROJ"})
                    
                elif action == "REQ_CIPHER":
                    # [精准保留 O(1) 密文传输请求] 不再需要发送庞大的 active_neighbors
                    assigned_weight = msg.get("weight", 0.0)
                    kappa_m_str = msg.get("kappa_m", "0")
                    t = msg.get("t", 1)
                    model_hash_str = msg.get("model_hash", "0")
                    
                    with sgx_semaphore:
                        c_grad = self.tee_adapter.generate_masked_gradient_sparse(
                            kappa_m_str, t, model_hash_str, self.client_id, self.w_new_cache, assigned_weight, self.param_size
                        )
                    c_grad_np = np.array(c_grad, dtype=np.int64)
                    net.send_msg(self.sock, {"action": "RES_CIPHER", "data": c_grad_np})
                    
                elif action == "REQ_SHARES":
                    # [精准保留 O(log N) SSS传输请求] 仅接受邻居状态
                    alive_neighbors = msg.get("alive_neighbors", [])
                    dropped_neighbors = msg.get("dropped_neighbors", [])
                    threshold = msg.get("threshold", 3)
                    kappa_s_str = msg.get("kappa_s", "0")
                    kappa_m_str = msg.get("kappa_m", "0")
                    t = msg.get("t", 1)
                    view_hash_str = msg.get("view_hash", "0")
                    
                    with sgx_semaphore:
                        shares = self.tee_adapter.get_scalar_shares_sparse(
                            kappa_s_str, kappa_m_str, t, view_hash_str, self.client_id, 
                            alive_neighbors, dropped_neighbors, threshold
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