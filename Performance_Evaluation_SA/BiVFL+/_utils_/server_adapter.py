import ctypes
import os
import numpy as np

class ServerAdapter:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 【修复路径】：指向 lib/libserver_core.so
        lib_path = os.path.join(current_dir, '..', 'lib', 'libserver_core.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find ServerCore library at {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.aggregate_and_unmask_sparse.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # U1
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # U2
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, # Flattened Shares
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int, # Flattened Ciphers & sizes
            ctypes.c_int, ctypes.c_int, # Total N, K_degree (For topology rebuilding in C++)
            ctypes.POINTER(ctypes.c_int64) # Output
        ]

    def aggregate_and_unmask_sparse(self, root_seed, global_seed, u1_ids, u2_ids, shares_list, ciphers_list, N_total, K_degree):
        u1_arr = np.array(u1_ids, dtype=np.int32)
        u2_arr = np.array(u2_ids, dtype=np.int32)
        
        # 将各客户端发来的稀疏标量分片展平以传入 C++
        flat_share_owners = []
        flat_share_vals = []
        for shares_sublist in shares_list:
            for dropped_id, share_val in shares_sublist:
                flat_share_owners.append(dropped_id)
                flat_share_vals.append(share_val)
                
        flat_owners_arr = np.array(flat_share_owners, dtype=np.int32)
        flat_vals_arr = np.array(flat_share_vals, dtype=np.uint64)
        num_shares = len(flat_share_vals)
        
        # 展平上传的 O(d) 维度密文矩阵
        flat_ciphers = np.concatenate(ciphers_list).astype(np.int64)
        num_ciphers = len(flat_ciphers)
        param_size = len(ciphers_list[0]) if ciphers_list else 0
        
        out_buffer = np.zeros(param_size, dtype=np.int64)
        
        self.lib.aggregate_and_unmask_sparse(
            root_seed, global_seed,
            u1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(u1_ids),
            u2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(u2_ids),
            flat_owners_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            flat_vals_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            num_shares,
            flat_ciphers.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            num_ciphers, param_size,
            N_total, K_degree,
            out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        )
        
        return out_buffer.tolist()