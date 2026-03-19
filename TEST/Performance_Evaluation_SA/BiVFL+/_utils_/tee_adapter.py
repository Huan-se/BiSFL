import ctypes
import os
import numpy as np

class TEEAdapter:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 【修复路径】：指向 lib/libtee_bridge.so
        lib_path = os.path.join(current_dir, '..', 'lib', 'libtee_bridge.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find TEE library at {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # 注册新的稀疏掩码生成接口
        self.lib.generate_masked_gradient_sparse.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, 
            ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)
        ]
        
        # 注册新的本地确定性标量分享推导接口
        self.lib.get_scalar_shares_sparse.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)
        ]

        self.lib.prepare_gradient.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

    def initialize_enclave(self):
        return 0

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old_cache):
        param_size = len(w_new)
        w_new_ptr = w_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_old_ptr = w_old_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        K_dim = 1024
        out_buffer = np.zeros(K_dim, dtype=np.float32)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        self.lib.prepare_gradient(client_id, proj_seed, param_size, w_new_ptr, w_old_ptr, out_ptr)
        return out_buffer.tolist(), None

    def generate_masked_gradient_sparse(self, root_seed, global_seed, client_id, active_neighbors, weight, param_size):
        """仅需 O(log N) 邻接数据的极速掩码生成"""
        num_neighbors = len(active_neighbors)
        neighbors_arr = np.array(active_neighbors, dtype=np.int32)
        neighbors_ptr = neighbors_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        out_buffer = np.zeros(param_size, dtype=np.int64)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        self.lib.generate_masked_gradient_sparse(
            root_seed, global_seed, client_id,
            neighbors_ptr, num_neighbors, 
            weight, param_size, out_ptr
        )
        return out_buffer.tolist()

    def get_scalar_shares_sparse(self, seed_sss, root_seed, dropped_neighbors, client_id, threshold):
        """零前置通信：本地利用 TEE 根种子推导掉线邻居的标量分片"""
        num_dropped = len(dropped_neighbors)
        if num_dropped == 0: return []
        
        dropped_arr = np.array(dropped_neighbors, dtype=np.int32)
        dropped_ptr = dropped_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # 返回的是标量分片，用 uint64 承载
        out_buffer = np.zeros(num_dropped, dtype=np.uint64)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        
        self.lib.get_scalar_shares_sparse(
            seed_sss, root_seed, dropped_ptr, num_dropped,
            client_id, threshold, out_ptr
        )
        
        # 返回格式：[(掉线ID, 标量分片值), ...]，极度轻量
        return [(dropped_neighbors[i], int(out_buffer[i])) for i in range(num_dropped)]

_tee_adapter_instance = None
def get_tee_adapter_singleton():
    global _tee_adapter_instance
    if _tee_adapter_instance is None:
        _tee_adapter_instance = TEEAdapter()
    return _tee_adapter_instance