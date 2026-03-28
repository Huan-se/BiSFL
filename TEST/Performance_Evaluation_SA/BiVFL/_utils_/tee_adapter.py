import ctypes
import os
import numpy as np

class TEEAdapter:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, '..', 'lib', 'libtee_bridge.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find TEE library at {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # [修复] 增加 tee_ 前缀与 C++ 严格对齐
        self.lib.tee_generate_masked_gradient_sparse.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)
        ]
        
        self.lib.tee_get_scalar_shares_sparse.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t
        ]

        self.lib.tee_prepare_gradient.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

    def _to_bytes(self, val): return str(val).encode('utf-8')

    def initialize_enclave(self): return 0

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old_cache):
        param_size = len(w_new)
        w_new_ptr = w_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_old_ptr = w_old_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        out_buffer = np.zeros(1024, dtype=np.float32)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        self.lib.tee_prepare_gradient(client_id, proj_seed, param_size, w_new_ptr, w_old_ptr, out_ptr)
        return out_buffer.tolist(), None

    def generate_masked_gradient_sparse(self, kappa_m_str, t, model_hash_str, client_id, w_new, weight, param_size):
        out_buffer = np.zeros(param_size, dtype=np.int64)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        self.lib.tee_generate_masked_gradient_sparse(
            self._to_bytes(kappa_m_str), t, self._to_bytes(model_hash_str),
            client_id, w_new, weight, param_size, out_ptr
        )
        return out_buffer.tolist()

    def get_scalar_shares_sparse(self, kappa_s_str, kappa_m_str, t, view_hash_str, client_id, alive_neighbors, dropped_neighbors, threshold):
        num_alive = len(alive_neighbors)
        num_dropped = len(dropped_neighbors)
        
        alive_arr = np.array(alive_neighbors, dtype=np.int32)
        dropped_arr = np.array(dropped_neighbors, dtype=np.int32)
        
        alive_ptr = alive_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) if num_alive > 0 else None
        dropped_ptr = dropped_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) if num_dropped > 0 else None
        
        max_len = 4096
        out_buffer = np.zeros(max_len, dtype=np.int64)
        out_ptr = out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        
        self.lib.tee_get_scalar_shares_sparse(
            self._to_bytes(kappa_s_str), self._to_bytes(kappa_m_str), t, self._to_bytes(view_hash_str),
            client_id, alive_ptr, num_alive, dropped_ptr, num_dropped,
            threshold, out_ptr, max_len
        )
        
        count = int(out_buffer[0])
        structured_shares = []
        idx = 1
        for _ in range(count):
            if idx + 2 >= max_len: break
            structured_shares.append((int(out_buffer[idx]), int(out_buffer[idx+1]), int(out_buffer[idx+2])))
            idx += 3
        return structured_shares

_tee_adapter_instance = None
def get_tee_adapter_singleton():
    global _tee_adapter_instance
    if _tee_adapter_instance is None: _tee_adapter_instance = TEEAdapter()
    return _tee_adapter_instance