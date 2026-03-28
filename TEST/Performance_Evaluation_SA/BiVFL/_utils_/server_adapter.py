import ctypes
import os
import numpy as np

class ServerAdapter:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, '..', 'lib', 'libserver_core.so')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find ServerCore library at {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.ta_offline_compute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)
        ]
        
        self.lib.aggregate_and_unmask_sparse.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,  
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int, 
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int, 
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float)
        ]

    def _to_bytes(self, val): return str(val).encode('utf-8')

    def ta_offline_compute(self, u1_ids, kappa_m_str, t, param_size):
        u1_arr = np.array(u1_ids, dtype=np.int32)
        out_s_alpha = np.zeros(param_size, dtype=np.int64)
        out_s_beta = np.zeros(param_size, dtype=np.int64)
        
        self.lib.ta_offline_compute(
            u1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(u1_ids),
            self._to_bytes(kappa_m_str), t, param_size,
            out_s_alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_s_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        )
        return out_s_alpha, out_s_beta

    def aggregate_and_unmask_sparse(self, u1_ids, u2_ids, shares_list, ciphers_list, kappa_m_str, t, model_hash_str, threshold, ta_s_alpha, ta_s_beta):
        u1_arr = np.array(u1_ids, dtype=np.int32)
        u2_arr = np.array(u2_ids, dtype=np.int32)
        
        flat_shares = []
        for client_idx, u2_cid in enumerate(u2_ids):
            for tag, target, share_val in shares_list[client_idx]:
                flat_shares.extend([u2_cid, tag, target, share_val])
        arr_shares = np.array(flat_shares, dtype=np.int64)
        
        flat_ciphers = np.concatenate(ciphers_list).astype(np.int64)
        param_size = len(ciphers_list[0]) if ciphers_list else 0
        
        out_buffer = np.zeros(param_size, dtype=np.float32)
        
        self.lib.aggregate_and_unmask_sparse(
            u1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(u1_ids),
            u2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(u2_ids),
            arr_shares.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(flat_shares),
            flat_ciphers.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), param_size,
            self._to_bytes(kappa_m_str), t, self._to_bytes(model_hash_str), threshold,
            ta_s_alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ta_s_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        return out_buffer.tolist()