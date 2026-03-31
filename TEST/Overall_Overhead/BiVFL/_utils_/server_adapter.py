import ctypes
import os
import numpy as np

class ServerAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libserver_core.so")
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(self.lib_path)
        self._init_functions()

    def _init_functions(self):
        # 声明 ta_offline_compute
        self.lib.ta_offline_compute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)
        ]
        
        self.lib.server_core_aggregate_and_unmask.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,  
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,  
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int,  
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int,  
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), # 引入 TA 数组      
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        ]

    def _to_bytes(self, val): return str(val).encode('utf-8')

    # 包装 TA 离线计算
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

    def aggregate_and_unmask(self, u1_ids, u2_ids, shares_list, ciphers_list, kappa_m, t, model_hash_str, threshold, ta_s_alpha, ta_s_beta):
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        flat_shares = []
        for client_idx, u2_cid in enumerate(u2_ids):
            for share_tuple in shares_list[client_idx]:
                flat_shares.extend([u2_cid, share_tuple[0], share_tuple[1], share_tuple[2]])
        arr_shares = np.array(flat_shares, dtype=np.int64)
        data_len = len(ciphers_list[0])
        flat_ciphers = np.concatenate(ciphers_list).astype(np.int64)
        
        output_result = np.zeros(data_len, dtype=np.float32)

        self.lib.server_core_aggregate_and_unmask(
            arr_u1, len(arr_u1), arr_u2, len(arr_u2),
            arr_shares, len(arr_shares), flat_ciphers, data_len,
            self._to_bytes(kappa_m), t, self._to_bytes(model_hash_str), threshold, 
            ta_s_alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ta_s_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            output_result
        )
        return output_result