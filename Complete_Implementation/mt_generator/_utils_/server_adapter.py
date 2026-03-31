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
        self.lib.server_core_aggregate_and_unmask.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,  # u1_ids
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_int,  # u2_ids
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int,  # shares
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_int,  # ciphers
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,                # kappa_m, t, hash, threshold
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')                 # output
        ]

    def _to_bytes(self, val): return str(val).encode('utf-8')

    def aggregate_and_unmask(self, u1_ids, u2_ids, shares_list, ciphers_list, kappa_m, t, model_hash_str, threshold):
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        flat_shares = []
        for client_idx, u2_cid in enumerate(u2_ids):
            for share_tuple in shares_list[client_idx]:
                flat_shares.extend([u2_cid, share_tuple[0], share_tuple[1], share_tuple[2]])
        arr_shares = np.array(flat_shares, dtype=np.int64)
        
        data_len = len(ciphers_list[0])
        flat_ciphers = np.concatenate(ciphers_list).astype(np.int64)
        output_result = np.zeros(data_len, dtype=np.int64)

        self.lib.server_core_aggregate_and_unmask(
            arr_u1, len(arr_u1), arr_u2, len(arr_u2),
            arr_shares, len(arr_shares), flat_ciphers, data_len,
            self._to_bytes(kappa_m), t, self._to_bytes(model_hash_str), threshold,
            output_result
        )
        return output_result