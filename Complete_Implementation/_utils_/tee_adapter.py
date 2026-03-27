import ctypes
import os
import numpy as np
import threading

_TEE_INSTANCE = None
_INIT_LOCK = threading.Lock()

def get_tee_adapter_singleton():
    global _TEE_INSTANCE
    if _TEE_INSTANCE is None:
        with _INIT_LOCK:
            if _TEE_INSTANCE is None: _TEE_INSTANCE = TEEAdapter()
    return _TEE_INSTANCE

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")
        self.lib_path = lib_path
        self.lib = None
        self.initialized = False
        self.lock = threading.Lock()
        self._load_library()
        self._init_functions()

    def _load_library(self):
        self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)

    def _to_bytes(self, val):
        return str(val).encode('utf-8')

    def _init_functions(self):
        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        
        self.lib.tee_prepare_gradient.argtypes = [
            ctypes.c_int, ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t, np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t, np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

        self.lib.tee_generate_masked_gradient_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

        self.lib.tee_get_vector_shares_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

    def initialize_enclave(self, enclave_path=None):
        if self.initialized: return
        with self.lock:
            if self.initialized: return
            if enclave_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            self.lib.tee_init(enclave_path.encode('utf-8'))
            self.initialized = True

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        w_new = w_new.astype(np.float32)
        w_old = w_old.astype(np.float32)
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        self.lib.tee_prepare_gradient(client_id, self._to_bytes(proj_seed), w_new, w_old, total_len, ranges, len(ranges), output_proj, output_dim)
        return output_proj, ranges

    def generate_masked_gradient_dynamic(self, kappa_m, t, model_hash_str, cid, active_ids, k_weight, output_len):
        if not self.initialized: self.initialize_enclave()
        ranges = np.array([0, output_len], dtype=np.int32)
        arr_active = np.array(active_ids, dtype=np.int32)
        out_buf = np.zeros(output_len, dtype=np.int64)
        self.lib.tee_generate_masked_gradient_dynamic(
            self._to_bytes(kappa_m), t, self._to_bytes(model_hash_str), cid,
            arr_active, len(arr_active), self._to_bytes(k_weight), output_len,
            ranges, len(ranges), out_buf, output_len
        )
        return out_buf

    def get_vector_shares_dynamic(self, kappa_s, kappa_m, t, u1_ids, u2_ids, my_cid):
        if not self.initialized: self.initialize_enclave()
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        # Max shares: max dropouts(len(U1)) * 2 + max survived * 1, times 3 for tags
        max_len = 1 + (len(u1_ids) * 2) * 3
        out_buf = np.zeros(max_len, dtype=np.int64)
        self.lib.tee_get_vector_shares_dynamic(
            self._to_bytes(kappa_s), self._to_bytes(kappa_m), t,
            arr_u1, len(arr_u1), arr_u2, len(arr_u2), my_cid,
            out_buf, max_len
        )
        return out_buf