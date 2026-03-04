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
            if _TEE_INSTANCE is None:
                _TEE_INSTANCE = TEEAdapter()
    return _TEE_INSTANCE

class TEEAdapter:
    def __init__(self, lib_path=None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lib_path = os.path.join(base_dir, "lib", "libtee_bridge.so")
        self.lib_path = lib_path
        self.lib = None
        self.enclave_path = None
        self.initialized = False
        self.lock = threading.Lock()
        
        # 尝试加载库，如果失败仅打印警告，不阻碍明文实验
        try:
            self._load_library()
            self._init_functions()
            try:
                self.lib.tee_set_verbose.argtypes = [ctypes.c_int]
                self.lib.tee_set_verbose.restype = None
            except Exception:
                pass
        except Exception as e:
            print(f"[TEEAdapter] Warning: Could not load TEE library ({e}). Running in Simulation Mode only.")

    def set_verbose(self, verbose_bool):
        if self.lib and self.initialized:
            try:
                level = 1 if verbose_bool else 0
                self.lib.tee_set_verbose(level)
            except: pass

    def _load_library(self):
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Lib not found: {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)

    def _to_bytes(self, val):
        return str(val).encode('utf-8')

    def _init_functions(self):
        # 仅当库加载成功时初始化
        if not self.lib: return

        self.lib.tee_init.argtypes = [ctypes.c_char_p]
        self.lib.tee_init.restype = ctypes.c_int
        self.lib.tee_destroy.argtypes = []
        
        self.lib.tee_prepare_gradient.argtypes = [
            ctypes.c_int, ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            ctypes.c_size_t
        ]

        self.lib.tee_generate_masked_gradient_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

        self.lib.tee_get_vector_shares_dynamic.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS'), ctypes.c_size_t
        ]

    def initialize_enclave(self, enclave_path=None):
        if self.initialized: return
        # 如果库未加载，跳过 Enclave 初始化
        if not self.lib: 
            self.initialized = True # 标记为 True 以避免重复尝试
            return

        with self.lock:
            if self.initialized: return
            if enclave_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enclave_path = os.path.join(base_dir, "lib", "enclave.signed.so")
            
            if os.path.exists(enclave_path):
                try:
                    self.lib.tee_init(enclave_path.encode('utf-8'))
                    print(f"[TEEAdapter] Init OK: {enclave_path}")
                except Exception as e:
                    print(f"[TEEAdapter] Enclave Init Failed: {e}")
            else:
                 print(f"[TEEAdapter] Enclave file not found, skipping init.")
            
            self.initialized = True

    # --- Simulation Wrapper (Python Implementation) ---
    
    def simulate_projection(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        """
        [Simulation] 在 Python 中直接计算投影，模拟 TEE 行为。
        计算公式：Proj = Matrix_Gaussian * (w_new - w_old)
        """
        # 1. 计算梯度
        if w_new.dtype != np.float32: w_new = w_new.astype(np.float32)
        if w_old.dtype != np.float32: w_old = w_old.astype(np.float32)
        grad = w_new - w_old
        total_len = grad.size
        
        # 2. 设置随机数生成器 (保证确定性)
        # 注意：这里的随机数序列与 C++ 不完全一致，但对于验证算法有效性是足够的（都是高斯分布）
        rng = np.random.RandomState(proj_seed)
        
        projection = np.zeros(output_dim, dtype=np.float32)
        
        # 3. 分块矩阵乘法 (防止大模型 OOM)
        # 每次处理一部分模型参数 (chunk_size 列)
        chunk_size = 50000 
        
        for start_idx in range(0, total_len, chunk_size):
            end_idx = min(start_idx + chunk_size, total_len)
            current_chunk_len = end_idx - start_idx
            
            # 生成对应的随机矩阵块 (Rows x Current_Cols)
            # 形状: (output_dim, current_chunk_len)
            mat_chunk = rng.randn(output_dim, current_chunk_len).astype(np.float32)
            
            # 获取梯度块
            grad_chunk = grad[start_idx:end_idx]
            
            # 累加投影结果
            projection += mat_chunk @ grad_chunk
            
        # 模拟 ranges 返回值 (全范围)
        ranges = np.array([0, total_len], dtype=np.int32)
        
        return projection, ranges

    # --- Original Wrappers (Keep for compatibility) ---

    def prepare_gradient(self, client_id, proj_seed, w_new, w_old, output_dim=1024):
        if not self.lib: 
            return self.simulate_projection(client_id, proj_seed, w_new, w_old, output_dim)
            
        if not self.initialized: self.initialize_enclave()
        total_len = w_new.size
        if w_new.dtype != np.float32: w_new = w_new.astype(np.float32)
        if w_old.dtype != np.float32: w_old = w_old.astype(np.float32)
        
        ranges = np.array([0, total_len], dtype=np.int32)
        output_proj = np.zeros(output_dim, dtype=np.float32)
        
        self.lib.tee_prepare_gradient(
            client_id, 
            self._to_bytes(proj_seed), 
            w_new, w_old, total_len, 
            ranges, len(ranges), 
            output_proj, output_dim
        )
        return output_proj, ranges

    def generate_masked_gradient_dynamic(self, seed_mask, seed_g0, cid, active_ids, k_weight, output_len):
        if not self.lib: return np.zeros(output_len, dtype=np.int64)
        if not self.initialized: self.initialize_enclave()
        
        model_len = output_len
        ranges = np.array([0, model_len], dtype=np.int32) 
        arr_active = np.array(active_ids, dtype=np.int32)
        out_buf = np.zeros(model_len, dtype=np.int64)
        
        self.lib.tee_generate_masked_gradient_dynamic(
            self._to_bytes(seed_mask), 
            self._to_bytes(seed_g0), 
            cid, 
            arr_active, len(arr_active), 
            self._to_bytes(k_weight), 
            model_len, 
            ranges, len(ranges), 
            out_buf, model_len
        )
        return out_buf

    def get_vector_shares_dynamic(self, seed_sss, seed_mask, u1_ids, u2_ids, my_cid, threshold):
        if not self.lib: return np.zeros(2 + len(u2_ids), dtype=np.int64)
        if not self.initialized: self.initialize_enclave()
        arr_u1 = np.array(u1_ids, dtype=np.int32)
        arr_u2 = np.array(u2_ids, dtype=np.int32)
        
        max_len = 2 + len(u2_ids)
        out_buf = np.zeros(max_len, dtype=np.int64)
        
        self.lib.tee_get_vector_shares_dynamic(
            self._to_bytes(seed_sss), 
            self._to_bytes(seed_mask), 
            arr_u1, len(arr_u1), 
            arr_u2, len(arr_u2), 
            my_cid, threshold, 
            out_buf, max_len
        )
        return out_buf