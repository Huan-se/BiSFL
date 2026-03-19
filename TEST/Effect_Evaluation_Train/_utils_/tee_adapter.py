import ctypes
import os
import numpy as np
import threading
import torch
import time

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
        
        # ====================================================
        # [核心优化]：混合存储与流式生成策略
        # ====================================================
        self.FIXED_PROJ_SEED = 8888        # 全局固定种子
        self.global_matrix_cache = {}      # 内存常驻缓存 (每次训练读取一次)
        self.MAX_CACHE_MB = 4000           # 缓存阈值(4GB)，拦截45GB的ResNet18
        
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
        if not self.lib: 
            self.initialized = True 
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
        [Ultra-Fast Hybrid Simulation]
        小模型 (LeNet5/ResNet20): 从外存读取一次至内存常驻，极速矩阵乘法。
        超大模型 (ResNet18): GPU 流式重生成，避免 45GB 爆存。
        """
        if w_new.dtype != np.float32: w_new = w_new.astype(np.float32)
        if w_old.dtype != np.float32: w_old = w_old.astype(np.float32)
        grad = w_new - w_old
        total_len = grad.size
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grad_tensor = torch.from_numpy(grad).to(device)
        projection = torch.zeros(output_dim, device=device)
        
        chunk_size = 200000 
        total_mb = (total_len * output_dim * 4) / (1024 * 1024)

        if total_mb <= self.MAX_CACHE_MB:
            # === 策略 A：外存预生成 + 内存常驻 (LeNet5/ResNet20) ===
            matrix_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proj_matrices")
            os.makedirs(matrix_dir, exist_ok=True)
            matrix_path = os.path.join(matrix_dir, f"proj_{total_len}_{output_dim}_seed{self.FIXED_PROJ_SEED}.pt")

            with self.lock:
                # 1. 如果内存里没有，去硬盘找
                if 'full' not in self.global_matrix_cache:
                    if os.path.exists(matrix_path):
                        try:
                            self.global_matrix_cache['full'] = torch.load(matrix_path, map_location='cpu')
                        except Exception:
                            pass # 如果并发读取损坏，静默处理，走下面重新生成
                    
                    # 2. 如果硬盘也没有(第一次运行)，就在 GPU 生成并保存到硬盘
                    if 'full' not in self.global_matrix_cache:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(self.FIXED_PROJ_SEED)
                        
                        chunks = []
                        for start_idx in range(0, total_len, chunk_size):
                            end_idx = min(start_idx + chunk_size, total_len)
                            current_chunk_len = end_idx - start_idx
                            # 生成完马上推到 CPU 防止显存溢出
                            mat_chunk = torch.randn((output_dim, current_chunk_len), generator=generator, device=device).cpu()
                            chunks.append(mat_chunk)
                        
                        full_mat = torch.cat(chunks, dim=1)
                        self.global_matrix_cache['full'] = full_mat
                        
                        # 原子化写入，防止多个进程并发把文件写坏
                        tmp_path = matrix_path + f".tmp.{os.getpid()}"
                        torch.save(full_mat, tmp_path)
                        try:
                            os.rename(tmp_path, matrix_path)
                        except OSError:
                            if os.path.exists(tmp_path): os.remove(tmp_path)
            
            # 3. 内存直接切片推 GPU 算矩阵乘法
            full_mat_cpu = self.global_matrix_cache['full']
            with torch.no_grad():
                for start_idx in range(0, total_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_len)
                    mat_chunk = full_mat_cpu[:, start_idx:end_idx].to(device)
                    grad_chunk = grad_tensor[start_idx:end_idx]
                    projection += torch.matmul(mat_chunk, grad_chunk)
                    
                    del mat_chunk
                    del grad_chunk
        else:
            # === 策略 B：纯流式实时生成 (ResNet18) ===
            generator = torch.Generator(device=device)
            generator.manual_seed(self.FIXED_PROJ_SEED)
            
            with torch.no_grad():
                for start_idx in range(0, total_len, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_len)
                    current_chunk_len = end_idx - start_idx
                    mat_chunk = torch.randn((output_dim, current_chunk_len), generator=generator, device=device)
                    grad_chunk = grad_tensor[start_idx:end_idx]
                    projection += torch.matmul(mat_chunk, grad_chunk)
                    
                    del mat_chunk
                    del grad_chunk
            
        ranges = np.array([0, total_len], dtype=np.int32)
        return projection.cpu().numpy(), ranges

    # --- Original Wrappers (Keep for compatibility) ---
    # 下方原有的 prepare_gradient 等方法无需改动
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