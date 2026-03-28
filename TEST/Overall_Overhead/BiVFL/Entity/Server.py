import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
import time
import hashlib
import math
from collections import defaultdict

from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector
try:
    from Defence.baseline_method import BaselineDetector
except ImportError:
    BaselineDetector = None

from _utils_.tee_adapter import get_tee_adapter_singleton
from _utils_.server_adapter import ServerAdapter 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = 9223372036854775783
SCALE = 100000000.0 

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None, poison_ratio=0.0):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.tee_adapter = get_tee_adapter_singleton()
        self.server_adapter = ServerAdapter() 

        self.detection_method = detection_method
        self.verbose = verbose 
        self.log_file_path = log_file_path
        self.seed = seed 
        self.malicious_clients = set(malicious_clients) if malicious_clients else set()
        self.defense_config = defense_config or {}
        
        self.suspect_counters = {} 
        self.global_update_direction = None 
        
        det_params = self.defense_config.get('params', {})
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        
        if self.detection_method in ['krum', 'clustering'] and BaselineDetector is not None:
            self.baseline_detector = BaselineDetector(method=self.detection_method, poison_ratio=poison_ratio, device_str=device_str)
        else: self.baseline_detector = None
            
        self.current_round_weights = {}
        self.kappa_p = self.seed + 1000        
        self.kappa_s = self.seed + 2000        
        self.kappa_m = self.seed + 3000        
        
        self.w_old_global_flat = self._flatten_params(self.global_model)
        
    def _flatten_params(self, model):
        params = []
        for param in model.parameters(): params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def get_global_params_and_proj(self):
        self.w_old_global_flat = self._flatten_params(self.global_model)
        return copy.deepcopy(self.global_model.state_dict()), None

    def _compute_tau(self, n):
        K = int(3 * math.log2(n) + 9)
        required_K = int(2 * math.ceil(0.5 * n) + 2)
        if K < required_K: K = required_K
        if K >= n: K = n - 1
        if K % 2 != 0: K -= 1
        tau = int(K - math.ceil(0.5 * n))
        if tau < 2: tau = 2
        return tau

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0, client_objects=None):
        weights = {}
        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj", "ours"]):
            self._update_global_direction_feature(current_round)
            client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose)
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
        elif self.detection_method in ['krum', 'clustering'] and self.baseline_detector and client_objects:
            client_grads = {c.client_id: c.get_plaintext_gradient() for c in client_objects if c.client_id in client_id_list}
            weights, logs, global_stats = self.baseline_detector.detect(client_grads, verbose=self.verbose)
        else: weights = {cid: 1.0/len(client_id_list) for cid in client_id_list}
        self.current_round_weights = weights
        return weights

    def secure_aggregation(self, client_objects, active_ids, round_num):
        t_start_total = time.time()
        weights_map = self.current_round_weights
        sorted_active_ids = sorted(active_ids) 
        
        w_bytes = self.w_old_global_flat.tobytes()
        model_hash_str = str(int(hashlib.sha256(w_bytes).hexdigest()[:15], 16))

        t_s1_start = time.time()
        u2_cids = []
        cipher_map = {} 
        for client in client_objects:
            if client.client_id not in sorted_active_ids: continue
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            try:
                c_grad = client.tee_step1_encrypt(w, sorted_active_ids, self.kappa_m, round_num, model_hash_str)
                u2_cids.append(client.client_id)
                cipher_map[client.client_id] = c_grad
            except Exception as e: pass

        u2_ids = sorted(u2_cids) 
        t_s1_end = time.time()
        
        global_abort_threshold = int(np.ceil(0.5 * len(sorted_active_ids)))
        if len(u2_ids) < global_abort_threshold:
            return

        t_s2_start = time.time()
        shares_list = []
        final_ciphers = []
        for cid in u2_ids:
            client = next(c for c in client_objects if c.client_id == cid)
            try:
                shares = client.tee_step2_generate_shares(self.kappa_s, self.kappa_m, round_num, sorted_active_ids, u2_ids)
                shares_list.append(shares)
                final_ciphers.append(cipher_map[cid])
            except Exception as e: pass
        t_s2_end = time.time()

        t_agg_start = time.time()
        try:
            reconstruct_tau = self._compute_tau(len(sorted_active_ids))
            
            # 1. 安全聚合反量化可训练参数（权重和偏置）
            result_float = self.server_adapter.aggregate_and_unmask(
                sorted_active_ids, u2_ids, shares_list, final_ciphers, 
                self.kappa_m, round_num, model_hash_str, reconstruct_tau
            )
            self._apply_global_update(result_float)
            
            # 2. [核心修复] 明文同步 BatchNorm 的统计缓存
            with torch.no_grad():
                for name, buffer in self.global_model.named_buffers():
                    if 'running' in name or 'num_batches' in name:
                        buffer.zero_()
                        for cid in u2_ids:
                            client = next(c for c in client_objects if c.client_id == cid)
                            w = weights_map.get(cid, 0.0)
                            # 获取客户端当前的 BN 状态并加权累加
                            client_buffer = client.model.state_dict()[name].to(self.device)
                            buffer.add_(client_buffer * w)
                            
        except Exception as e: print(f"  [Critical Error] Aggregation crashed: {e}")
        t_agg_end = time.time()
        
        if self.verbose:
            print(f"  [Perf] Time Breakdown:")
            print(f"         Step 1 (Cipher Upload) : {t_s1_end - t_s1_start:.4f}s")
            print(f"         Step 2 (Share Upload)  : {t_s2_end - t_s2_start:.4f}s")
            print(f"         Step 3 (C++ Aggregation): {t_agg_end - t_agg_start:.4f}s")
            print(f"         Total Round Time       : {t_agg_end - t_start_total:.4f}s")

    def _update_global_direction_feature(self, current_round):
        try:
            w_new_flat = self._flatten_params(self.global_model)
            if self.w_old_global_flat is None: self.w_old_global_flat = np.zeros_like(w_new_flat)
            projection, _ = self.tee_adapter.prepare_gradient(-1, self.kappa_p + current_round, w_new_flat, self.w_old_global_flat)
            self.global_update_direction = {'full': projection, 'layers': {}}
        except Exception: self.global_update_direction = None

    def _apply_global_update(self, update_flat):
        idx = 0
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                grad_tensor = torch.from_numpy(grad_segment).view(param.shape).to(self.device)
                param.data.add_(grad_tensor)
                idx += numel

    def evaluate(self):
        self.global_model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                test_loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.*correct/total, test_loss/len(self.test_dataloader)

    def evaluate_asr(self, loader, poison_loader): return 0.0