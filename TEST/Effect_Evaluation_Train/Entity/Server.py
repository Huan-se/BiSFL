import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
import time
import gc
from collections import defaultdict

from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector
from Defence.baseline_method import BaselineDetector 
from _utils_.tee_adapter import get_tee_adapter_singleton

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None, poison_ratio=0.0):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.tee_adapter = get_tee_adapter_singleton()
        
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
        
        if self.detection_method in ['krum', 'clustering']:
            self.baseline_detector = BaselineDetector(
                method=self.detection_method, 
                poison_ratio=poison_ratio, 
                device_str=device_str
            )
        else:
            self.baseline_detector = None
            
        self.current_round_weights = {}
        self.w_old_global_flat = self._flatten_params(self.global_model)
        
        if self.log_file_path: self._init_log_file()

    def _init_log_file(self):
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        
        headers = ["Round", "Client_ID", "Type", "Score", "Status"]
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics = ['l2', 'var', 'dist']
        
        for scope in scopes:
            for metric in metrics:
                base = f"{scope}_{metric}"
                headers.extend([base, f"{base}_median", f"{base}_threshold"])
        
        try:
            with open(self.log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception: pass
        
    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def get_global_params_and_proj(self):
        self.w_old_global_flat = self._flatten_params(self.global_model)
        return copy.deepcopy(self.global_model.state_dict()), None

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0, client_objects=None):
        # [修改]：移除了无条件的全局投影计算
        weights = {}
        
        # 1. 我们的方法 (OURS: 特征投影 + 聚类检测)
        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj", "ours"]):
            # [修改]：仅在 OURS 方案下才更新全局梯度的投影特征
            self._update_global_direction_feature(current_round)
            
            if self.verbose: 
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            
            client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose
            )
            
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)
            
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
            
        # 2. 基线方法 (Krum / Clustering)
        elif self.detection_method in ['krum', 'clustering'] and self.baseline_detector:
            if self.verbose:
                print(f"  [Server] Executing Baseline {self.detection_method.upper()} detection (Round {current_round})...")
            
            client_grads = {c.client_id: c.get_plaintext_gradient() for c in client_objects if c.client_id in client_id_list}
            weights, logs, global_stats = self.baseline_detector.detect(client_grads, verbose=self.verbose)
            
            if self.log_file_path:
                self._write_detection_log(current_round, logs, weights, global_stats)
        
        # 3. 未开启检测 (No Defense / Median)
        else:
            weights = {cid: 1.0/len(client_id_list) for cid in client_id_list}
            
        self.current_round_weights = weights
        return weights

    def _write_detection_log(self, round_num, logs, weights, global_stats):
        if not self.log_file_path: return
        
        target_layers = self.defense_config.get('target_layers', [])
        scopes = ['full'] + [f'layer_{name}' for name in target_layers]
        metrics_list = ['l2', 'var', 'dist']

        try:
            with open(self.log_file_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                for cid in sorted(logs.keys()):
                    info = logs[cid]
                    score = weights.get(cid, 0.0)
                    status = info.get('status', 'NORMAL')
                    c_type = "Malicious" if cid in self.malicious_clients else "Benign"
                    
                    row = [round_num, cid, c_type, f"{score:.4f}", status]
                    
                    for scope in scopes:
                        for metric in metrics_list:
                            key_base = f"{scope}_{metric}"
                            row.append(f"{info.get(key_base, 0):.4f}")
                            row.append(f"{global_stats.get(f'{key_base}_median', 0):.4f}")
                            row.append(f"{global_stats.get(f'{key_base}_threshold', 0):.4f}")
                    
                    writer.writerow(row)
        except Exception: pass

    def secure_aggregation(self, client_objects, active_ids, round_num):
        t_start_total = time.time()
        sorted_active_ids = sorted(active_ids) 
        
        if self.detection_method == 'median':
            if self.verbose:
                print(f"\n[Server] >>> STARTING MEDIAN AGGREGATION (ROUND {round_num}) <<<")
            
            grads_list = []
            for client in client_objects:
                if client.client_id in sorted_active_ids:
                    grads_list.append(torch.from_numpy(client.get_plaintext_gradient()).to(self.device))
            
            if len(grads_list) > 0:
                stacked_grads = torch.stack(grads_list) 
                median_grad, _ = torch.median(stacked_grads, dim=0) 
                self._apply_global_update(median_grad.cpu().numpy())
                
                # 【修改】：Median 非常吃显存，算完直接手动超度
                del stacked_grads
                del median_grad
                del grads_list
                torch.cuda.empty_cache()
                gc.collect()
                
                if self.verbose:
                    print(f"  [Success] Median Aggregation Completed.")
            return

        if self.verbose:
            print(f"\n[Server] >>> STARTING WEIGHTED AGGREGATION (ROUND {round_num}) <<<")
            
        weights_map = self.current_round_weights
        global_update_flat = None
        valid_clients_count = 0

        for client in client_objects:
            if client.client_id not in sorted_active_ids: continue
            
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: continue
            
            client_grad = client.get_plaintext_gradient()
            if global_update_flat is None:
                global_update_flat = np.zeros_like(client_grad, dtype=np.float64)
                
            global_update_flat += w * client_grad
            valid_clients_count += 1

        if global_update_flat is not None and valid_clients_count > 0:
            self._apply_global_update(global_update_flat)
            if self.verbose:
                print(f"  [Success] Aggregation Completed with {valid_clients_count} valid clients.")
        else:
            print(f"  [Warning] No valid clients to aggregate in Round {round_num}.")

    def _update_global_direction_feature(self, current_round):
        try:
            w_new_flat = self._flatten_params(self.global_model)
            if self.w_old_global_flat is None:
                self.w_old_global_flat = np.zeros_like(w_new_flat)
            proj_seed = int(self.seed + current_round)
            
            projection, _ = self.tee_adapter.simulate_projection(
                -1, proj_seed, w_new_flat, self.w_old_global_flat
            )
            self.global_update_direction = {'full': projection, 'layers': {}}
        except Exception:
            self.global_update_direction = None

    def _apply_global_update(self, update_flat):
        idx = 0
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                grad_tensor = torch.from_numpy(grad_segment).view(param.shape).to(self.device).type_as(param.data)
                param.data.add_(grad_tensor)
                idx += numel

    def recalibrate_bn(self, loader, num_batches=20):
        self.global_model.train()
        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                if i >= num_batches: break
                data = data.to(self.device)
                self.global_model(data)
        self.global_model.eval()

    def evaluate(self):
        if self.test_dataloader:
            self.recalibrate_bn(self.test_dataloader, num_batches=20)
            
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

    def evaluate_asr(self, loader, poison_loader):
        self.global_model.eval()
        correct, total = 0, 0
        original_params = copy.deepcopy(poison_loader.attack_params)
        
        poison_loader.attack_params['backdoor_ratio'] = 1.0  
        poison_loader.attack_params['poison_ratio'] = 1.0    
        
        attack_methods = poison_loader.attack_methods
        target_class, filter_fn = None, None 
        
        if "backdoor" in attack_methods:
            target_class = original_params.get("backdoor_target", 0)
            filter_fn = lambda t: t != target_class
        elif "label_flip" in attack_methods:
            target_class = original_params.get("target_class", 7)
            source_class = original_params.get("source_class", 1)
            filter_fn = lambda t: t == source_class
        else:
            if "target_class" in original_params:
                target_class = original_params["target_class"]
                filter_fn = lambda t: t != target_class
            else:
                poison_loader.attack_params = original_params
                return 0.0

        if target_class is None: 
            poison_loader.attack_params = original_params
            return 0.0
            
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                indices = torch.where(filter_fn(target))[0]
                if len(indices) == 0: continue
                
                data_subset = data[indices]
                target_subset = target[indices]
                
                data_poisoned, target_poisoned = poison_loader.apply_data_poison(data_subset, target_subset)
                
                data_poisoned = data_poisoned.to(self.device)
                target_poisoned = target_poisoned.to(self.device)
                
                output = self.global_model(data_poisoned)
                _, predicted = output.max(1)
                
                total += len(target_poisoned)
                correct += predicted.eq(target_poisoned).sum().item()
        
        poison_loader.attack_params = original_params
        if total == 0: return 0.0
        return 100. * correct / total