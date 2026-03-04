import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
import time
from collections import defaultdict

from Defence.score import ScoreCalculator
from Defence.kickout import KickoutManager
from Defence.layers_proj_detect import Layers_Proj_Detector
from _utils_.tee_adapter import get_tee_adapter_singleton

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Server(object):
    def __init__(self, model_class, test_dataloader, device_str, detection_method="none", defense_config=None, seed=42, verbose=False, log_file_path=None, malicious_clients=None):
        self.device = torch.device(device_str)
        self.global_model = model_class().to(self.device)
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.tee_adapter = get_tee_adapter_singleton()
        # [移除] 不再需要 ServerAdapter (C++ 聚合核心)
        
        self.detection_method = detection_method
        self.verbose = verbose 
        self.log_file_path = log_file_path
        self.seed = seed 
        self.malicious_clients = set(malicious_clients) if malicious_clients else set()
        self.defense_config = defense_config or {}
        self.suspect_counters = {} 
        self.global_update_direction = None 
        self.detection_history = defaultdict(lambda: {'suspect_cnt': 0, 'kicked_cnt': 0, 'events': []})
        
        det_params = self.defense_config.get('params', {})
        self.mesas_detector = Layers_Proj_Detector(config=det_params)
        self.score_calculator = ScoreCalculator() if "score" in detection_method else None
        self.kickout_manager = KickoutManager() if "kickout" in detection_method else None
        
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

    def calculate_weights(self, client_id_list, client_features_dict_list, client_data_sizes, current_round=0):
        client_projections = {cid: feat for cid, feat in zip(client_id_list, client_features_dict_list)}
        self._update_global_direction_feature(current_round)
        weights = {}
        if any(k in self.detection_method for k in ["mesas", "projected", "layers_proj"]):
            if self.verbose: 
                print(f"  [Server] Executing {self.detection_method} detection (Round {current_round})...")
            
            raw_weights, logs, global_stats = self.mesas_detector.detect(
                client_projections, self.global_update_direction, self.suspect_counters, verbose=self.verbose
            )
            
            if self.log_file_path:
                self._write_detection_log(current_round, logs, raw_weights, global_stats)
            
            total_score = sum(raw_weights.values())
            weights = {cid: s / total_score for cid, s in raw_weights.items()} if total_score > 0 else {cid: 0.0 for cid in raw_weights}
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
                            val = info.get(key_base, 0)
                            median = global_stats.get(f"{key_base}_median", 0)
                            thresh = global_stats.get(f"{key_base}_threshold", 0)
                            
                            row.append(f"{val:.4f}")
                            row.append(f"{median:.4f}")
                            row.append(f"{thresh:.4f}")
                    
                    writer.writerow(row)
        except Exception: pass

    def secure_aggregation(self, client_objects, active_ids, round_num):
        """
        [修改] 替换为明文加权聚合，省去复杂的 TEE 掩码和分片过程
        """
        t_start_total = time.time()
        
        if self.verbose:
            print(f"\n[Server] >>> STARTING PLAINTEXT AGGREGATION (ROUND {round_num}) <<<")
        
        weights_map = self.current_round_weights
        sorted_active_ids = sorted(active_ids) 
        
        global_update_flat = None
        valid_clients_count = 0

        for client in client_objects:
            if client.client_id not in sorted_active_ids: continue
            
            w = weights_map.get(client.client_id, 0.0)
            if w <= 1e-9: 
                # 被防御机制剔除或权重极低，不参与聚合
                continue
            
            # 直接获取 Client 的明文梯度
            client_grad = client.get_plaintext_gradient()
            
            if global_update_flat is None:
                global_update_flat = np.zeros_like(client_grad, dtype=np.float64)
                
            # 加权累加
            global_update_flat += w * client_grad
            valid_clients_count += 1

        if global_update_flat is not None and valid_clients_count > 0:
            try:
                # 将聚合后的梯度应用到全局模型
                self._apply_global_update(global_update_flat)
                if self.verbose:
                    print(f"  [Success] Aggregation Completed with {valid_clients_count} valid clients.")
            except Exception as e:
                print(f"  [Critical Error] Aggregation crashed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [Warning] No valid clients to aggregate in Round {round_num}.")

        t_agg_end = time.time()
        
        if self.verbose:
            print(f"  [Perf] Total Round Time (Plaintext): {t_agg_end - t_start_total:.4f}s")

    def _update_global_direction_feature(self, current_round):
        try:
            w_new_flat = self._flatten_params(self.global_model)
            if self.w_old_global_flat is None:
                self.w_old_global_flat = np.zeros_like(w_new_flat)
            proj_seed = int(self.seed + current_round)
            
            # [修改] 使用 Python 层的模拟投影生成全局方向特征
            projection, _ = self.tee_adapter.simulate_projection(
                -1, proj_seed, w_new_flat, self.w_old_global_flat
            )
            self.global_update_direction = {'full': projection, 'layers': {}}
        except Exception as e:
            print(f"  [Warning] Failed to update global direction feature: {e}")
            self.global_update_direction = None

    def _apply_global_update(self, update_flat):
        idx = 0
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                numel = param.numel()
                grad_segment = update_flat[idx : idx+numel]
                # 转换为与参数相同的数据类型并移至对应设备
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
        # 1. 评估前先校准 BN (使用测试集的一小部分)
        if self.test_dataloader:
            self.recalibrate_bn(self.test_dataloader, num_batches=20)
        
        # 2. 标准 eval
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
        
        correct = 0
        total = 0
        
        original_params = copy.deepcopy(poison_loader.attack_params)
        
        poison_loader.attack_params['backdoor_ratio'] = 1.0  
        poison_loader.attack_params['poison_ratio'] = 1.0    
        
        attack_methods = poison_loader.attack_methods
        target_class = None
        filter_fn = None 
        
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