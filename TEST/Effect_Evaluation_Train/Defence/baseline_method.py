import torch
import numpy as np
from sklearn.cluster import KMeans
import gc

class BaselineDetector:
    def __init__(self, method, poison_ratio, device_str='cuda'):
        self.method = method
        self.poison_ratio = poison_ratio
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        
    def detect(self, client_grads_dict, verbose=False):
        cids = sorted(list(client_grads_dict.keys()))
        num_clients = len(cids)
        
        ratio = self.poison_ratio if self.poison_ratio > 0 else 0.4
        f = int(num_clients * ratio)
        if f >= num_clients // 2:
            f = num_clients // 2 - 1 
        if f < 0: f = 0

        grads_list = []
        for cid in cids:
            grads_list.append(torch.from_numpy(client_grads_dict[cid]).to(self.device))
        
        grads_tensor = torch.stack(grads_list) 

        weights = {cid: 0.0 for cid in cids}
        logs = {cid: {'status': 'NORMAL', 'full_l2': 0.0} for cid in cids}
        global_stats = {}

        if self.method == 'krum':
            weights, logs = self._krum(cids, grads_tensor, f, num_clients)
        elif self.method == 'clustering':
            weights, logs = self._clustering(cids, grads_tensor, num_clients)
            
        # 【修改】：强制清空巨型聚合张量，解救显存
        del grads_tensor
        del grads_list
        torch.cuda.empty_cache()
        gc.collect()
            
        return weights, logs, global_stats

    def _krum(self, cids, grads_tensor, f, num_clients):
        dist_matrix = torch.cdist(grads_tensor, grads_tensor, p=2)
        
        scores = []
        k = num_clients - f
        if k <= 0: k = 1
        
        for i in range(num_clients):
            dists = dist_matrix[i]
            sorted_dists, _ = torch.sort(dists)
            score = torch.sum(sorted_dists[:k]).item()
            scores.append(score)
            
        best_idx = int(np.argmin(scores))
        
        weights = {cid: 0.0 for cid in cids}
        weights[cids[best_idx]] = 1.0 
        
        logs = {}
        for i, cid in enumerate(cids):
            status = 'NORMAL' if i == best_idx else 'KICK_OUT'
            logs[cid] = {'status': status, 'full_l2': scores[i]}
            
        del dist_matrix
        return weights, logs

    def _clustering(self, cids, grads_tensor, num_clients):
        normalized_grads = torch.nn.functional.normalize(grads_tensor, p=2, dim=1)
        grads_cpu = normalized_grads.cpu().numpy()
        
        del normalized_grads
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(grads_cpu)
        
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)
        
        benign_label = 0 if count_0 >= count_1 else 1
        
        weights = {}
        logs = {}
        benign_cids = []
        
        for i, cid in enumerate(cids):
            if labels[i] == benign_label:
                benign_cids.append(cid)
                logs[cid] = {'status': 'NORMAL', 'full_l2': float(labels[i])}
            else:
                weights[cid] = 0.0
                logs[cid] = {'status': 'KICK_OUT', 'full_l2': float(labels[i])}
                
        if len(benign_cids) > 0:
            w = 1.0 / len(benign_cids)
            for cid in benign_cids:
                weights[cid] = w
        
        return weights, logs