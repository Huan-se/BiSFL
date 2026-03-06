import torch
import numpy as np
from sklearn.cluster import KMeans

class BaselineDetector:
    def __init__(self, method, poison_ratio, device_str='cuda'):
        self.method = method
        self.poison_ratio = poison_ratio
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        
    def detect(self, client_grads_dict, verbose=False):
        """
        根据明文梯度计算权重分配
        client_grads_dict: {cid: flat_gradient_numpy_array}
        Returns: weights_dict, logs, global_stats
        """
        cids = sorted(list(client_grads_dict.keys()))
        num_clients = len(cids)
        
        # 推算 Krum 需要的假设恶意节点数 f
        ratio = self.poison_ratio if self.poison_ratio > 0 else 0.4
        f = int(num_clients * ratio)
        if f >= num_clients // 2:
            f = num_clients // 2 - 1 # Krum 要求 f < N/2，保证至少一半是好人
        if f < 0: f = 0

        # 将 numpy 梯度转换为 GPU Tensor 以加速距离计算
        grads_list = []
        for cid in cids:
            grads_list.append(torch.from_numpy(client_grads_dict[cid]).to(self.device))
        
        # 形状: (N, D) -> N个客户端，D维参数
        grads_tensor = torch.stack(grads_list) 

        weights = {cid: 0.0 for cid in cids}
        logs = {cid: {'status': 'NORMAL', 'full_l2': 0.0} for cid in cids}
        global_stats = {}

        if self.method == 'krum':
            weights, logs = self._krum(cids, grads_tensor, f, num_clients)
        elif self.method == 'clustering':
            weights, logs = self._clustering(cids, grads_tensor, num_clients)
            
        return weights, logs, global_stats

    def _krum(self, cids, grads_tensor, f, num_clients):
        # 1. 计算两两欧氏距离 (N, N)
        dist_matrix = torch.cdist(grads_tensor, grads_tensor, p=2)
        
        scores = []
        # Krum：寻找距离最近的 N - f - 2 个邻居
        # 在距离矩阵中，包含自己（距离为0），所以我们取前 N - f 个最小距离来累加
        k = num_clients - f
        if k <= 0: k = 1
        
        for i in range(num_clients):
            dists = dist_matrix[i]
            sorted_dists, _ = torch.sort(dists)
            # 累加前 k 个距离作为该节点的 Krum Score
            score = torch.sum(sorted_dists[:k]).item()
            scores.append(score)
            
        # 2. 选出得分最小（最中心）的一个节点
        best_idx = int(np.argmin(scores))
        
        weights = {cid: 0.0 for cid in cids}
        weights[cids[best_idx]] = 1.0 # Krum 赋予其权重 1.0，完全采用它的梯度
        
        logs = {}
        for i, cid in enumerate(cids):
            status = 'NORMAL' if i == best_idx else 'KICK_OUT'
            logs[cid] = {'status': status, 'full_l2': scores[i]}
            
        return weights, logs

    def _clustering(self, cids, grads_tensor, num_clients):
        # 1. 将明文梯度进行 L2 归一化。
        # 这一步极其重要，使得普通的欧氏距离 KMeans 在数学上等效于基于余弦相似度的 KMeans
        normalized_grads = torch.nn.functional.normalize(grads_tensor, p=2, dim=1)
        grads_cpu = normalized_grads.cpu().numpy()
        
        # 2. 在归一化后的数据上进行聚类 (强制聚为 2 类)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(grads_cpu)
        
        # 3. 假设数量较多的簇是良性节点，数量较少的簇是恶意节点
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
                
        # 4. 将良性节点的权重均分
        if len(benign_cids) > 0:
            w = 1.0 / len(benign_cids)
            for cid in benign_cids:
                weights[cid] = w
        
        return weights, logs