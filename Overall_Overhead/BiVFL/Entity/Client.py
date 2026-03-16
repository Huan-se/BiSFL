import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from _utils_.tee_adapter import get_tee_adapter_singleton

class Client(object):
    def __init__(self, client_id, train_loader, model_class, poison_loader, device_str='cuda', verbose=False, log_interval=50):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model_class = model_class 
        self.poison_loader = poison_loader 
        self.verbose = verbose
        self.log_interval = log_interval 
        
        # 保存目标计算设备 (GPU)
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 【重要显存优化】：初始化时，模型强行放在 CPU 上休眠，不占显存
        self.model = model_class().cpu()
        
        # 默认参数 (会被 main.py 覆盖)
        self.learning_rate = 0.1 
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.local_epochs = 1
        
        self.tee_adapter = get_tee_adapter_singleton()
        self.ranges = None 
        self.w_old_cache = None

    def _get_poisoned_dataloader(self):
        return self.train_loader

    def _flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def receive_model(self, global_params):
        """接收 Server 下发的最新模型，并更新旧模型缓存"""
        self.model.load_state_dict(global_params)
        self.w_old_cache = self._flatten_params(self.model)

    def phase1_local_train(self, epochs=None):
        """[Phase 1] 本地真实训练 (带显存优化与毒化支持)"""
        t_start = time.time()
        
        # 【重要显存优化】：训练开始，把模型拉到 GPU
        self.model.to(self.device)
        self.model.train()
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        run_epochs = epochs if epochs is not None else self.local_epochs
        MAX_GRAD_NORM = 10.0

        if self.poison_loader and self.poison_loader.attack_methods:
            self.poison_loader.attack_params['local_epochs'] = run_epochs
            
            # PoisonLoader 内部执行攻击
            new_state_dict, _ = self.poison_loader.execute_attack(
                model=self.model,
                dataloader=self.train_loader,
                model_class=self.model_class,
                device=self.device,
                optimizer=optimizer,
                verbose=self.verbose,
                uid=self.client_id
            )
            self.model.load_state_dict(new_state_dict)
            
        else:
            criterion = nn.CrossEntropyLoss()
            for epoch in range(run_epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                    optimizer.step()

                    if self.verbose and batch_idx % self.log_interval == 0:
                        print(f"    [Client {self.client_id}] Epoch {epoch+1}/{run_epochs} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        # 【重要显存优化】：训练结束，强制把模型踢回 CPU，并清理计算图遗留的显存缓存
        self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        t_end = time.time()
        return t_end - t_start

    def phase2_tee_process(self, proj_seed):
        """[Phase 2] TEE 内部提取投影缓存梯度"""
        t_start = time.time()
        w_new_flat = self._flatten_params(self.model)
        
        if self.w_old_cache is None: 
            self.w_old_cache = np.zeros_like(w_new_flat)
        
        if np.isnan(w_new_flat).any(): 
            print(f"  [Warning] Client {self.client_id} has NaN weights!")
            w_new_flat = np.zeros_like(w_new_flat)

        # 调用 TEE C++ 接口生成投影，C++ 端会自动在 Enclave 内部缓存明文梯度
        output, ranges = self.tee_adapter.prepare_gradient(
            self.client_id, proj_seed, w_new_flat, self.w_old_cache
        )
        self.ranges = ranges
        
        t_end = time.time()
        data_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        return {'full': output}, data_size

    def tee_step1_encrypt(self, w, active_ids, seed_mask_root, seed_global_0):
        """[Phase 3A] Server通过后，TEE内部生成加密且加掩码的梯度 (Cipher)"""
        w_new_flat = self._flatten_params(self.model)
        model_len = len(w_new_flat)
        if self.ranges is None: 
            self.ranges = np.array([0, model_len], dtype=np.int32)
        
        # 调用 TEE 生成密文。注意 w 是 Server 分配的聚合权重（需传入 Enclave 与梯度相乘）
        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
            seed_mask_root, seed_global_0, self.client_id, active_ids, 
            w, model_len
        )
        return c_grad

    def tee_step2_generate_shares(self, seed_sss, seed_mask_root, u1_ids, u2_ids):
        """[Phase 3B] 根据实际存活列表生成密码学分片 (Shares)"""
        threshold = len(u1_ids) // 2 + 1
        
        # u1_ids: 本轮计划参与者 (被 Server 接受的 Clients)
        # u2_ids: 实际在线者 (用于模拟掉线的 Online Users，通常实验中可设为等于 u1_ids)
        shares = self.tee_adapter.get_vector_shares_dynamic(
            seed_sss, seed_mask_root, 
            u1_ids, u2_ids, 
            self.client_id, threshold
        )
        return shares