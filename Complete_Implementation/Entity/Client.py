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
        if device_str == 'cuda' and torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")
            
        self.model = model_class().cpu()
        self.learning_rate = 0.1 
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.local_epochs = 1
        
        self.tee_adapter = get_tee_adapter_singleton()
        self.ranges = None 
        self.w_old_cache = None

    def _flatten_params(self, model):
        params = []
        for param in model.parameters(): params.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(params).astype(np.float32)

    def receive_model(self, global_params):
        self.model.load_state_dict(global_params)
        self.w_old_cache = self._flatten_params(self.model)

    def phase1_local_train(self, epochs=None):
        t_start = time.time()
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        run_epochs = epochs if epochs is not None else self.local_epochs

        if self.poison_loader and self.poison_loader.attack_methods:
            self.poison_loader.attack_params['local_epochs'] = run_epochs
            new_state_dict, _ = self.poison_loader.execute_attack(
                model=self.model, dataloader=self.train_loader, model_class=self.model_class,
                device=self.device, optimizer=optimizer, verbose=self.verbose, uid=self.client_id)
            self.model.load_state_dict(new_state_dict)
        else:
            criterion = nn.CrossEntropyLoss()
            for epoch in range(run_epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(self.model(data), target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    optimizer.step()
        
        self.model.to('cpu')
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return time.time() - t_start

    def phase2_tee_process(self, proj_seed):
        w_new_flat = self._flatten_params(self.model)
        if self.w_old_cache is None: self.w_old_cache = np.zeros_like(w_new_flat)
        output, ranges = self.tee_adapter.prepare_gradient(self.client_id, proj_seed, w_new_flat, self.w_old_cache)
        self.ranges = ranges
        data_size = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 1000
        return {'full': output}, data_size

    def tee_step1_encrypt(self, w, active_ids, kappa_m, t, model_hash_str):
        w_new_flat = self._flatten_params(self.model)
        model_len = len(w_new_flat)
        if self.ranges is None: self.ranges = np.array([0, model_len], dtype=np.int32)
        c_grad = self.tee_adapter.generate_masked_gradient_dynamic(
            kappa_m, t, model_hash_str, self.client_id, active_ids, w, model_len
        )
        return c_grad

    def tee_step2_generate_shares(self, kappa_s, kappa_m, t, u1_ids, u2_ids):
        # shares 的格式为平铺数组: [Count, Tag1, Tgt1, Val1, Tag2, Tgt2, Val2...]
        flat_shares = self.tee_adapter.get_vector_shares_dynamic(
            kappa_s, kappa_m, t, u1_ids, u2_ids, self.client_id
        )
        count = int(flat_shares[0])
        structured_shares = []
        idx = 1
        for _ in range(count):
            if idx + 2 >= len(flat_shares): break
            tag = int(flat_shares[idx])
            tgt = int(flat_shares[idx+1])
            val = int(flat_shares[idx+2])
            structured_shares.append((tag, tgt, val))
            idx += 3
        return structured_shares