import sys
import os
import argparse
import yaml
import time
import numpy as np
import torch
import copy
import random

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Entity.Client import Client
from Entity.Server import Server
from _utils_.tee_adapter import get_tee_adapter_singleton
from _utils_.server_adapter import ServerAdapter 
from _utils_.dataloader import load_and_split_dataset
from _utils_.poison_loader import PoisonLoader
from _utils_.save_config import save_result_with_config, check_result_exists, get_result_filename
from model.Lenet5 import LeNet5
from model.Resnet20 import resnet20
from model.Resnet18 import ResNet18_CIFAR10

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_single_mode(full_config, mode_name, current_mode_config):
    exp_conf = full_config['experiment']
    verbose_flag = exp_conf.get('verbose', False)

    def log(*args, **kwargs):
        if verbose_flag:
            print(*args, **kwargs)

    print(f"\n{'='*65}")
    print(f"  Configuration Summary | Mode: {mode_name}")
    print(f"  Verbose Logging: {'ON' if verbose_flag else 'OFF'}")
    print(f"{'='*65}")
    
    fed_conf = full_config['federated']
    data_conf = full_config['data']
    atk_conf = full_config['attack']
    defense_conf = full_config['defense']
    
    # 随机数种子对齐
    seed = exp_conf['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    device_str = exp_conf.get('device', 'auto')
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [System] Using Device: {device_str}")
    
    log_interval = exp_conf.get('log_interval', 100)
    save_dir = os.path.join(current_dir, "results")
    
    detection_method = current_mode_config.get('defense_method', 'none')

    print(f"  [Init] Loading dataset {data_conf['dataset']}...")
    all_client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=data_conf['dataset'],
        num_clients=fed_conf['total_clients'],
        batch_size=fed_conf['batch_size'],
        if_noniid=data_conf['if_noniid'],
        alpha=data_conf['alpha'],
        data_dir="./data"
    )
    
    model_name = data_conf.get('model', '').lower()
    if model_name == 'lenet5': 
        ModelClass = LeNet5
    elif model_name == 'resnet20': 
        ModelClass = resnet20
    elif model_name == 'resnet18': 
        ModelClass = ResNet18_CIFAR10
    else: 
        raise ValueError(f"Unknown model architecture in config: {model_name}.")

    # 初始化攻击配置
    poison_client_ids = []
    current_poison_ratio = current_mode_config.get('poison_ratio', 0.0)
    if current_poison_ratio > 0:
        poison_client_ids = random.sample(range(fed_conf['total_clients']), int(fed_conf['total_clients'] * current_poison_ratio))
        poison_client_ids.sort()
        print(f"  [Attack] Malicious Clients ({len(poison_client_ids)}): {poison_client_ids}")
    else:
        print(f"  [Attack] Malicious Clients: None")

    log_file_path = None
    if any(k in detection_method for k in ["mesas", "projected", "layers_proj", "ours"]):
        log_filename = get_result_filename(mode_name, data_conf['model'], data_conf['dataset'], detection_method, current_mode_config).replace('.npz', '_detection_log.csv')
        log_file_path = os.path.join(save_dir, log_filename)
    
    # 初始化服务端
    server = Server(
        model_class=ModelClass,
        test_dataloader=test_loader,
        device_str=device_str,
        detection_method=detection_method, 
        defense_config=defense_conf,
        seed=seed,
        verbose=verbose_flag,               
        log_file_path=log_file_path,
        malicious_clients=poison_client_ids,
        poison_ratio=current_poison_ratio
    )
    
    # 初始化客户端
    clients = []
    active_attacks = atk_conf.get('active_attacks', [])
    attack_params_dict = atk_conf.get('params', {})
    attack_idx = 0
    primary_attack_type = active_attacks[0] if active_attacks else None
    
    server_poison_loader = None
    if primary_attack_type:
        server_poison_loader = PoisonLoader([primary_attack_type], attack_params_dict.get(primary_attack_type, {}))

    for cid in range(fed_conf['total_clients']):
        client_poison_loader = None
        if cid in poison_client_ids and active_attacks:
            a_type = active_attacks[attack_idx % len(active_attacks)]
            attack_idx += 1
            a_params = attack_params_dict.get(a_type, {})
            client_poison_loader = PoisonLoader([a_type], a_params)
        
        c_verbose = (verbose_flag and cid == 0)
        c = Client(cid, all_client_dataloaders[cid], ModelClass, client_poison_loader, device_str=device_str, verbose=c_verbose, log_interval=log_interval)
        c.learning_rate = fed_conf.get('lr', 0.01)
        c.local_epochs = fed_conf.get('local_epochs', 1)
        clients.append(c)

    print("  [System] Pre-initializing TEE Enclave (Global Singleton)...")
    tee_adapter = get_tee_adapter_singleton()
    tee_adapter.initialize_enclave()
    
    try:
        if hasattr(tee_adapter, 'set_verbose'):
            tee_adapter.set_verbose(verbose_flag)
    except: pass
    
    try:
        temp_server_adapter = ServerAdapter()
        if hasattr(temp_server_adapter, 'set_verbose'):
            temp_server_adapter.set_verbose(verbose_flag)
    except: pass

    total_rounds = fed_conf['comm_rounds']
    acc_history = []
    asr_history = []
    loss_history = []
    
    total_time_train = 0.0
    total_time_proj = 0.0
    total_time_detect = 0.0
    total_time_secagg = 0.0
    
    start_time = time.time()
    
    for r in range(1, total_rounds + 1):
        log(f"\n>>> Round {r}/{total_rounds} Start...")
        round_start_time = time.time()
        
        active_ids = random.sample(range(fed_conf['total_clients']), fed_conf['active_clients'])
        active_ids.sort()
        
        global_params, _ = server.get_global_params_and_proj()
        current_proj_seed = int(seed + r)
        
        for c in clients: c.receive_model(global_params)

        # Phase 1: Local Training
        log(f"  [Phase 1] Starting Local Training (Device: {device_str})...")
        t_p1_start = time.time()
        for cid in active_ids:
            clients[cid].phase1_local_train()
        t_p1_cost = time.time() - t_p1_start
        total_time_train += t_p1_cost
        log(f"  >> Training Phase Finished in {t_p1_cost:.2f}s")

        # Phase 2: TEE Projection Extraction
        log(f"  [Phase 2] Starting TEE Projection...")
        t_p2_start = time.time()
        client_features_list = []
        client_data_sizes = []
        
        for cid in active_ids:
            feat, dsize = clients[cid].phase2_tee_process(current_proj_seed)
            client_features_list.append(feat)
            client_data_sizes.append(dsize)
            
        t_p2_cost = time.time() - t_p2_start
        total_time_proj += t_p2_cost
        log(f"  >> TEE Projection Finished in {t_p2_cost:.2f}s")

        # Phase 3: Server Detection & Weight Calculation
        t_p3_start = time.time()
        weights_map = server.calculate_weights(active_ids, client_features_list, client_data_sizes, current_round=r, client_objects=clients)
        t_p3_cost = time.time() - t_p3_start
        total_time_detect += t_p3_cost

        accepted_ids = [cid for cid, w in weights_map.items() if w > 1e-6]
        accepted_ids.sort()
        
        if not accepted_ids: 
            print("  [Warning] No accepted clients. Skipping aggregation.")
            continue

        # Phase 4 & 5: Secure Masking & Aggregation
        t_p4_start = time.time()
        server.secure_aggregation(clients, accepted_ids, round_num=r)
        t_p4_cost = time.time() - t_p4_start
        total_time_secagg += t_p4_cost

        # Phase 6: Evaluation
        acc, loss = server.evaluate()
        acc_history.append(acc)
        loss_history.append(loss)
        
        print(f"  [Round {r}] Global Acc: {acc:.2f}%, Loss: {loss:.4f}")
        
        if server_poison_loader:
            asr = server.evaluate_asr(test_loader, server_poison_loader)
            asr_history.append(asr)
            print(f"  [Round {r}] ASR: {asr:.2f}%")
        else:
            asr_history.append(0.0)
            
        print(f"  [Time Stats] Train: {t_p1_cost:.2f}s | Proj: {t_p2_cost:.2f}s | Detect: {t_p3_cost:.2f}s | SecAgg: {t_p4_cost:.2f}s")

    total_pipeline_time = total_time_train + total_time_proj + total_time_detect + total_time_secagg
    print("\n" + "="*50)
    print(" 🕒 FINAL TIME PROFILING REPORT")
    print("="*50)
    print(f" Total Active Time : {total_pipeline_time:.2f} s")
    print(f" 1. Local Training : {total_time_train:.2f} s ({(total_time_train/total_pipeline_time)*100:.2f}%)")
    print(f" 2. TEE Projection : {total_time_proj:.2f} s ({(total_time_proj/total_pipeline_time)*100:.2f}%)")
    print(f" 3. Server Detect  : {total_time_detect:.2f} s ({(total_time_detect/total_pipeline_time)*100:.2f}%)")
    print(f" 4. Secure Agg     : {total_time_secagg:.2f} s ({(total_time_secagg/total_pipeline_time)*100:.2f}%)")
    print("-" * 50)
    projection_defense_total = total_time_proj + total_time_detect
    print(f" ✨ Projection Detection Overhead: {projection_defense_total:.2f} s ({(projection_defense_total/total_pipeline_time)*100:.2f}%)")
    print("="*50 + "\n")

    print("[Saving] Saving final results...")
    save_result_with_config(
        save_dir,
        mode_name,
        data_conf['model'],
        data_conf['dataset'],
        detection_method,
        current_mode_config,
        acc_history,
        asr_history,
        loss_history=loss_history
    )
    print(f"[Done] Mode {mode_name} Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    parser.add_argument('--mode', type=str, default=None, help='指定只运行某个模式(逗号分隔)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # [核心修复区]：强行绑定数据集和模型，避免通道数/输入尺寸等维度冲突
    model_name = config['data'].get('model', '').lower()
    if model_name == 'lenet5':
        config['data']['dataset'] = 'mnist'
    elif model_name in ['resnet20', 'resnet18']:
        config['data']['dataset'] = 'cifar10'
    else:
        print(f"[Warning] Unknown model '{model_name}'. Hoping the dataset matches.")

    base_flat_config = {
        'total_clients': config['federated']['total_clients'],
        'batch_size': config['federated']['batch_size'],
        'comm_rounds': config['federated']['comm_rounds'],
        'if_noniid': config['data']['if_noniid'],
        'alpha': config['data']['alpha'],
        'attack_types': config['attack']['active_attacks'],
        'seed': config['experiment']['seed'],
        'model_type': config['data']['model'],
        'dataset_type': config['data']['dataset']
    }
    
    default_poison_ratio = config['attack']['poison_ratio']
    default_defense = config['defense']['method']

    all_modes = [
        {
            'name': 'pure_training', 
            'mode_config': {**base_flat_config, 'poison_ratio': 0.0, 'defense_method': 'none'}
        },
        {
            'name': 'poison_no_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': 'none'}
        },
        {
            'name': 'poison_with_detection', 
            'mode_config': {**base_flat_config, 'poison_ratio': default_poison_ratio, 'defense_method': default_defense}
        }
    ]
    
    target_modes_str = args.mode
    if target_modes_str is None: target_modes_str = config['experiment'].get('modes', 'all')
    
    if target_modes_str == 'all':
        modes_to_run = all_modes
    else:
        target_names = [m.strip() for m in target_modes_str.split(',')]
        modes_to_run = [m for m in all_modes if m['name'] in target_names]

    print(f"计划运行模式: {[m['name'] for m in modes_to_run]}")

    for mode in modes_to_run:
        run_single_mode(config, mode['name'], mode['mode_config'])

if __name__ == '__main__':
    main()