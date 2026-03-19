import os
import sys
import yaml
import argparse
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Entity.Client import Client
from Entity.Server import Server
from _utils_.dataloader import load_and_split_dataset
from _utils_.poison_loader import PoisonLoader
from _utils_.save_config import save_result_with_config

from model.Lenet5 import LeNet5
from model.Resnet20 import resnet20
from model.Resnet18 import ResNet18_CIFAR10

def parse_args():
    parser = argparse.ArgumentParser(description="PPFL-TEE Plaintext Simulation Framework")
    
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to base config file')
    
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], help='Dataset to use (auto-bound to model if omitted)')
    parser.add_argument('--model', type=str, choices=['lenet5', 'resnet20', 'resnet18'], required=True, help='Model architecture')
    
    parser.add_argument('--attack', type=str, choices=['none', 'random_poison', 'gradient_amplify', 'backdoor', 'label_flip'], default='none')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='Ratio of poisoned clients (0.0 to 1.0)')
    
    parser.add_argument('--defense', type=str, choices=['none', 'layers_proj_detect', 'krum', 'median', 'clustering'], default='none')
    
    parser.add_argument('--rounds', type=int, default=None, help='Number of FL rounds')
    parser.add_argument('--local_epochs', type=int, default=None, help='Local training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    for sec in ['experiment', 'federated', 'data', 'defense', 'attack']:
        if sec not in config: config[sec] = {}
        
    if args.model == 'lenet5':
        args.dataset = 'mnist'
    elif args.model in ['resnet20', 'resnet18']:
        args.dataset = 'cifar10'
        
    config['data']['dataset'] = args.dataset
    config['data']['model'] = args.model
    
    if args.attack != 'none':
        config['attack']['active_attacks'] = [args.attack]
        config['attack']['poison_ratio'] = args.poison_ratio
    else:
        config['attack']['active_attacks'] = []
        config['attack']['poison_ratio'] = 0.0

    config['defense']['method'] = args.defense
    
    if args.rounds is not None:
        config['federated']['comm_rounds'] = args.rounds
    if args.local_epochs is not None:
        config['federated']['local_epochs'] = args.local_epochs
    config['experiment']['device'] = args.device
    
    config['poison_ratio'] = config['attack']['poison_ratio']
    config['if_noniid'] = config['data'].get('if_noniid', False)
    
    return config

MODEL_REGISTRY = {
    'lenet5': LeNet5,
    'resnet20': resnet20,
    'resnet18': ResNet18_CIFAR10
}

def main():
    args = parse_args()
    config = load_config(args)
    
    exp_name = f"{args.dataset}_{args.model}_{args.attack}_p{args.poison_ratio}_{args.defense}"
    results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "detection_log.csv")
    
    with open(os.path.join(results_dir, "run_config.yaml"), 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: {exp_name}")
    print(f"{'='*50}")

    device_str = args.device if torch.cuda.is_available() else "cpu"
    num_clients = config['federated'].get('total_clients', 20)
    batch_size = config['federated'].get('batch_size', 64)
    global_rounds = config['federated'].get('comm_rounds', 100)
    seed = config['experiment'].get('seed', 42)
    
    print("\n[1] Loading Data...")
    client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=args.dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        if_noniid=config['data'].get('if_noniid', False),
        alpha=config['data'].get('alpha', 0.5),
        data_dir="./data"
    )

    poison_loader = None
    malicious_cids = []
    active_attacks = config['attack'].get('active_attacks', [])
    poison_ratio = config['attack'].get('poison_ratio', 0.0)
    
    if len(active_attacks) > 0 and poison_ratio > 0:
        num_attackers = int(num_clients * poison_ratio)
        malicious_cids = np.random.choice(num_clients, num_attackers, replace=False).tolist()
        print(f"\n[!] Attack Enabled: {active_attacks} | Attackers: {len(malicious_cids)}/{num_clients}")
        
        try:
            poison_loader = PoisonLoader(
                attack_methods=active_attacks,
                attack_params=config['attack'].get('params', {}),
                dataset_name=args.dataset
            )
        except TypeError:
            poison_loader = PoisonLoader(
                attack_methods=active_attacks,
                attack_params=config['attack'].get('params', {})
            )

    model_class = MODEL_REGISTRY[args.model]

    print("\n[2] Initializing Server...")
    server = Server(
        model_class=model_class,
        test_dataloader=test_loader,
        device_str=device_str,
        detection_method=config['defense'].get('method', 'none'),
        defense_config=config.get('defense', {}),
        seed=seed,
        verbose=config['experiment'].get('verbose', False),
        log_file_path=log_file,
        malicious_clients=malicious_cids,
        poison_ratio=poison_ratio
    )

    print("\n[3] Initializing Clients...")
    clients = []
    for i in range(num_clients):
        client = Client(
            client_id=i,
            train_loader=client_dataloaders[i],
            model_class=model_class,
            poison_loader=poison_loader if i in malicious_cids else None,
            device_str=device_str,
            verbose=(i == 0) 
        )
        client.learning_rate = config['federated'].get('lr', 0.005)
        client.local_epochs = config['federated'].get('local_epochs', 3)
        clients.append(client)

    print("\n[4] Starting Federated Learning Loop...")
    
    history = {'acc': [], 'loss': [], 'asr': []}
    
    # [修改]：判断当前是否是我们的特征投影方法
    is_ours_method = (config['defense']['method'] == 'layers_proj_detect')
    
    for round_num in range(1, global_rounds + 1):
        print(f"\n--- Global Round {round_num}/{global_rounds} ---")
        t_round_start = time.time()
        
        global_params, _ = server.get_global_params_and_proj()
        for client in clients:
            client.receive_model(global_params)
            
        client_features = []
        client_data_sizes = []
        active_ids = []
        
        for client in clients:
            client.phase1_local_train()
            
            # [修改]：只有 OURS 才会触发耗时的 TEE 投影计算，其他模型走轻量级明文提取通道
            if is_ours_method:
                proj_seed = int(seed + round_num) 
                feat, d_size = client.phase2_tee_process(proj_seed)
            else:
                feat, d_size = client.phase2_plaintext_process()
                
            client_features.append(feat)
            client_data_sizes.append(d_size)
            active_ids.append(client.client_id)
            
        server.calculate_weights(
            client_id_list=active_ids,
            client_features_dict_list=client_features,
            client_data_sizes=client_data_sizes,
            current_round=round_num,
            client_objects=clients
        )
        
        server.secure_aggregation(clients, active_ids, round_num)
        
        acc, loss = server.evaluate()
        history['acc'].append(acc)
        history['loss'].append(loss)
        print(f"  [Evaluate] Main Task Acc: {acc:.2f}% | Loss: {loss:.4f}")
        
        if poison_loader and args.attack in ['backdoor', 'label_flip']:
            asr = server.evaluate_asr(test_loader, poison_loader)
            history['asr'].append(asr)
            print(f"  [Evaluate] Attack Success Rate (ASR): {asr:.2f}%")
            
        print(f"  [Time] Round {round_num} completed in {time.time() - t_round_start:.2f}s")

    print("\n[5] Saving results and generating plots...")
    
    if args.attack == 'none':
        mode_name = "pure_training"
    elif args.defense == 'none':
        mode_name = "poison_no_detection"
    else:
        mode_name = "poison_with_detection"
        
    save_result_with_config(
        save_dir=results_dir,
        mode_name=mode_name,
        model_type=args.model,
        dataset_type=args.dataset,
        detection_method=config['defense']['method'],
        config=config,
        accuracy_history=history['acc'],
        asr_history=history['asr'] if len(history['asr']) > 0 else None,
        loss_history=history['loss']
    )
    
    print(f"\n✅ Experiment finished! All data and plots saved to: {results_dir}")

if __name__ == "__main__":
    main()