import os
import sys
import yaml
import argparse
import time
import torch
import numpy as np

# 确保能找到根目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Entity.Client import Client
from Entity.Server import Server
from _utils_.dataloader import load_and_split_dataset
from _utils_.poison_loader import PoisonLoader
from _utils_.save_config import save_result_with_config

# 导入我们刚刚编写的模型库
from model.Lenet5 import LeNet5
from model.Resnet20 import resnet20
from model.Resnet18 import ResNet18_CIFAR10

# ==========================================
# 1. 实验参数与配置路由 (Argparse + YAML)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="PPFL-TEE Plaintext Simulation Framework")
    
    # 基础配置文件
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to base config file')
    
    # 实验核心变量 (覆盖 YAML)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], required=True, help='Dataset to use')
    parser.add_argument('--model', type=str, choices=['lenet5', 'resnet20', 'resnet18'], required=True, help='Model architecture')
    
    # 攻击相关
    parser.add_argument('--attack', type=str, choices=['NoAttack', 'random', 'scaling', 'backdoor', 'label_flip'], default='NoAttack')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='Ratio of poisoned clients (0.0 to 1.0)')
    parser.add_argument('--backdoor_target', type=int, default=0, help='Target class for backdoor attack')
    
    # 防御相关
    parser.add_argument('--defense', type=str, choices=['none', 'ours', 'krum', 'median', 'clustering'], default='none')
    
    # 其他超参 (可选覆盖)
    parser.add_argument('--rounds', type=int, default=None, help='Number of FL rounds')
    parser.add_argument('--local_epochs', type=int, default=None, help='Local training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    return parser.parse_args()

def load_config(args):
    """加载 YAML 并用命令行参数覆盖核心变量"""
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 用 Argparse 覆盖 YAML
    config['data']['dataset'] = args.dataset
    config['data']['model'] = args.model
    config['attack']['attack_methods'] = [args.attack] if args.attack != 'NoAttack' else []
    config['attack']['poison_ratio'] = args.poison_ratio
    config['attack']['backdoor_target'] = args.backdoor_target
    config['defense']['detection_method'] = 'layers_proj' if args.defense == 'ours' else args.defense
    
    if args.rounds is not None:
        config['training']['global_rounds'] = args.rounds
    if args.local_epochs is not None:
        config['training']['local_epochs'] = args.local_epochs
    
    return config

# 模型分发字典
MODEL_REGISTRY = {
    'lenet5': LeNet5,
    'resnet20': resnet20,
    'resnet18': ResNet18_CIFAR10
}

# ==========================================
# 2. 主流程
# ==========================================
def main():
    args = parse_args()
    config = load_config(args)
    
    # 动态生成日志和结果保存路径
    exp_name = f"{args.dataset}_{args.model}_{args.attack}_p{args.poison_ratio}_{args.defense}"
    results_dir = os.path.join("main", "results", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "detection_log.csv")
    
    # 保存本次实验的确切配置，方便复现
    save_result_with_config(config, os.path.join(results_dir, "run_config.yaml"))

    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: {exp_name}")
    print(f"{'='*50}")

    device_str = args.device if torch.cuda.is_available() else "cpu"
    num_clients = config['clients']['num_clients']
    batch_size = config['training']['batch_size']
    global_rounds = config['training']['global_rounds']
    
    # 1. 准备数据
    print("\n[1] Loading Data...")
    client_dataloaders, test_loader = load_and_split_dataset(
        dataset_name=args.dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        if_noniid=config['clients'].get('if_noniid', False),
        alpha=config['clients'].get('alpha', 0.1),
        data_dir="./data"
    )

    # 2. 初始化攻击加载器 (如果有)
    poison_loader = None
    malicious_cids = []
    if args.attack != 'NoAttack' and args.poison_ratio > 0:
        num_attackers = int(num_clients * args.poison_ratio)
        malicious_cids = np.random.choice(num_clients, num_attackers, replace=False).tolist()
        print(f"\n[!] Attack Enabled: {args.attack} | Attackers: {len(malicious_cids)}/{num_clients}")
        
        poison_loader = PoisonLoader(
            attack_methods=[args.attack],
            attack_params=config['attack'],
            malicious_clients=malicious_cids,
            dataset_name=args.dataset
        )

    # 3. 实例化模型类引用
    model_class = MODEL_REGISTRY[args.model]

    # 4. 初始化 Server
    print("\n[2] Initializing Server...")
    server = Server(
        model_class=model_class,
        test_dataloader=test_loader,
        device_str=device_str,
        detection_method=config['defense']['detection_method'],
        defense_config=config.get('defense', {}),
        seed=config['env'].get('seed', 42),
        verbose=True,
        log_file_path=log_file,
        malicious_clients=malicious_cids
    )

    # 5. 初始化 Clients
    print("\n[3] Initializing Clients...")
    clients = []
    for i in range(num_clients):
        client = Client(
            client_id=i,
            train_loader=client_dataloaders[i],
            model_class=model_class,
            poison_loader=poison_loader,
            device_str=device_str,
            verbose=(i == 0) # 仅打印客户端0的训练日志以防刷屏
        )
        # 注入本地训练超参
        client.learning_rate = config['training'].get('learning_rate', 0.1)
        client.local_epochs = config['training'].get('local_epochs', 1)
        clients.append(client)

    # 6. 开始联邦训练循环
    print("\n[4] Starting Federated Learning Loop...")
    
    # 记录评估指标用于画图
    history = {'acc': [], 'loss': [], 'asr': []}
    
    for round_num in range(1, global_rounds + 1):
        print(f"\n--- Global Round {round_num}/{global_rounds} ---")
        t_round_start = time.time()
        
        # 6.1 Server 下发模型
        global_params, _ = server.get_global_params_and_proj()
        for client in clients:
            client.receive_model(global_params)
            
        # 6.2 Client 本地训练 & 模拟提取特征
        client_features = []
        client_data_sizes = []
        active_ids = []
        
        for client in clients:
            client.phase1_local_train()
            # proj_seed 保持与轮数相关，确保所有端一致
            proj_seed = int(config['env'].get('seed', 42) + round_num) 
            feat, d_size = client.phase2_tee_process(proj_seed)
            
            client_features.append(feat)
            client_data_sizes.append(d_size)
            active_ids.append(client.client_id)
            
        # 6.3 Server 计算权重 (执行防御检测)
        server.calculate_weights(
            client_id_list=active_ids,
            client_features_dict_list=client_features,
            client_data_sizes=client_data_sizes,
            current_round=round_num
        )
        
        # 6.4 Server 明文聚合
        server.secure_aggregation(clients, active_ids, round_num)
        
        # 6.5 全局评估
        acc, loss = server.evaluate()
        history['acc'].append(acc)
        history['loss'].append(loss)
        print(f"  [Evaluate] Main Task Acc: {acc:.2f}% | Loss: {loss:.4f}")
        
        # 如果是目标攻击，评估 ASR (Attack Success Rate)
        if poison_loader and args.attack in ['backdoor', 'label_flip']:
            asr = server.evaluate_asr(test_loader, poison_loader)
            history['asr'].append(asr)
            print(f"  [Evaluate] Attack Success Rate (ASR): {asr:.2f}%")
            
        print(f"  [Time] Round {round_num} completed in {time.time() - t_round_start:.2f}s")

    # 7. 保存实验结果
    res_file = os.path.join(results_dir, "metrics.yaml")
    with open(res_file, 'w') as f:
        yaml.dump(history, f)
    print(f"\n✅ Experiment finished! Results saved to: {results_dir}")

if __name__ == "__main__":
    main()