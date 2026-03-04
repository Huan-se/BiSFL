import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os

def get_transform(dataset_name, is_train=True):
    """获取对应数据集的预处理转换，区分训练集和测试集"""
    if dataset_name == "mnist":
        # MNIST 较为简单，通常不需要数据增强
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "cifar10":
        if is_train:
            # [关键修改] CIFAR-10 训练集：加入标准数据增强，防止 ResNet 过拟合
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            # [关键修改] CIFAR-10 测试集：仅做张量转换和标准化，不进行随机增强
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

def load_dataset(dataset_name, data_dir="./data"):
    """加载指定数据集的训练集和测试集"""
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 区分训练集和测试集的 Transform
    train_transform = get_transform(dataset_name, is_train=True)
    test_transform = get_transform(dataset_name, is_train=False)

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=test_transform
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_dataset, test_dataset

def split_iid(dataset, num_clients, batch_size):
    """IID数据划分"""
    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    split_size = total_size // num_clients
    client_indices = [
        indices[i * split_size: (i + 1) * split_size].tolist()
        for i in range(num_clients)
    ]

    # 处理剩余数据
    remaining = indices[num_clients * split_size:]
    for i in range(len(remaining)):
        client_indices[i].append(remaining[i])

    # 创建数据加载器
    client_dataloaders = []
    for indices in client_indices:
        # 确保数据量不小于批次大小
        if len(indices) < batch_size:
            # 如果数据不够一个batch，简单复制填充（边缘情况）
            while len(indices) < batch_size:
                indices.extend(indices[:min(len(indices), batch_size - len(indices))])
        
        subset = Subset(dataset, indices)
        client_dataloaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )

    return client_dataloaders

def split_noniid(dataset, num_clients, batch_size, dataset_name, alpha=0.1):
    """Non-IID数据划分（基于Dirichlet分布）"""
    if dataset_name == "mnist":
        targets = dataset.targets.numpy()
    elif dataset_name == "cifar10":
        targets = np.array(dataset.targets)
    else:
        # 尝试通用获取
        targets = np.array(dataset.targets)

    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]
    class_indices = {c: np.where(targets == c)[0] for c in classes}

    for c in classes:
        # 为每个类别生成Dirichlet分布的客户端分配权重
        dirichlet_weights = np.random.dirichlet(np.ones(num_clients) * alpha)
        # 归一化并计算划分点
        dirichlet_weights = dirichlet_weights / dirichlet_weights.sum()
        class_size = len(class_indices[c])
        
        split_points = (np.cumsum(dirichlet_weights) * class_size).astype(int)[:-1]
        client_split = np.split(class_indices[c], split_points)
        
        for i, idx in enumerate(client_split):
            client_indices[i].extend(idx.tolist())

    # 创建数据加载器
    client_dataloaders = []
    for indices in client_indices:
        # 混洗当前客户端的索引
        np.random.shuffle(indices)
        
        # 确保数据量不小于批次大小
        if len(indices) < batch_size:
             if len(indices) == 0:
                 # 极端Non-IID可能导致某客户端无数据，防止报错分配少量随机数据
                 print(f"Warning: Client has no data for Non-IID alpha={alpha}. Assigning random sample.")
                 indices = np.random.choice(len(dataset), batch_size).tolist()
             else:
                while len(indices) < batch_size:
                    indices.extend(indices[:min(len(indices), batch_size - len(indices))])
        
        subset = Subset(dataset, indices)
        client_dataloaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )

    return client_dataloaders

def load_and_split_dataset(dataset_name, num_clients, batch_size, if_noniid=True, alpha=0.1, data_dir="./main/data"):
    """
    统一接口：加载并划分数据集
    """
    # 加载原始数据集
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)

    # 划分训练集
    if if_noniid:
        print(f"  [Data] Splitting {dataset_name} Non-IID (alpha={alpha})...")
        client_dataloaders = split_noniid(
            train_dataset, num_clients, batch_size, dataset_name, alpha
        )
    else:
        print(f"  [Data] Splitting {dataset_name} IID...")
        client_dataloaders = split_iid(
            train_dataset, num_clients, batch_size
        )

    # 创建测试集加载器
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return client_dataloaders, test_loader