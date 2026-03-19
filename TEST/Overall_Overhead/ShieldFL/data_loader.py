import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random

class PoisonedDataset(Dataset):
    """
    数据投毒包装器：模拟恶意客户端篡改本地数据标签（Label-flipping）
    """
    def __init__(self, original_dataset, num_classes=10):
        self.dataset = original_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 随机投毒：随机分配一个错误的标签
        poisoned_label = random.choice([i for i in range(self.num_classes) if i != label])
        return image, poisoned_label

def get_federated_dataloaders(dataset_name='MNIST', num_clients=10, poison_rate=0.2, batch_size=64):
    """
    获取联邦学习的数据加载器，支持 MNIST 和 CIFAR-10
    """
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset. Choose 'MNIST' or 'CIFAR10'.")

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 修复余数除不尽的问题
    dataset_len = len(train_dataset)
    base_size = dataset_len // num_clients
    split_lengths = [base_size] * num_clients
    # 将除不尽的余数全部补偿给最后一个客户端
    split_lengths[-1] += dataset_len - sum(split_lengths)
    
    # 使用修复后的长度列表进行切分
    data_split = torch.utils.data.random_split(train_dataset, split_lengths)
    
    client_dataloaders = []
    num_malicious = int(num_clients * poison_rate)
    
    for i in range(num_clients):
        local_data = data_split[i]
        total_size = len(local_data)
        enclave_size = int(total_size * 0.1) # 10% 作为 TEE 的可信子集
        
        indices = np.random.permutation(total_size)
        enclave_indices = indices[:enclave_size]
        host_indices = indices[enclave_size:]
        
        enclave_subset = Subset(local_data, enclave_indices)
        host_subset = Subset(local_data, host_indices)
        
        is_malicious = (i < num_malicious)
        
        # 对恶意节点的 Host 外部数据进行标签投毒
        if is_malicious:
            host_subset = PoisonedDataset(host_subset, num_classes=num_classes)
            
        host_loader = DataLoader(host_subset, batch_size=batch_size, shuffle=True)
        enclave_loader = DataLoader(enclave_subset, batch_size=batch_size, shuffle=True)
        
        client_dataloaders.append({
            'client_id': i,
            'is_malicious': is_malicious,
            'host_loader': host_loader,
            'enclave_loader': enclave_loader,
            'm_host': len(host_subset),
            'm_enclave': len(enclave_subset)
        })
        
    return client_dataloaders, test_loader