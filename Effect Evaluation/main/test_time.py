import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import time

# ===== 基本参数 =====
NUM_CLIENTS = 20
LOCAL_EPOCHS = 3
ROUNDS = 5
BATCH_SIZE = 128
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ===== 数据加载 =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 均匀划分客户端数据
client_data_size = len(trainset) // NUM_CLIENTS
client_datasets = torch.utils.data.random_split(
    trainset,
    [client_data_size] * NUM_CLIENTS
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=4
)

# ===== 模型 =====
def get_model():
    model = torchvision.models.resnet18(num_classes=10)
    return model.to(DEVICE)

global_model = get_model()

# ===== 评估函数 =====
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ===== 联邦训练 =====
for round in range(ROUNDS):

    start_time = time.time()
    local_weights = []

    for client_id in range(NUM_CLIENTS):

        local_model = copy.deepcopy(global_model)
        local_model.train()

        trainloader = torch.utils.data.DataLoader(
            client_datasets[client_id],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        optimizer = optim.SGD(local_model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(LOCAL_EPOCHS):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_weights.append(copy.deepcopy(local_model.state_dict()))

    # ===== FedAvg 聚合 =====
    global_weights = copy.deepcopy(local_weights[0])

    for key in global_weights.keys():
        for i in range(1, NUM_CLIENTS):
            global_weights[key] += local_weights[i][key]
        global_weights[key] = torch.div(global_weights[key], NUM_CLIENTS)

    global_model.load_state_dict(global_weights)

    acc = evaluate(global_model)
    round_time = time.time() - start_time

    print(f"Round {round+1}/{ROUNDS} | "
          f"Accuracy: {acc:.2f}% | "
          f"Time: {round_time:.2f}s")