import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR10, self).__init__()
        # 加载基础的 ResNet18 架构（不使用预训练权重）
        self.model = resnet18(weights=None)
        
        # [核心魔改] 替换 7x7 卷积为 3x3 卷积，防止下采样过度
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # [核心魔改] 去除影响空间分辨率的 MaxPool 
        self.model.maxpool = nn.Identity() 
        
        # 调整全连接层以匹配类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)