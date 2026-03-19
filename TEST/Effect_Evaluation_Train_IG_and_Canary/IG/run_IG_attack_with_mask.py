"""
Example script to run attacks in this repository directly without simulation.
Modified to accept a specific input image, apply an ADDITIVE RANDOM MASK to gradients,
perform Inverting Gradients (IG) attack, and save the reconstructed image.
"""
import os
import argparse
import logging
import torch
import torchvision
import breaching
from omegaconf import DictConfig
from PIL import Image
import torchvision.utils as vutils

# 配置基础的日志输出，确保能看到 breaching 的底层打印信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_cfg_default = DictConfig({
    "modality": "vision",
    "task": "classification",
    "size": 1_281_167,
    "classes": 1000,
    "shape": [3, 224, 224],
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
})

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
    ]
)

def denormalize(tensor, mean, std):
    """将经过标准化的张量反标准化，以便保存为正常图像"""
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std_tensor + mean_tensor
    return torch.clamp(tensor, 0, 1)

def main(args):
    setup = dict(device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"), dtype=torch.float)

    print(f"Loading model ResNet152 on {setup['device']}...")
    model = torchvision.models.resnet152(pretrained=True).to(**setup)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Loading input image from {args.image_path}...")
    # 读取指定图片并进行预处理
    image = Image.open(args.image_path).convert("RGB")
    datapoint = transforms(image).to(**setup)
    labels = torch.as_tensor([args.label]).to(setup["device"])

    print("Initializing Inverting Gradients (IG) attacker...")
    cfg_attack = breaching.get_attack_config("invertinggradients")
    
    # +++ 关键修改区：优化控制与进度显示 +++
    # 修改最大迭代次数（默认通常很大），缩小它能大幅加快运行速度以用于测试验证
    cfg_attack.optim.max_iterations = args.max_iterations 
    # 控制日志输出的频率，比如每 50 次迭代向终端输出一次当前进度和 Loss
    cfg_attack.optim.callback = 50  
    
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

    print("Simulating federated learning protocol computations...")
    # Server-side computation:
    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]
    
    # User-side computation:
    loss = loss_fn(model(datapoint[None, ...]), labels)
    
    # 1. 计算出原始的真实梯度
    original_gradients = torch.autograd.grad(loss, model.parameters())
    
    # 2. 生成并施加加性随机数掩码
    masked_gradients = []
    print(f"Applying additive random mask with scale: {args.mask_scale}")
    for g in original_gradients:
        # 生成与梯度张量等长的随机数 (正态分布)，并乘以掩码强度缩放因子
        mask = torch.randn_like(g) * args.mask_scale
        # 将掩码直接叠加到真实梯度上
        masked_gradients.append(g + mask)
        
    # 转换为 tuple 以匹配原始数据结构
    masked_gradients = tuple(masked_gradients)

    # 3. 将带有掩码的梯度打包共享
    shared_data = [
        dict(
            gradients=masked_gradients,
            buffers=None,
            metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None,),
        )
    ]

    print(f"Starting the gradient inversion attack for {args.max_iterations} iterations...")
    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)

    # 提取重构的张量数据并进行反标准化处理
    recon_tensor = reconstructed_user_data["data"]
    recon_tensor_denorm = denormalize(recon_tensor[0], data_cfg_default.mean, data_cfg_default.std)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构造输出路径并保存，文件名包含掩码强度信息便于对照实验
    filename = os.path.basename(args.image_path).split('.')[0]
    output_path = os.path.join(args.output_dir, f"{filename}_recon_mask_{args.mask_scale}.png")
    vutils.save_image(recon_tensor_denorm, output_path)
    print(f"Attack finished! Reconstructed image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inverting Gradients attack on a masked gradient.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--label", type=int, default=1, help="Ground truth class label for the image (default: 1).")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the reconstructed image.")
    parser.add_argument("--mask_scale", type=float, default=1, help="Scale (standard deviation) of the additive random mask.")
    parser.add_argument("--max_iterations", type=int, default=500, help="Max iterations for the optimization attack.")
    args = parser.parse_args()
    main(args)