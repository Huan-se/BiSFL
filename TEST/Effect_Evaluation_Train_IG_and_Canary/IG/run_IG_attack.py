"""
Example script to run attacks in this repository directly without simulation.
Modified to accept a specific input image, perform Inverting Gradients (IG) attack,
and explicitly show optimization progress to avoid feeling "stuck".
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

    # 建议：如果只是测试全流程，可以先把 resnet152 换成 resnet18 会快很多。这里依然保留你选用的模型
    print(f"Loading model on {setup['device']}...")
    model = torchvision.models.resnet152(pretrained=True).to(**setup)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Loading input image from {args.image_path}...")
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
    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]
    
    loss = loss_fn(model(datapoint[None, ...]), labels)
    shared_data = [
        dict(
            gradients=torch.autograd.grad(loss, model.parameters()),
            buffers=None,
            metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None,),
        )
    ]

    print(f"Starting the gradient inversion attack for {args.max_iterations} iterations...")
    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)

    recon_tensor = reconstructed_user_data["data"]
    recon_tensor_denorm = denormalize(recon_tensor[0], data_cfg_default.mean, data_cfg_default.std)

    os.makedirs(args.output_dir, exist_ok=True)
    
    filename = os.path.basename(args.image_path).split('.')[0]
    output_path = os.path.join(args.output_dir, f"{filename}_reconstructed_iter_{args.max_iterations}.png")
    vutils.save_image(recon_tensor_denorm, output_path)
    print(f"\nAttack finished! Reconstructed image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inverting Gradients attack with progress logging.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--label", type=int, default=1, help="Ground truth class label for the image.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save.")
    # 新增了一个参数，允许你在命令行直接指定要跑多少次迭代
    parser.add_argument("--max_iterations", type=int, default=500, help="Max iterations for the optimization attack.")
    
    args = parser.parse_args()
    main(args)