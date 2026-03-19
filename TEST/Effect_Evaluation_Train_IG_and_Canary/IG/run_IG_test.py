"""
Unified Experiment Script for Inverting Gradients (IG) Attack Defence Verification.
"""
import os
import argparse
import logging
import torch
import torchvision
import breaching
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from PIL import Image
import torchvision.utils as vutils

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

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=data_cfg_default.mean, std=data_cfg_default.std),
])

def denormalize(tensor, mean, std):
    recon_tensor = tensor.detach().cpu()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    recon_tensor = recon_tensor * std_tensor + mean_tensor
    return torch.clamp(recon_tensor, 0, 1)

def save_image_tensor(tensor, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    vutils.save_image(tensor, output_path)
    logging.info(f"Image saved to: {output_path}")

def get_attack_config_safe(attack_name):
    """安全获取 attack config，调用前后均清理 GlobalHydra 避免重复初始化报错"""
    GlobalHydra.instance().clear()
    cfg = breaching.get_attack_config(attack_name)
    GlobalHydra.instance().clear()
    return cfg

def run_attack(attacker, server_payload, gradients, labels, setup, task_name):
    logging.info(f"--- Starting {task_name} ... (This will take a significant amount of time) ---")
    shared_data = [
        dict(
            gradients=tuple(gradients),
            buffers=None,
            metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None),
        )
    ]
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)
    recon_tensor = reconstructed_user_data["data"]
    recon_tensor_denorm = denormalize(recon_tensor[0], data_cfg_default.mean, data_cfg_default.std)
    return recon_tensor_denorm

def main(args):
    setup = dict(device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"), dtype=torch.float)

    logging.info(f"Loading ResNet18 model on {setup['device']}...")
    model = torchvision.models.resnet18(pretrained=True).to(**setup)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # 使用安全版本初始化 attacker
    print("Initializing Inverting Gradients (IG) attacker...")
    cfg_attack = get_attack_config_safe("invertinggradients")
    cfg_attack.optim.callback = 2000
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

    server_payload = [
        dict(
            parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
        )
    ]

    logging.info(f"Loading input image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    datapoint = transforms(image).to(**setup)
    labels = torch.as_tensor([args.label]).to(setup["device"])
    filename_base = os.path.basename(args.image_path).split('.')[0]

    # 任务一：保存裁剪后的输入图像
    logging.info("--- Outputting the actual cropped input image (Ground Truth)... ---")
    cropped_input_denorm = denormalize(datapoint, data_cfg_default.mean, data_cfg_default.std)
    save_image_tensor(cropped_input_denorm, args.output_dir, f"{filename_base}_cropped_input.png")

    # 计算真实梯度
    logging.info("Computing original gradients...")
    loss = loss_fn(model(datapoint[None, ...]), labels)
    original_gradients_tuple = torch.autograd.grad(loss, model.parameters())
    original_gradients_list = [g.detach().clone() for g in original_gradients_tuple]

    # 任务二：无掩码 IG 攻击
    recon_clean = run_attack(attacker, server_payload, original_gradients_list, labels, setup, "Clean IG Attack")
    save_image_tensor(recon_clean, args.output_dir, f"{filename_base}_recon_clean.png")

    # 任务三：加性掩码 IG 攻击
    logging.info("Applying additive random mask with scale 1.0 to gradients...")
    masked_gradients_list = []
    for g in original_gradients_list:
        mask = torch.randn_like(g) * 100
        masked_gradients_list.append(g + mask)

    recon_masked = run_attack(attacker, server_payload, masked_gradients_list, labels, setup, "Masked IG Attack (Scale 1.0)")
    save_image_tensor(recon_masked, args.output_dir, f"{filename_base}_recon_mask_1.0.png")

    logging.info(f"Successfully output all three images to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script to save cropped input and run clean/masked IG attacks.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./experiment_output")
    args = parser.parse_args()
    main(args)