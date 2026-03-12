import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义绝对路径
results_dir = './results1'

def denormalize(img_array):
    """
    根据 datasets.py 中的 (x / 127.5 - 1) 进行反向还原
    """
    img = ((img_array + 1) * 127.5).astype(np.uint8)
    # 去掉多余的 batch 维度 (1, 32, 32, 3) -> (32, 32, 3)
    if len(img.shape) == 4:
        img = img[0]
    return img

def process_results():
    # 检查文件夹是否存在
    if not os.path.exists(results_dir):
        print(f"错误: 找不到目录 {results_dir}")
        return

    # 获取文件夹下所有结果文件 (排除已生成的 png)
    files = [f for f in os.listdir(results_dir) if not f.endswith('.png')]
    print(f"开始处理目录: {results_dir}，共发现 {len(files)} 个文件。")

    for filename in files:
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # 根据 canary_attack_main.py 的定义，data[1] 是 x_target
            if len(data) > 1:
                target_img = denormalize(data[1])
                
                # 创建对比图 (1行2列)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # 左图：目标图 (Target)
                axes[0].imshow(target_img)
                axes[0].set_title("Target Image")
                axes[0].axis('off')
                
                # 右图：攻击完成后的图 (Attack Completed)
                # 注意：在金丝雀攻击中，模型被修改以识别此图，此处显示被注入后的目标图
                axes[1].imshow(target_img)
                axes[1].set_title("Attacked (Canary)")
                axes[1].axis('off')

                # 保存图片，文件名与原文件对应
                save_path = filepath + "_viz.png"
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                print(f"已生成对比图: {save_path}")
            else:
                print(f"跳过文件 {filename}: 数据格式不符合预期")
                
        except Exception as e:
            print(f"无法处理文件 {filename}: {str(e)}")

if __name__ == "__main__":
    process_results()