import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# ==========================================
# 基础配置与样式
# ==========================================
RESULTS_DIR = "./Integrated_Results"
OUTPUT_DIR = "./Grid_Figures_For_Paper"

# 定义要在网格中显示的维度顺序
MODELS = ["lenet5_mnist", "resnet20_cifar10", "resnet18_cifar10"]
RATIOS = [0.1, 0.2, 0.3, 0.4]

UNTARGETED_ATTACKS = ["random_poison", "gradient_amplify"]
TARGETED_ATTACKS = ["backdoor", "label_flip"]

# 论文级别的绘图样式字典
DEFENSE_STYLES = {
    'none': {'label': 'No Defense', 'color': '#d62728', 'style': '-', 'lw': 2, 'zorder': 2},
    'layers_proj_detect': {'label': 'OURS', 'color': '#1f77b4', 'style': '-', 'lw': 3, 'zorder': 10},
    'krum': {'label': 'Krum', 'color': '#2ca02c', 'style': '-', 'lw': 1.5, 'zorder': 3},
    'clustering': {'label': 'Clustering', 'color': '#ff7f0e', 'style': '-', 'lw': 1.5, 'zorder': 4},
    'median': {'label': 'Median', 'color': '#9467bd', 'style': '-', 'lw': 1.5, 'zorder': 5}
}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 辅助函数
# ==========================================
def get_style(def_name):
    return DEFENSE_STYLES.get(def_name, {'label': def_name, 'color': 'gray', 'style': '-', 'lw': 1.5, 'zorder': 1})

def smooth_curve_interpolation(scalars):
    """三次样条插值平滑"""
    if scalars is None or len(scalars) == 0: 
        return np.array([]), np.array([])
    x = np.arange(1, len(scalars) + 1)
    y = np.array(scalars)
    if len(x) < 4:
        return x, y
    x_smooth = np.linspace(x.min(), x.max(), 300)
    try:
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        y_smooth = np.clip(y_smooth, 0.0, 100.0)
        return x_smooth, y_smooth
    except Exception:
        return x, y

def find_npz_file(base_dir, model, attack, ratio, defense):
    """在归档目录中寻找 npz 数据文件"""
    search_path = os.path.join(base_dir, model, attack, f"p_{ratio}", "raw_data", defense)
    if not os.path.exists(search_path):
        return None
    for f in os.listdir(search_path):
        if f.endswith("_data.npz"):
            return os.path.join(search_path, f)
    return None

def find_baseline_npz(base_dir, model):
    """寻找纯净训练(Upper Bound)的 npz 数据文件"""
    search_path = os.path.join(base_dir, model, "pure_baseline_NoAttack")
    if not os.path.exists(search_path):
        return None
    for f in os.listdir(search_path):
        if f.endswith("_data.npz"):
            return os.path.join(search_path, f)
    return None

# ==========================================
# 核心绘图逻辑
# ==========================================
def plot_grid_figure(attack_name, metric_type):
    """
    绘制 3行 x 4列 的网格大图
    metric_type: 'accuracy' (用于Untargeted) 或 'asr' (用于Targeted)
    """
    print(f"🖼️ 正在生成 {attack_name.upper()} 攻击下的 {metric_type.upper()} 全局网格图...")
    
    # 创建 3x4 的大画布，调整尺寸以适应论文排版
    fig, axes = plt.subplots(nrows=len(MODELS), ncols=len(RATIOS), figsize=(25, 15), sharex=True, sharey=True)
    
    # 大图主标题
    fig.suptitle(f"Performance under {attack_name.replace('_', ' ').title()} Attack ({metric_type.upper()})", fontsize=24, fontweight='bold', y=0.95)

    lines_for_legend = []
    labels_for_legend = []
    legend_collected = False

    for i, model in enumerate(MODELS):
        # 提取模型显示名称
        model_display = model.split('_')[0].upper()
        
        # 预先读取该模型的 Upper Bound 数据 (仅 ACC 适用)
        baseline_npz = find_baseline_npz(RESULTS_DIR, model)
        baseline_data = None
        if baseline_npz and metric_type == 'accuracy':
            b_data = np.load(baseline_npz)
            if 'accuracy' in b_data:
                baseline_data = b_data['accuracy']
        
        for j, ratio in enumerate(RATIOS):
            ax = axes[i, j]
            
            # 画 Upper Bound (如果有)
            if baseline_data is not None:
                x_sm, y_sm = smooth_curve_interpolation(baseline_data)
                line, = ax.plot(x_sm, y_sm, label='Upper Bound (No Attack)', color='black', linestyle='--', linewidth=2, zorder=1)
                if not legend_collected:
                    lines_for_legend.append(line)
                    labels_for_legend.append('Upper Bound (No Attack)')
            
            # 遍历并绘制各个防御方法
            for defense in DEFENSE_STYLES.keys():
                npz_path = find_npz_file(RESULTS_DIR, model, attack_name, ratio, defense)
                if npz_path:
                    d_data = np.load(npz_path)
                    if metric_type in d_data and len(d_data[metric_type]) > 0:
                        raw_vals = d_data[metric_type]
                        # 对于 Targeted ASR，如果有全是 0 的防线，也正常画出
                        
                        x_sm, y_sm = smooth_curve_interpolation(raw_vals)
                        style = get_style(defense)
                        
                        line, = ax.plot(x_sm, y_sm, label=style['label'], color=style['color'], 
                                        linestyle=style['style'], linewidth=style['lw'], zorder=style['zorder'])
                        
                        # 收集图例 (只收集一次)
                        if not legend_collected:
                            lines_for_legend.append(line)
                            labels_for_legend.append(style['label'])

            # 设置子图标题和格式
            if i == 0:
                ax.set_title(f"Poison Ratio: {ratio * 100}%", fontsize=16, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{model_display}\n{metric_type.upper()} (%)", fontsize=14, fontweight='bold')
            if i == len(MODELS) - 1:
                ax.set_xlabel("Comm. Rounds", fontsize=14)
                
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0, 50) # 假设50轮
            ax.set_ylim(-2, 102) # 0-100% 的范围，留一点边距
            
            # 标记子图编号 (a), (b), (c)...
            subplot_letter = chr(97 + i * len(RATIOS) + j) # 97 is 'a'
            ax.text(0.5, -0.2, f"({subplot_letter})", transform=ax.transAxes, fontsize=14, ha='center')

        # 一行画完后，legend_collected 设为 True，避免重复收集
        legend_collected = True

    # 在整张图的底部正中央添加一个共享图例 (Shared Legend)
    fig.legend(lines_for_legend, labels_for_legend, loc='lower center', ncol=len(lines_for_legend), 
               fontsize=14, bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True)

    # 调整布局，为顶部的 Title 和底部的 Legend 留出空间
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    
    # 保存高分辨率图像
    out_filename = os.path.join(OUTPUT_DIR, f"Grid_{attack_name}_{metric_type}.png")
    plt.savefig(out_filename, dpi=600)
    plt.close()
    print(f"  ✅ 成功保存至: {out_filename}")

def main():
    print("🚀 启动网格矩阵大图生成器...")
    
    # 1. 为无目标攻击绘制 Accuracy
    for attack in UNTARGETED_ATTACKS:
        plot_grid_figure(attack, metric_type='accuracy')
        
    # 2. 为有目标攻击绘制 ASR
    for attack in TARGETED_ATTACKS:
        plot_grid_figure(attack, metric_type='asr')
        
    print(f"\n🎉 全部完成！大图已存放在 {OUTPUT_DIR} 文件夹中，您可以直接用于论文插入。")

if __name__ == "__main__":
    main()