import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import medfilt

# ==========================================
# 基础配置与样式
# ==========================================
RESULTS_DIR = "./Integrated_Results"
OUTPUT_DIR = "./Grid_Figures_For_Paper"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Untargeted_Acc_Statistics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 维度定义
MODELS = ["lenet5_mnist", "resnet20_cifar10", "resnet18_cifar10"]
UNTARGETED_ATTACKS = ["random_poison", "gradient_amplify"]
RATIOS = [0.1, 0.2, 0.3, 0.4]
RATIO_LABELS = ["10%", "20%", "30%", "40%"]

DEFENSES = ['none', 'median', 'krum', 'clustering', 'layers_proj_detect']

# 包含特定 Marker 的精美绘图样式
DEFENSE_STYLES = {
    'none': {'label': 'No Defense', 'color': '#d62728', 'style': '-', 'lw': 1.5, 'marker': 'v', 'markersize': 8, 'zorder': 2},
    'layers_proj_detect': {'label': 'OURS', 'color': '#1f77b4', 'style': '-', 'lw': 2.5, 'marker': 'o', 'markersize': 10, 'zorder': 10},
    'krum': {'label': 'Krum', 'color': '#2ca02c', 'style': '-', 'lw': 1.5, 'marker': '^', 'markersize': 8, 'zorder': 3},
    'clustering': {'label': 'Clustering', 'color': '#ff7f0e', 'style': '-', 'lw': 1.5, 'marker': 'x', 'markersize': 8, 'zorder': 4},
    'median': {'label': 'Median', 'color': '#9467bd', 'style': '-', 'lw': 1.5, 'marker': 's', 'markersize': 8, 'zorder': 5}
}

def find_npz_file(base_dir, model, attack, ratio, defense):
    search_path = os.path.join(base_dir, model, attack, f"p_{ratio}", "raw_data", defense)
    if not os.path.exists(search_path): return None
    for f in os.listdir(search_path):
        if f.endswith("_data.npz"): return os.path.join(search_path, f)
    return None

def find_baseline_npz(base_dir, model):
    search_path = os.path.join(base_dir, model, "pure_baseline_NoAttack")
    if not os.path.exists(search_path): return None
    for f in os.listdir(search_path):
        if f.endswith("_data.npz"): return os.path.join(search_path, f)
    return None

def main():
    print("📊 正在提取 Untargeted Attacks 的收敛期平均准确率 (最后 20% 轮次)...")
    
    plot_data = {attack: {model: {defense: [] for defense in DEFENSES} for model in MODELS} for attack in UNTARGETED_ATTACKS}
    baseline_data = {model: None for model in MODELS}
    csv_records = []

    # 1. 提取基线 (Upper Bound) 数据，并将其加入 CSV 记录
    for model in MODELS:
        b_npz = find_baseline_npz(RESULTS_DIR, model)
        if b_npz:
            data = np.load(b_npz)
            if 'accuracy' in data and len(data['accuracy']) > 0:
                acc_array = data['accuracy']
                start_idx = int(len(acc_array) * 0.8) 
                val = np.mean(acc_array[start_idx:])
                baseline_data[model] = val
                
                # 【新增】将 NoAttack 的干净准确率单独作为一行加入表格中
                row_data = {
                    "Attack": "NoAttack (Baseline)", 
                    "Model": model, 
                    "Defense": "Upper Bound"
                }
                # 对于 NoAttack，不同投毒比例下的准确率是同一个常量
                for ratio in RATIOS:
                    row_data[f"{int(ratio*100)}%"] = f"{val:.2f}"
                csv_records.append(row_data)

    # 2. 提取攻击下的防御数据
    for attack in UNTARGETED_ATTACKS:
        for model in MODELS:
            for defense in DEFENSES:
                row_data = {"Attack": attack, "Model": model, "Defense": DEFENSE_STYLES[defense]['label']}
                
                for ratio in RATIOS:
                    npz_path = find_npz_file(RESULTS_DIR, model, attack, ratio, defense)
                    val = np.nan
                    if npz_path:
                        data = np.load(npz_path)
                        if 'accuracy' in data and len(data['accuracy']) > 0:
                            acc_array = data['accuracy']
                            start_idx = int(len(acc_array) * 0.8)
                            val = np.mean(acc_array[start_idx:])
                    
                    plot_data[attack][model][defense].append(val)
                    row_data[f"{int(ratio*100)}%"] = f"{val:.2f}" if not np.isnan(val) else "N/A"
                    
                csv_records.append(row_data)

    # 保存数据到 CSV 供写论文时引用
    df = pd.DataFrame(csv_records)
    df.set_index(["Attack", "Model", "Defense"], inplace=True)
    df.to_csv(OUTPUT_CSV)
    print(f"✅ 数据提取完成，已包含 NoAttack 基线，保存至: {OUTPUT_CSV}")

    # ==========================================
    # 绘制横向一排长图 (30x6.5)
    # ==========================================
    print("🖼️ 正在生成 1排排版 的长图趋势图...")
    
    total_subplots = len(UNTARGETED_ATTACKS) * len(MODELS)
    fig, axes = plt.subplots(nrows=1, ncols=total_subplots, figsize=(30, 6.5), sharey=True)

    lines_for_legend = []
    labels_for_legend = []
    legend_collected = False

    plot_idx = 0
    for attack in UNTARGETED_ATTACKS:
        for model in MODELS:
            ax = axes[plot_idx]
            
            # 画 Upper Bound (水平虚线)
            if baseline_data[model] is not None:
                b_val = baseline_data[model]
                line = ax.axhline(y=b_val, color='#7f7f7f', linestyle='--', linewidth=2, zorder=1)
                if not legend_collected:
                    lines_for_legend.append(line)
                    labels_for_legend.append('Upper Bound (No Attack)')

            # 画各种防御的折线
            for defense in DEFENSES:
                y_values = plot_data[attack][model][defense]
                valid_x = [RATIO_LABELS[idx] for idx, val in enumerate(y_values) if not np.isnan(val)]
                valid_y = [val for val in y_values if not np.isnan(val)]
                
                if valid_y:
                    style = DEFENSE_STYLES[defense]
                    line, = ax.plot(valid_x, valid_y, label=style['label'], 
                                    color=style['color'], linestyle=style['style'], 
                                    linewidth=style['lw'], marker=style['marker'], 
                                    markersize=style['markersize'], zorder=style['zorder'])
                    
                    if not legend_collected:
                        lines_for_legend.append(line)
                        labels_for_legend.append(style['label'])

            # 子图格式化
            ax.grid(True, linestyle='-', alpha=0.5)
            ax.set_ylim(-5, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            
            # 只有第一个子图显示 Y 轴百分比标签
            if plot_idx == 0:
                ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)
                ax.set_ylabel("Accuracy", fontsize=24, fontweight='bold')
            
            ax.tick_params(axis='x', labelsize=14)
            
            # 子图描述文字
            model_display = model.split('_')[0].upper()
            atk_display = attack.replace('_', ' ').title()
            subplot_letter = chr(97 + plot_idx)
            
            ax.text(0.5, -0.1, f"({subplot_letter}) {atk_display}\n{model_display}", 
                    transform=ax.transAxes, fontsize=24, fontweight='bold', ha='center', va='top')

            plot_idx += 1
            legend_collected = True 

    # ==========================================
    # 全局悬浮图例 (顶部居中)
    # ==========================================
    fig.legend(lines_for_legend, labels_for_legend, loc='upper center', 
               ncol=len(lines_for_legend), fontsize=24, 
               bbox_to_anchor=(0.5, 1.0), frameon=False)

    # 调整布局
    plt.subplots_adjust(top=0.85, bottom=0.22, left=0.05, right=0.98, wspace=0.15)
    
    out_img = os.path.join(OUTPUT_DIR, "Grid_Untargeted_Ratio_vs_Accuracy_SingleRow.pdf")
    plt.savefig(out_img, dpi=600)
    plt.close()
    
    print(f"🎉 横向 1排 绘图完成！大图已保存至: {out_img}")

if __name__ == "__main__":
    main()