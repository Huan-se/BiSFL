import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 基础配置与样式
# ==========================================
RESULTS_DIR = "./Integrated_Results"
OUTPUT_DIR = "./Grid_Figures_For_Paper"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Targeted_ASR_Statistics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 维度定义
MODELS = ["lenet5_mnist", "resnet20_cifar10", "resnet18_cifar10"]
TARGETED_ATTACKS = ["backdoor", "label_flip"]
RATIOS = [0.1, 0.2, 0.3, 0.4]
RATIO_LABELS = ["10%", "20%", "30%", "40%"]

# 按照论文展示顺序排列
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
    """在归档目录中寻找 npz 数据文件"""
    search_path = os.path.join(base_dir, model, attack, f"p_{ratio}", "raw_data", defense)
    if not os.path.exists(search_path): return None
    for f in os.listdir(search_path):
        if f.endswith("_data.npz"): return os.path.join(search_path, f)
    return None

def main():
    print("📊 正在提取 Targeted Attacks 的 ASR 数据 (截取后 3/5 轮次)...")
    
    # 绘图数据存储 (存 Median ASR 用于画图)
    plot_data = {attack: {model: {defense: [] for defense in DEFENSES} for model in MODELS} for attack in TARGETED_ATTACKS}
    csv_records = []

    for attack in TARGETED_ATTACKS:
        for model in MODELS:
            dataset_model_str = model.replace('_', ' ').upper()
            attack_str = attack.replace('_', ' ').title()
            
            for defense in DEFENSES:
                row_data = {
                    "Attack": attack_str,
                    "Model": dataset_model_str,
                    "Defense": DEFENSE_STYLES[defense]['label']
                }
                
                for ratio in RATIOS:
                    npz_path = find_npz_file(RESULTS_DIR, model, attack, ratio, defense)
                    plot_val = np.nan
                    
                    if npz_path:
                        data = np.load(npz_path)
                        if 'asr' in data and len(data['asr']) > 0:
                            asr_array = data['asr']
                            
                            # 核心修改：舍弃前 2/5 轮次，保留后 3/5 轮次
                            start_idx = len(asr_array) * 2 // 5
                            valid_asr = asr_array[start_idx:]
                            
                            if len(valid_asr) > 0:
                                max_asr = np.max(valid_asr)
                                min_asr = np.min(valid_asr)
                                median_asr = np.median(valid_asr)
                                
                                # 用于绘图的数据：中位数 ASR (最能代表稳态表现)
                                plot_val = max_asr
                                
                                # 用于表格的数据: Median (Min-Max)
                                stat_str = f"{median_asr:.1f} ({min_asr:.1f}-{max_asr:.1f})"
                            else:
                                stat_str = "N/A"
                        else:
                            stat_str = "N/A"
                    else:
                        stat_str = "N/A"
                        
                    plot_data[attack][model][defense].append(plot_val)
                    row_data[f"{int(ratio*100)}%"] = stat_str
                    
                csv_records.append(row_data)

    # 1. 保存极具学术说服力的 CSV 表格
    df = pd.DataFrame(csv_records)
    df.set_index(["Attack", "Model", "Defense"], inplace=True)
    df.to_csv(OUTPUT_CSV)
    print(f"✅ ASR 统计完成！数据 [Median (Min-Max)] 已保存至: {OUTPUT_CSV}")

    # ==========================================
    # 2. 绘制 1 行 x 6 列 的网格大图 (Targeted ASR)
    # ==========================================
    print("🖼️ 正在生成 Targeted Attacks 的 Median ASR 趋势大图...")
    
    total_subplots = len(TARGETED_ATTACKS) * len(MODELS)
    fig, axes = plt.subplots(nrows=1, ncols=total_subplots, figsize=(30, 6.5), sharey=True)

    lines_for_legend = []
    labels_for_legend = []
    legend_collected = False

    plot_idx = 0
    for attack in TARGETED_ATTACKS:
        for model in MODELS:
            ax = axes[plot_idx]

            # 画各种防御的折线 (基于 Median ASR)
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
            # ASR 范围 0 - 100%
            ax.set_ylim(-5, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            
            # 只有第一个子图显示 Y 轴百分比标签
            if plot_idx == 0:
                ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=14)
                ax.set_ylabel("MAX ASR ", fontsize=24, fontweight='bold')
            
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
    
    out_img = os.path.join(OUTPUT_DIR, "Grid_Targeted_Ratio_vs_ASR_SingleRow.pdf")
    plt.savefig(out_img, dpi=600)
    plt.close()
    
    print(f"🎉 Targeted 攻击 1排 绘图完成！大图已保存至: {out_img}")

if __name__ == "__main__":
    main()