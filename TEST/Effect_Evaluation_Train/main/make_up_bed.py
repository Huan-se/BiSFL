import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 搜索目录与输出集成目录
SEARCH_DIR = "." 
TARGET_DIR = "./Integrated_Results"

# 防御方法的绘图样式字典
DEFENSE_STYLES = {
    'none': {'label': 'No Defense (Lower Bound)', 'color': '#d62728', 'style': '-', 'lw': 2, 'marker': ''},
    'layers_proj_detect': {'label': 'OURS (Layers Proj)', 'color': '#1f77b4', 'style': '-', 'lw': 2.5, 'marker': ''},
    'krum': {'label': 'Krum', 'color': '#2ca02c', 'style': '-', 'lw': 1.5, 'marker': ''},
    'clustering': {'label': 'Clustering', 'color': '#ff7f0e', 'style': '-', 'lw': 1.5, 'marker': ''},
    'median': {'label': 'Median', 'color': '#9467bd', 'style': '-', 'lw': 1.5, 'marker': ''}
}

def get_style(def_name):
    return DEFENSE_STYLES.get(def_name, {'label': def_name, 'color': 'gray', 'style': '-', 'lw': 1.5, 'marker': ''})

def main():
    print("🔍 正在扫描结果文件...")
    json_files = list(Path(SEARCH_DIR).rglob('*_config.json'))
    
    if not json_files:
        print("⚠️ 未找到任何 *_config.json 文件，请确认您的实验已经跑出结果且执行目录正确。")
        return

    # 数据归类字典
    pure_baselines = {} # pure_baselines[model][dataset] = npz_path
    exp_groups = {}     # exp_groups[model][dataset][attack][ratio][defense] = {...}

    for j_path in json_files:
        if "Integrated_Results" in str(j_path): 
            continue # 跳过已经整理过的目录
            
        with open(j_path, 'r') as f:
            data = json.load(f)
        cfg = data['config']
        
        model = cfg['data']['model']
        dataset = cfg['data']['dataset']
        defense = cfg['defense']['method']
        
        active_attacks = cfg.get('attack', {}).get('active_attacks', [])
        attack = "+".join(active_attacks) if active_attacks else "none"
        ratio = cfg.get('attack', {}).get('poison_ratio', 0.0)

        npz_path = str(j_path).replace('_config.json', '_data.npz')
        if not os.path.exists(npz_path):
            continue

        # 记录上限基线 (No Attack)
        if attack == 'none':
            if model not in pure_baselines: pure_baselines[model] = {}
            pure_baselines[model][dataset] = {
                'npz': npz_path, 'dir': os.path.dirname(str(j_path))
            }
            continue 
        
        # 构建分组结构
        if model not in exp_groups: exp_groups[model] = {}
        if dataset not in exp_groups[model]: exp_groups[model][dataset] = {}
        if attack not in exp_groups[model][dataset]: exp_groups[model][dataset][attack] = {}
        if ratio not in exp_groups[model][dataset][attack]: exp_groups[model][dataset][attack][ratio] = {}
        
        exp_groups[model][dataset][attack][ratio][defense] = {
            'npz': npz_path,
            'json': str(j_path),
            'dir': os.path.dirname(str(j_path))
        }

    print("📊 开始生成对比图并整理文件...")
    
    for model in exp_groups:
        for dataset in exp_groups[model]:
            baseline_info = pure_baselines.get(model, {}).get(dataset, None)
            
            # 首先把该模型的 baseline 复制到专属文件夹
            if baseline_info:
                base_target_dir = os.path.join(TARGET_DIR, f"{model}_{dataset}", "pure_baseline_NoAttack")
                if not os.path.exists(base_target_dir):
                    shutil.copytree(baseline_info['dir'], base_target_dir)

            for attack in exp_groups[model][dataset]:
                for ratio in exp_groups[model][dataset][attack]:
                    group_defenses = exp_groups[model][dataset][attack][ratio]
                    
                    target_group_dir = os.path.join(TARGET_DIR, f"{model}_{dataset}", attack, f"p_{ratio}")
                    os.makedirs(target_group_dir, exist_ok=True)
                    
                    # ==========================================
                    # 1. 绘制 Accuracy 对比图
                    # ==========================================
                    plt.figure(figsize=(10, 6))
                    
                    # 画基线 (Upper Bound)
                    if baseline_info:
                        b_data = np.load(baseline_info['npz'])
                        acc = b_data['accuracy']
                        plt.plot(np.arange(1, len(acc)+1), acc, label='Upper Bound (No Attack)', color='black', linestyle='--', linewidth=2)
                    
                    # 画所有防御方法
                    for def_name, paths in group_defenses.items():
                        d_data = np.load(paths['npz'])
                        acc = d_data['accuracy']
                        style = get_style(def_name)
                        label_with_acc = f"{style['label']} ({acc[-1]:.1f}%)"
                        plt.plot(np.arange(1, len(acc)+1), acc, label=label_with_acc, color=style['color'], linestyle=style['style'], linewidth=style['lw'])
                    
                    plt.title(f"Accuracy Comparison\nModel: {model.upper()} | Dataset: {dataset.upper()} | Attack: {attack.upper()} | Poison Ratio: {ratio}")
                    plt.xlabel("Communication Rounds")
                    plt.ylabel("Accuracy (%)")
                    plt.legend(loc='lower right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(target_group_dir, "Comparison_Accuracy.png"), dpi=300)
                    plt.close()

                    # ==========================================
                    # 2. 绘制 ASR 对比图 (仅在数据存在时)
                    # ==========================================
                    has_asr = False
                    plt.figure(figsize=(10, 6))
                    for def_name, paths in group_defenses.items():
                        d_data = np.load(paths['npz'])
                        if 'asr' in d_data and len(d_data['asr']) > 0 and np.max(d_data['asr']) > 0:
                            has_asr = True
                            asr = d_data['asr']
                            style = get_style(def_name)
                            label_with_asr = f"{style['label']} ({asr[-1]:.1f}%)"
                            plt.plot(np.arange(1, len(asr)+1), asr, label=label_with_asr, color=style['color'], linestyle=style['style'], linewidth=style['lw'])
                    
                    if has_asr:
                        plt.title(f"Attack Success Rate (ASR) Comparison\nModel: {model.upper()} | Dataset: {dataset.upper()} | Attack: {attack.upper()} | Poison Ratio: {ratio}")
                        plt.xlabel("Communication Rounds")
                        plt.ylabel("Attack Success Rate (%)")
                        plt.legend(loc='upper left')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(target_group_dir, "Comparison_ASR.png"), dpi=300)
                    plt.close()

                    # ==========================================
                    # 3. 归档原始文件
                    # ==========================================
                    raw_dir = os.path.join(target_group_dir, "raw_data")
                    os.makedirs(raw_dir, exist_ok=True)
                    
                    for def_name, paths in group_defenses.items():
                        src_dir = paths['dir']
                        dest_dir = os.path.join(raw_dir, def_name)
                        if not os.path.exists(dest_dir):
                            shutil.copytree(src_dir, dest_dir)
                            
                    print(f"  ✅ 集成完毕: {model}_{dataset} | {attack} | p_{ratio}")

    print(f"\n🎉 整理完成！所有对比结果和归档数据均已放入: {os.path.abspath(TARGET_DIR)}")

if __name__ == "__main__":
    main()