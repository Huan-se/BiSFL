#!/bin/bash

# ==========================================
# PPFL-TEE 攻防实验矩阵 (安全并发版)
# ==========================================

# ⚠️ 核心设置：最大并发任务数
# 请根据你的 GPU 显存来计算：假设每个 ResNet 实验吃 2GB 显存。
# 如果你有一张 24GB 显存的显卡，建议设为 8 或 10；如果只有 12GB，建议设为 4。
MAX_JOBS=4

# 1. 定义核心变量矩阵
MODELS=("lenet5" "resnet20" "resnet18")
POISON_RATIOS=(0.1 0.2 0.3 0.4)
DEFENSES=("none" "ours" "krum" "clustering" "median")

TARGETED_ATTACKS=("backdoor" "label_flip")
UNTARGETED_ATTACKS=("random" "scaling")

ROUNDS=50

# 存放所有待执行命令的临时文件
CMD_FILE="tasks_queue.txt"
> $CMD_FILE # 清空旧文件

echo "🚀 正在生成 PPFL-TEE 攻防实验任务队列..."

# ---------------------------------------------------------
# 阶段一：纯净联邦训练 (上限基线 No Attack)
# ---------------------------------------------------------
for model in "${MODELS[@]}"; do
    echo "python main.py --model $model --attack NoAttack --defense none --rounds $ROUNDS > main/results/log_${model}_NoAttack.txt 2>&1" >> $CMD_FILE
done

# ---------------------------------------------------------
# 阶段二：有目标攻击测试 (Targeted Attacks: Backdoor, Label Flip)
# ---------------------------------------------------------
for attack in "${TARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > main/results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

# ---------------------------------------------------------
# 阶段三：无目标攻击测试 (Untargeted Attacks: Random, Scaling)
# ---------------------------------------------------------
for attack in "${UNTARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > main/results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

TOTAL_TASKS=$(wc -l < $CMD_FILE)
echo "✅ 成功生成 $TOTAL_TASKS 个实验任务！"
echo "🔥 开始并行执行，最大并发进程数限制为: $MAX_JOBS"
echo "=========================================="

# 核心并行执行逻辑：使用 xargs 维持最大并发数为 MAX_JOBS
# -P $MAX_JOBS : 保持 MAX_JOBS 个并行进程
# -I {} : 逐行读取命令
cat $CMD_FILE | xargs -P $MAX_JOBS -I {} bash -c '{}'

echo "🎉 所有并行实验执行完毕！"
# rm $CMD_FILE # 运行结束后可选择是否删除任务列表文件