#!/bin/bash

# ==========================================
# PPFL-TEE 攻防实验矩阵 (安全并发版)
# ==========================================

MAX_JOBS=6

MODELS=("resnet18")  #"lenet5" "resnet20" "resnet18"
POISON_RATIOS=(0.4) # 0.1 0.2 0.3 0.4
DEFENSES=("layers_proj_detect" "clustering")

TARGETED_ATTACKS=("backdoor") #"backdoor" "label_flip"
UNTARGETED_ATTACKS=() #"random_poison" "gradient_amplify"

ROUNDS=100

mkdir -p ./results

CMD_FILE="tasks_queue.txt"
> $CMD_FILE 

echo "🚀 正在生成 PPFL-TEE 攻防实验任务队列..."

# [修改]：引入 -u 标志 (python -u)，取消标准输出的缓冲，使重定向到文件的日志能够做到实时刷新
for model in "${MODELS[@]}"; do
    echo "python -u main.py --model $model --attack none --defense none --rounds $ROUNDS > ./results/log_${model}_noneAttack.txt 2>&1" >> $CMD_FILE
done

for attack in "${TARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python -u main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > ./results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

for attack in "${UNTARGETED_ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ratio in "${POISON_RATIOS[@]}"; do
            for defense in "${DEFENSES[@]}"; do
                echo "python -u main.py --model $model --attack $attack --poison_ratio $ratio --defense $defense --rounds $ROUNDS > ./results/log_${model}_${attack}_p${ratio}_${defense}.txt 2>&1" >> $CMD_FILE
            done
        done
    done
done

TOTAL_TASKS=$(wc -l < $CMD_FILE)
echo "✅ 成功生成 $TOTAL_TASKS 个实验任务！"
echo "🔥 开始并行执行，最大并发进程数限制为: $MAX_JOBS"
echo "=========================================="
echo "💡 提示：如果想要观察实时的实验进度，请开启一个新的终端，并运行："
echo "    watch -n 2 'tail -n 1 ./results/*.txt'"
echo "=========================================="

cat $CMD_FILE | xargs -P $MAX_JOBS -I {} bash -c '{}'

echo "🎉 所有并行实验执行完毕！"