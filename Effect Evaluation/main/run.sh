#!/bin/bash

# 定义我们想要测试的变量
MODELS=("resnet20" "resnet18")
POISON_RATIOS=(0.1 0.2 0.3 0.4)
DEFENSES=("none" "ours" "krum" "median")

# 1. 测试 CIFAR-10 上 Backdoor 攻击对 ASR 的影响 (表格数据)
for model in "${MODELS[@]}"; do
    for ratio in "${POISON_RATIOS[@]}"; do
        for defense in "${DEFENSES[@]}"; do
            echo "Running: Model=$model, Ratio=$ratio, Defense=$defense"
            python main.py \
                --dataset cifar10 \
                --model $model \
                --attack backdoor \
                --poison_ratio $ratio \
                --defense $defense \
                --rounds 50
        done
    done
done