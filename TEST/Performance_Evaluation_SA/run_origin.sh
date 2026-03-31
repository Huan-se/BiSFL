#!/bin/bash

# ==============================================================================
# 联邦学习安全聚合方案性能自动化评估脚本 (串行精准测时 + BatchCrypt 豁免版)
# ==============================================================================
# 1. 开启严格模式：未定义变量报错，管道中途失败报错，遇到错误立即退出
set -euo pipefail

# 2. 基础环境与目录准备
ROOT_DIR=$(pwd)
RESULTS_DIR="benchmark_results_origin"
FAILED_LOG="${RESULTS_DIR}/failed_runs.log"

# 全局耗时统计起点
START_TIME_ALL=$(date +%s)

# 终端兼容性处理 (Emoji 降级适配)
USE_EMOJI=${USE_EMOJI:-true}
MARK_INFO=$([ "$USE_EMOJI" = true ] && echo "🚀" || echo "[INFO]")
MARK_OK=$([ "$USE_EMOJI" = true ] && echo "✅" || echo "[OK]")
MARK_TIMEOUT=$([ "$USE_EMOJI" = true ] && echo "🔴" || echo "[TIMEOUT]")
MARK_FAIL=$([ "$USE_EMOJI" = true ] && echo "❌" || echo "[FAIL]")
MARK_WARN=$([ "$USE_EMOJI" = true ] && echo "⚠️" || echo "[WARN]")

# 尝试突破系统限制 (失败则静默跳过)
ulimit -n 100000 2>/dev/null || echo "$MARK_WARN ulimit 设置失败，若遇文件描述符耗尽请使用 root 权限执行"
sudo sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null || true
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535" 2>/dev/null || true

# 3. 定义实验参数
SCHEMES=("BiVFL" ) #"SecAgg" "SecAgg+" "BiVFL" 
CLIENT_COUNTS=(10 50 100)
PARAM_POWERS=(10 15 20)
CURRENT_PORT=10000

# 初始化日志目录
mkdir -p "$RESULTS_DIR"
echo "========================================================" > "$FAILED_LOG"
echo " 异常与失败实验记录 (Failed Runs)" >> "$FAILED_LOG"
echo " 开始执行时间: $(date)" >> "$FAILED_LOG"
echo "========================================================" >> "$FAILED_LOG"

echo "$MARK_INFO 开始执行批量性能测试 (串行精准模式)..."
echo "📂 结果将隔离保存在 ./$RESULTS_DIR/<Scheme>/ 目录下"
echo "--------------------------------------------------------"

# 4. 嵌套循环执行实验
for scheme in "${SCHEMES[@]}"; do
    mkdir -p "${RESULTS_DIR}/${scheme}"
    
    for c in "${CLIENT_COUNTS[@]}"; do
        LOG_FILE="${ROOT_DIR}/${RESULTS_DIR}/${scheme}/${scheme}_clients_${c}.log"
        
        echo "========================================================" > "$LOG_FILE"
        echo " Scheme: $scheme | Total Clients: $c | Started: $(date)" >> "$LOG_FILE"
        echo "========================================================" >> "$LOG_FILE"

        for p_pow in "${PARAM_POWERS[@]}"; do
            param_size=$((2 ** p_pow))
            
            # --- 独立计算操作系统资源回收的等待时间 (WAIT_TIME) ---
            if [ "$c" -ge 10000 ] && [ "$p_pow" -ge 20 ]; then
                WAIT_TIME=120
            elif [ "$c" -ge 10000 ]; then
                WAIT_TIME=60
            elif [ "$c" -ge 1000 ] && [ "$p_pow" -ge 20 ]; then
                WAIT_TIME=60
            elif [ "$c" -ge 1000 ]; then
                WAIT_TIME=30
            elif [ "$c" -ge 100 ]; then
                WAIT_TIME=10
            else
                WAIT_TIME=3
            fi

            # --- 独立计算程序的超时限制 (TIMEOUT_VAL) ---
            if [ "$scheme" == "BatchCrypt" ]; then
                # BatchCrypt 同态加密极其耗时，强制赋予 7 天 (604800秒) 的超长存活期以实现豁免
                TIMEOUT_VAL=604800
                TIMEOUT_DISPLAY="豁免限制 (7天)"
            else
                if [ "$c" -ge 10000 ] && [ "$p_pow" -ge 20 ]; then
                    TIMEOUT_VAL=10800  # 3小时
                elif [ "$c" -ge 10000 ]; then
                    TIMEOUT_VAL=7200   # 2小时
                elif [ "$c" -ge 1000 ] && [ "$p_pow" -ge 20 ]; then
                    TIMEOUT_VAL=7200   # 1小时
                elif [ "$c" -ge 1000 ]; then
                    TIMEOUT_VAL=7200
                elif [ "$c" -ge 100 ]; then
                    TIMEOUT_VAL=7200   # 30分钟
                else
                    TIMEOUT_VAL=7200    # 10分钟
                fi
                TIMEOUT_DISPLAY="${TIMEOUT_VAL}s"
            fi
            
            echo "[*] 运行中 -> 方案: $scheme | 客户端: $c | 参数: 2^$p_pow | 端口: $CURRENT_PORT"
            
            echo "" >> "$LOG_FILE"
            echo "--------------------------------------------------------" >> "$LOG_FILE"
            echo " 测试配置 -> Params: 2^$p_pow ($param_size) | Port: $CURRENT_PORT | Timeout: $TIMEOUT_DISPLAY " >> "$LOG_FILE"
            echo "--------------------------------------------------------" >> "$LOG_FILE"
            
            # 临时关闭 exit on error 以捕获 Python 或 cd 的报错
            set +e 
            
            # 子 Shell 绝对路径隔离与执行
            (
                cd "${ROOT_DIR}/${scheme}/main" || {
                    echo "[DIR_NOT_FOUND] 目录 ${ROOT_DIR}/${scheme}/main 不存在"
                    exit 127
                }
                timeout "$TIMEOUT_VAL" python server_simulator.py \
                    --num_clients "$c" \
                    --param_size "$param_size" \
                    --port "$CURRENT_PORT"
            ) >> "$LOG_FILE" 2>&1
            EXIT_CODE=$?
            
            # 恢复严格模式
            set -e 
            
            # --- 异常分类与记录 ---
            if [ $EXIT_CODE -eq 127 ]; then
                ERR_MSG="$MARK_FAIL [DIR_NOT_FOUND] 方案 $scheme 目录不存在，跳过当前实验 (端口未消耗)"
                echo -e "  $ERR_MSG"
                echo "$ERR_MSG" >> "$FAILED_LOG"
                echo "$ERR_MSG" >> "$LOG_FILE"
                continue 
            elif [ $EXIT_CODE -eq 124 ]; then
                ERR_MSG="$MARK_TIMEOUT [TIMEOUT] 方案 $scheme | 客户端 $c | 参数 2^$p_pow | 端口 $CURRENT_PORT | 超过 ${TIMEOUT_DISPLAY} 被强杀"
                echo -e "  $ERR_MSG"
                echo "$ERR_MSG" >> "$FAILED_LOG"
                echo "$ERR_MSG" >> "$LOG_FILE"
            elif [ $EXIT_CODE -ne 0 ]; then
                ERR_MSG="$MARK_FAIL [FAILED] 方案 $scheme | 客户端 $c | 参数 2^$p_pow | 端口 $CURRENT_PORT | 崩溃，退出码 $EXIT_CODE"
                echo -e "  $ERR_MSG"
                echo "$ERR_MSG" >> "$FAILED_LOG"
                echo "$ERR_MSG" >> "$LOG_FILE"
            else
                echo -e "  $MARK_OK [SUCCESS] 正常完成"
            fi
            
            # 无论成功或失败，消耗了端口就递增
            CURRENT_PORT=$((CURRENT_PORT + 1))
            
            # 严格休眠，等待操作系统回收内存和 Socket
            sleep "$WAIT_TIME"
        done
        echo "  - [$scheme] $c 客户端规模测试落盘完毕。"
    done
done

# 5. 全局耗时统计与结算
END_TIME_ALL=$(date +%s)
ELAPSED=$(( END_TIME_ALL - START_TIME_ALL ))
TOTAL_TIME_STR="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"

echo "--------------------------------------------------------"
echo "🎉 所有实验执行完毕！"
echo "⏱️  总物理耗时: $TOTAL_TIME_STR"
echo "📊 成功日志目录: $RESULTS_DIR/"
echo "⚠️  失败与异常汇总: $FAILED_LOG"

echo "" >> "$FAILED_LOG"
echo "========================================================" >> "$FAILED_LOG"
echo " 测试结束时间: $(date) | 总耗时: $TOTAL_TIME_STR" >> "$FAILED_LOG"
echo "========================================================" >> "$FAILED_LOG"