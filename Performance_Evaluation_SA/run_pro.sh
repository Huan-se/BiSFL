#!/bin/bash
# nohup run_pro.sh > pro_log.txt 2>&1 &
# ==============================================================================
# 联邦学习安全聚合方案性能自动化评估脚本
# 扩展版：支持 dropout 率变化，用于绘制 Fig(a)(b)(c)(d) 四张图
#
# 实验设计：
#   Fig(a): 耗时 vs N        固定 d=2^20，N ∈ {10,20,30,50,75,100}，dropout ∈ {0,10,20,30}%
#   Fig(b): 耗时 vs d        固定 N=100，d ∈ {2^10,2^12,2^15,2^17,2^20}，dropout ∈ {0,10,20,30}%
#   Fig(c): 通信量 vs N      固定 d=2^20，N ∈ {10,20,30,50,75,100}，dropout=0（由公式计算，可选实测）
#   Fig(d): Expansion vs d   固定 N=100，d 连续范围（由公式计算，无需实测）
# ==============================================================================

set -euo pipefail

# ── 基础环境 ──────────────────────────────────────────────────────────────────
ROOT_DIR=$(pwd)
RESULTS_DIR="benchmark_results"
FAILED_LOG="${RESULTS_DIR}/failed_runs.log"
START_TIME_ALL=$(date +%s)

USE_EMOJI=${USE_EMOJI:-true}
MARK_INFO=$([ "$USE_EMOJI" = true ] && echo "🚀" || echo "[INFO]")
MARK_OK=$([ "$USE_EMOJI" = true ] && echo "✅" || echo "[OK]")
MARK_TIMEOUT=$([ "$USE_EMOJI" = true ] && echo "🔴" || echo "[TIMEOUT]")
MARK_FAIL=$([ "$USE_EMOJI" = true ] && echo "❌" || echo "[FAIL]")
MARK_WARN=$([ "$USE_EMOJI" = true ] && echo "⚠️" || echo "[WARN]")

ulimit -n 100000 2>/dev/null || echo "$MARK_WARN ulimit 设置失败"
sudo sysctl -w net.ipv4.tcp_tw_reuse=1      2>/dev/null || true
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535" 2>/dev/null || true

# ── 实验参数定义 ──────────────────────────────────────────────────────────────
SCHEMES=("BiVFL+" "SecAgg" "SecAgg+" "BiVFL")   # 可追加 "SecAgg" "SecAgg+" "BiVFL" "BatchCrypt"

# Dropout 率列表（百分比整数，传给 --drop_rate 时除以100）
DROPOUT_RATES=(0 10 20 30)

# ---- Fig(a): 耗时 vs N，固定 d=2^20 ----
FIG_A_N_LIST=(10 50 100 200 300 500)
FIG_A_D_POW=20          # 固定参数维度指数

# ---- Fig(b): 耗时 vs d，固定 N=100 ----
FIG_B_D_POW_LIST=(10 12 15 17 20)
FIG_B_N=100             # 固定客户端数

# ---- Fig(c): 通信量 vs N，固定 d=2^20, dropout=0 ----
#   与 Fig(a) dropout=0 的数据复用，无需额外实验

CURRENT_PORT=10000

mkdir -p "$RESULTS_DIR"
cat > "$FAILED_LOG" <<EOF
========================================================
 异常与失败实验记录 (Failed Runs)
 开始执行时间: $(date)
========================================================
EOF

# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 根据 (N, d_pow) 自动计算超时秒数
get_timeout() {
    local n=$1 d_pow=$2 scheme=$3
    if [ "$scheme" == "BatchCrypt" ]; then
        echo 604800
        return
    fi
    if   [ "$n" -ge 1000 ] && [ "$d_pow" -ge 20 ]; then echo 10800
    elif [ "$n" -ge 1000 ];                          then echo 7200
    elif [ "$n" -ge 100  ] && [ "$d_pow" -ge 20 ]; then echo 7200
    elif [ "$n" -ge 100  ];                          then echo 7200
    else                                                  echo 7200
    fi
}

# 根据 (N, d_pow) 自动计算实验间等待秒数
get_wait() {
    local n=$1 d_pow=$2
    if   [ "$n" -ge 1000 ] && [ "$d_pow" -ge 20 ]; then echo 120
    elif [ "$n" -ge 1000 ];                          then echo 60
    elif [ "$n" -ge 100  ] && [ "$d_pow" -ge 20 ]; then echo 15
    elif [ "$n" -ge 100  ];                          then echo 10
    else                                                  echo 3
    fi
}

# 执行单次实验并记录结果
run_one() {
    local scheme=$1 n=$2 d_pow=$3 drop_rate=$4
    local param_size=$((2 ** d_pow))
    local drop_float=$(echo "scale=2; $drop_rate / 100" | bc)
    local timeout_val; timeout_val=$(get_timeout "$n" "$d_pow" "$scheme")
    local wait_time;   wait_time=$(get_wait "$n" "$d_pow")

    # 结果文件按 scheme / experiment_type 分层存储
    local out_dir="${RESULTS_DIR}/${scheme}"
    mkdir -p "$out_dir"
    local log_file="${out_dir}/n${n}_d${d_pow}_drop${drop_rate}.log"

    echo "[*] scheme=$scheme | N=$n | d=2^$d_pow | dropout=${drop_rate}% | port=$CURRENT_PORT"

    cat > "$log_file" <<HDR
========================================================
 Scheme   : $scheme
 N        : $n
 d        : 2^$d_pow  ($param_size)
 dropout  : ${drop_rate}%
 Port     : $CURRENT_PORT
 Started  : $(date)
========================================================
HDR

    set +e
    (
        cd "${ROOT_DIR}/${scheme}/main" 2>/dev/null || {
            echo "[DIR_NOT_FOUND] ${ROOT_DIR}/${scheme}/main 不存在"
            exit 127
        }
        timeout "$timeout_val" python server_simulator.py \
            --num_clients  "$n"          \
            --param_size   "$param_size" \
            --port         "$CURRENT_PORT" \
            --drop_rate    "$drop_float"
    ) >> "$log_file" 2>&1
    local exit_code=$?
    set -e

    if   [ $exit_code -eq 127 ]; then
        echo "  $MARK_FAIL [DIR_NOT_FOUND]"
        echo "[DIR_NOT_FOUND] $scheme N=$n d=2^$d_pow drop=${drop_rate}%" >> "$FAILED_LOG"
        # 目录不存在则不消耗端口，直接返回
        return
    elif [ $exit_code -eq 124 ]; then
        echo "  $MARK_TIMEOUT [TIMEOUT after ${timeout_val}s]"
        echo "[TIMEOUT] $scheme N=$n d=2^$d_pow drop=${drop_rate}% port=$CURRENT_PORT" >> "$FAILED_LOG"
    elif [ $exit_code -ne 0 ]; then
        echo "  $MARK_FAIL [FAILED exit=$exit_code]"
        echo "[FAILED=$exit_code] $scheme N=$n d=2^$d_pow drop=${drop_rate}% port=$CURRENT_PORT" >> "$FAILED_LOG"
    else
        echo "  $MARK_OK [SUCCESS]"
    fi

    CURRENT_PORT=$((CURRENT_PORT + 1))
    sleep "$wait_time"
}

# ══════════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════════
for scheme in "${SCHEMES[@]}"; do

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " $MARK_INFO Scheme: $scheme"
    echo "════════════════════════════════════════════════════════"

    # ------------------------------------------------------------------
    # Experiment Group A: 耗时 vs N  (固定 d=2^20，遍历所有 dropout 率)
    # 同时覆盖 Fig(c) 的通信量数据（dropout=0 时）
    # ------------------------------------------------------------------
    echo ""
    echo "▶ [Group A] 耗时 vs N  (固定 d=2^${FIG_A_D_POW}，各 dropout 率)"
    for drop in "${DROPOUT_RATES[@]}"; do
        for n in "${FIG_A_N_LIST[@]}"; do
            run_one "$scheme" "$n" "$FIG_A_D_POW" "$drop"
        done
    done

    # ------------------------------------------------------------------
    # Experiment Group B: 耗时 vs d  (固定 N=100，遍历所有 dropout 率)
    # ------------------------------------------------------------------
    echo ""
    echo "▶ [Group B] 耗时 vs d  (固定 N=${FIG_B_N}，各 dropout 率)"
    for drop in "${DROPOUT_RATES[@]}"; do
        for d_pow in "${FIG_B_D_POW_LIST[@]}"; do
            # 跳过已在 Group A 中测过的 (N=100, d=2^20, drop=*) 组合以节省时间
            if [ "$FIG_B_N" -eq 100 ] && [ "$d_pow" -eq "$FIG_A_D_POW" ]; then
                echo "  [SKIP] N=${FIG_B_N} d=2^${d_pow} drop=${drop}% (已在 Group A 覆盖)"
                continue
            fi
            run_one "$scheme" "$FIG_B_N" "$d_pow" "$drop"
        done
    done

done

# ── 全局耗时统计 ──────────────────────────────────────────────────────────────
END_TIME_ALL=$(date +%s)
ELAPSED=$(( END_TIME_ALL - START_TIME_ALL ))
TOTAL_TIME_STR="$(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"

echo ""
echo "════════════════════════════════════════════════════════"
echo "🎉 所有实验执行完毕！"
echo "⏱️  总物理耗时 : $TOTAL_TIME_STR"
echo "📊 结果目录   : $RESULTS_DIR/"
echo "⚠️  失败汇总   : $FAILED_LOG"
echo "════════════════════════════════════════════════════════"

cat >> "$FAILED_LOG" <<EOF

========================================================
 测试结束时间: $(date) | 总耗时: $TOTAL_TIME_STR
========================================================
EOF