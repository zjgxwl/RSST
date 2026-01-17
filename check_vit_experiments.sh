#!/bin/bash

################################################################################
# ViT实验监控脚本
################################################################################

# 查找最新的日志文件
TIMESTAMP="0118_0139"

LOGS=(
    "logs_vit_small_70p/cifar10_refill_70p_${TIMESTAMP}.log"
    "logs_vit_small_70p/cifar10_rsst_70p_${TIMESTAMP}.log"
    "logs_vit_small_70p/cifar100_refill_70p_${TIMESTAMP}.log"
    "logs_vit_small_70p/cifar100_rsst_70p_${TIMESTAMP}.log"
)

echo "========================================================================"
echo "ViT实验监控 - $(date +'%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# GPU状态
echo "📊 GPU状态:"
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | while IFS=, read -r idx name util mem_used mem_total temp; do
    echo "GPU ${idx}: 利用率 ${util}%, 显存 ${mem_used}MB/${mem_total}MB, 温度 ${temp}°C"
done
echo ""

# 主进程状态
echo "🔄 主进程状态:"
echo "------------------------------------------------------------------------"
PIDS=(3080681 3080881 3081145 3081617)
NAMES=("CIFAR-10 Refill" "CIFAR-10 RSST" "CIFAR-100 Refill" "CIFAR-100 RSST")

for i in ${!PIDS[@]}; do
    if ps -p ${PIDS[$i]} > /dev/null 2>&1; then
        CPU=$(ps -p ${PIDS[$i]} -o %cpu= | tr -d ' ')
        MEM=$(ps -p ${PIDS[$i]} -o rss= | awk '{print int($1/1024)"MB"}')
        echo "✓ ${NAMES[$i]} (PID ${PIDS[$i]}): CPU ${CPU}%, MEM ${MEM}"
    else
        echo "✗ ${NAMES[$i]} (PID ${PIDS[$i]}): 进程已停止"
    fi
done
echo ""

# 训练进度
echo "========================================================================"
echo "📈 训练进度"
echo "========================================================================"
echo ""

for i in ${!LOGS[@]}; do
    LOG=${LOGS[$i]}
    NAME=${NAMES[$i]}
    
    echo "【${NAME}】"
    echo "------------------------------------------------------------------------"
    
    if [ -f "$LOG" ]; then
        # 当前State
        STATE=$(grep "pruning state" $LOG | tail -1 | awk '{print $NF}')
        if [ ! -z "$STATE" ]; then
            echo "当前State: $STATE"
        fi
        
        # 最近训练
        LAST_EPOCH=$(grep -E "Epoch: \[[0-9]+\]\[[0-9]+/[0-9]+\]" $LOG | tail -1)
        if [ ! -z "$LAST_EPOCH" ]; then
            echo "最近记录: $LAST_EPOCH"
        fi
        
        # 最佳准确率
        BEST=$(grep "best SA=" $LOG | tail -1)
        if [ ! -z "$BEST" ]; then
            echo "  $BEST"
        fi
        
        # 检查错误
        if grep -q "RuntimeError.*device\|Expected all tensors to be on the same device" $LOG; then
            echo "  ❌ 检测到设备错误！"
        else
            echo "  ✓ 无设备错误"
        fi
        
    else
        echo "❌ 日志文件不存在"
    fi
    echo ""
done

echo "========================================================================"
echo "📌 监控命令"
echo "========================================================================"
echo ""
echo "# 实时查看所有日志"
echo "tail -f logs_vit_small_70p/*${TIMESTAMP}.log"
echo ""
echo "# 查看特定实验"
echo "tail -f logs_vit_small_70p/cifar10_refill_70p_${TIMESTAMP}.log"
echo ""
echo "# 重新运行此脚本"
echo "./check_vit_experiments.sh"
echo ""
echo "========================================================================"
