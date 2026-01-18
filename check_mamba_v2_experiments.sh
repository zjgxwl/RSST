#!/bin/bash

# ============================================================================
# Mamba-Small 优化版实验监控脚本
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Mamba-Small 优化版实验状态监控"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# GPU状态
echo "【GPU状态】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s: %s%% 使用率 | 显存: %s/%s MB\n", $1, $3, $4, $5}'
echo ""

# 进程状态
echo "【实验进程状态】"
ps aux | grep "main_imp_fillback.py.*mamba" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    
    # 提取关键信息
    if echo "$CMD" | grep -q "cifar10.*refill"; then
        NAME="CIFAR-10-Refill"
    elif echo "$CMD" | grep -q "cifar10.*rsst"; then
        NAME="CIFAR-10-RSST"
    elif echo "$CMD" | grep -q "cifar100.*refill"; then
        NAME="CIFAR-100-Refill"
    elif echo "$CMD" | grep -q "cifar100.*rsst"; then
        NAME="CIFAR-100-RSST"
    else
        NAME="未知实验"
    fi
    
    GPU=$(echo "$CMD" | grep -oP "CUDA_VISIBLE_DEVICES=\K[0-9]+" || echo "?")
    echo "  ✓ PID $PID ($NAME) [GPU $GPU]: 运行中"
done

RUNNING_COUNT=$(ps aux | grep "main_imp_fillback.py.*mamba" | grep -v grep | wc -l)
if [ $RUNNING_COUNT -eq 0 ]; then
    echo "  ℹ️ 当前没有Mamba实验在运行"
fi
echo ""

# 训练进度
echo "【训练进度】"
LOG_DIR="logs_mamba_small_70p_v2"

if [ -d "$LOG_DIR" ]; then
    for LOG_FILE in $(ls -t $LOG_DIR/*.log 2>/dev/null | head -4); do
        BASENAME=$(basename $LOG_FILE .log)
        
        echo ""
        echo "  【${BASENAME}】"
        
        # 检查当前state
        CURRENT_STATE=$(grep -o "pruning state [0-9]*" $LOG_FILE 2>/dev/null | tail -1 | awk '{print $3}')
        if [ -z "$CURRENT_STATE" ]; then
            CURRENT_STATE="初始化中..."
        else
            CURRENT_STATE="State $CURRENT_STATE"
        fi
        echo "    当前阶段: $CURRENT_STATE"
        
        # 最新的验证精度
        LATEST_ACC=$(grep "valid_accuracy" $LOG_FILE 2>/dev/null | tail -1)
        if [ -z "$LATEST_ACC" ]; then
            echo "    最新精度: 训练中，尚未完成第一个epoch..."
        else
            echo "    最新精度: $LATEST_ACC"
        fi
        
        # 检查是否有错误
        ERROR_COUNT=$(grep -i "error\|exception\|traceback" $LOG_FILE 2>/dev/null | grep -v "no_weight_decay" | wc -l)
        if [ $ERROR_COUNT -gt 0 ]; then
            echo "    ⚠️ 发现 $ERROR_COUNT 个错误，请检查日志！"
            echo "    最近错误："
            grep -i "error\|exception" $LOG_FILE 2>/dev/null | grep -v "no_weight_decay" | tail -2 | sed 's/^/      /'
        fi
    done
else
    echo "  ℹ️ 日志目录不存在: $LOG_DIR"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【快捷监控命令】"
echo "  实时日志: tail -f $LOG_DIR/*.log"
echo "  查看所有State: grep 'pruning state' $LOG_DIR/*.log"
echo "  查看精度变化: grep 'valid_accuracy' $LOG_DIR/*.log | tail -20"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
