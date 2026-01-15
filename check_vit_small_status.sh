#!/bin/bash

# 快速查看ViT-Small实验状态脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ViT-Small 实验状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查进程
echo "【运行中的实验】"
echo ""
PROCESSES=$(ps aux | grep "vit_small.*main_imp_fillback" | grep -v grep)

if [ -z "$PROCESSES" ]; then
    echo "❌ 没有正在运行的实验"
else
    # 解析进程信息
    echo "$PROCESSES" | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CPU=$(echo $line | awk '{print $3}')
        MEM=$(echo $line | awk '{print $4}')
        TIME=$(echo $line | awk '{print $10}')
        CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
        
        # 提取数据集和方法
        if echo "$CMD" | grep -q "cifar10"; then
            DATASET="CIFAR-10"
        else
            DATASET="CIFAR-100"
        fi
        
        if echo "$CMD" | grep -q "refill"; then
            METHOD="Refill"
        else
            METHOD="RSST"
        fi
        
        echo "  $DATASET + $METHOD"
        echo "    PID: $PID | CPU: ${CPU}% | MEM: ${MEM}% | 运行时间: $TIME"
        echo ""
    done
    
    TOTAL_PROCS=$(echo "$PROCESSES" | wc -l)
    echo "  总实验数: $TOTAL_PROCS"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# GPU使用情况
echo "【GPU使用情况】"
echo ""
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r mem_used mem_total util temp; do
        echo "  显存: $mem_used MB / $mem_total MB"
        echo "  GPU利用率: $util%"
        echo "  温度: ${temp}°C"
    done
else
    echo "  ❌ nvidia-smi 不可用"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 最新日志
echo "【最新日志】"
echo ""
if [ -d "logs_vit_small" ]; then
    LOG_FILES=$(ls -t logs_vit_small/*.log 2>/dev/null | head -4)
    if [ -n "$LOG_FILES" ]; then
        for log in $LOG_FILES; do
            BASENAME=$(basename $log)
            SIZE=$(du -h $log | awk '{print $1}')
            LINES=$(wc -l < $log)
            LAST_MOD=$(stat -c %y $log 2>/dev/null || stat -f "%Sm" $log 2>/dev/null)
            
            echo "  $BASENAME ($SIZE, $LINES lines)"
            
            # 尝试提取训练进度
            if grep -q "Epoch:" $log; then
                LAST_EPOCH=$(grep "Epoch:" $log | tail -1)
                echo "    最新: $LAST_EPOCH"
            fi
            echo ""
        done
    else
        echo "  ❌ 没有找到日志文件"
    fi
else
    echo "  ❌ logs_vit_small 目录不存在"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 快捷命令提示
echo "【快捷命令】"
echo ""
echo "  查看实时日志:"
echo "    tail -f logs_vit_small/cifar10_refill_0114_2219.log"
echo ""
echo "  查看所有实验:"
echo "    ./manage_vit_small_experiments.sh"
echo ""
echo "  WandB项目:"
echo "    https://wandb.ai/ycx/RSST"
echo ""
