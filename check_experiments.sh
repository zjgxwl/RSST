#!/bin/bash
# 快速查看ViT-Small实验进展的脚本

LOG_DIR="/workspace/ycx/RSST/RSST-master/logs_vit_small_pretrained"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 ViT-Small 实验快速查看工具"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查进程状态
echo "【进程状态】"
RUNNING=$(ps aux | grep "main_imp_fillback.py" | grep -v grep | wc -l)
echo "运行中的实验: $RUNNING 个"
echo ""

# 显示每个实验的最新状态
for exp in "cifar10_refill" "cifar10_rsst" "cifar100_refill" "cifar100_rsst"; do
    LOG_FILE="${LOG_DIR}/${exp}_0114_2335.log"
    
    if [ -f "$LOG_FILE" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "【${exp}】"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 最新epoch
        EPOCH=$(grep "Epoch:" "$LOG_FILE" | tail -1)
        echo "最新训练: $EPOCH"
        
        # 最近3个验证准确率
        echo ""
        echo "最近验证准确率:"
        grep "valid_accuracy" "$LOG_FILE" | tail -3 | while read line; do
            echo "  $line"
        done
        
        # 稀疏度（如果有）
        SPARSITY=$(grep "Overall sparsity" "$LOG_FILE" | tail -1)
        if [ ! -z "$SPARSITY" ]; then
            echo ""
            echo "稀疏度: $SPARSITY"
        fi
        
        echo ""
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 使用说明:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "查看完整日志:"
echo "  cat $LOG_DIR/cifar10_refill_0114_2335.log"
echo ""
echo "实时跟踪日志:"
echo "  tail -f $LOG_DIR/cifar10_refill_0114_2335.log"
echo ""
echo "查看所有验证准确率:"
echo "  grep 'valid_accuracy' $LOG_DIR/cifar10_refill_0114_2335.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
