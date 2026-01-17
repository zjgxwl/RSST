#!/bin/bash
# 检查Mamba测试实验的进度和结果

LOG_DIR="logs_mamba_test"

echo "========================================================="
echo "Mamba测试实验 - 进度检查"
echo "========================================================="

# 查找最新的日志文件
REFILL_LOG=$(ls -t $LOG_DIR/mamba_test_refill_*.log 2>/dev/null | head -1)
RSST_LOG=$(ls -t $LOG_DIR/mamba_test_rsst_*.log 2>/dev/null | head -1)

if [ -z "$REFILL_LOG" ] && [ -z "$RSST_LOG" ]; then
    echo "❌ 未找到实验日志文件"
    echo "请先运行: ./run_mamba_test_1state.sh"
    exit 1
fi

echo ""
echo "日志文件："
[ -n "$REFILL_LOG" ] && echo "  - Refill: $REFILL_LOG"
[ -n "$RSST_LOG" ] && echo "  - RSST:   $RSST_LOG"
echo ""

# ==================== 检查进程状态 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 进程状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RUNNING=$(ps aux | grep 'main_imp_fillback.py.*mamba_test' | grep -v grep | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "✓ 实验正在运行中 ($RUNNING 个进程)"
    ps aux | grep 'main_imp_fillback.py.*mamba_test' | grep -v grep | awk '{print "  PID:", $2, "GPU:", $NF}'
else
    echo "✓ 实验已完成或未运行"
fi
echo ""

# ==================== 检查GPU使用 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. GPU使用情况"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | \
    awk -F, '{printf "GPU %s: %s, Memory: %s/%s, Util: %s, Temp: %s\n", $1, $2, $3, $4, $5, $6}'
echo ""

# ==================== Refill实验详情 ====================
if [ -n "$REFILL_LOG" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3. Refill实验详情"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 检查当前进度
    CURRENT_EPOCH=$(grep -oP 'Epoch: \[\K[0-9]+' "$REFILL_LOG" | tail -1)
    TOTAL_EPOCHS=20
    
    if [ -n "$CURRENT_EPOCH" ]; then
        echo "📊 训练进度: Epoch $CURRENT_EPOCH / $TOTAL_EPOCHS"
        
        # 最近的训练信息
        echo ""
        echo "最近训练记录:"
        grep -E "Epoch: \[[0-9]+\].*Accuracy" "$REFILL_LOG" | tail -3
        
        # 耗时信息
        echo ""
        echo "⏱️  耗时统计:"
        
        # State耗时
        STATE_TIME=$(grep -oP 'State [0-9]+ 耗时: \K[0-9.]+' "$REFILL_LOG" | tail -1)
        if [ -n "$STATE_TIME" ]; then
            echo "  - State 0 总耗时: ${STATE_TIME}秒 ($(echo "scale=2; $STATE_TIME/60" | bc)分钟)"
        fi
        
        # Epoch平均耗时
        EPOCH_TIMES=$(grep -oP 'Time \K[0-9.]+' "$REFILL_LOG" | tail -5)
        if [ -n "$EPOCH_TIMES" ]; then
            AVG_TIME=$(echo "$EPOCH_TIMES" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count}')
            echo "  - 最近5个Epoch平均耗时: ${AVG_TIME}秒"
            
            # 估算剩余时间
            if [ -n "$CURRENT_EPOCH" ]; then
                REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
                REMAINING_TIME=$(echo "scale=2; $AVG_TIME * $REMAINING / 60" | bc)
                echo "  - 预计剩余时间: ${REMAINING_TIME}分钟"
            fi
        fi
        
        # 剪枝信息
        echo ""
        echo "✂️  剪枝信息:"
        if grep -q "Pruning completed" "$REFILL_LOG"; then
            echo "  ✓ 剪枝已完成"
            
            # SSM剪枝统计
            echo ""
            echo "  SSM各层保留情况 (前10层):"
            grep 'ssm.out_proj.*kept' "$REFILL_LOG" | head -10 | \
                awk '{print "    " $0}'
            
            # MLP剪枝统计
            echo ""
            echo "  MLP各层保留情况 (前10层):"
            grep 'mlp.*kept' "$REFILL_LOG" | head -10 | \
                awk '{print "    " $0}'
            
            # 总体统计
            echo ""
            echo "  总体参数变化:"
            grep -E "Parameters:.*→" "$REFILL_LOG" | tail -2 | \
                awk '{print "    " $0}'
        else
            echo "  ⏳ 剪枝尚未开始或进行中"
        fi
        
    else
        echo "⏳ 实验刚启动，等待训练开始..."
    fi
    echo ""
fi

# ==================== RSST实验详情 ====================
if [ -n "$RSST_LOG" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4. RSST实验详情"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 检查当前进度
    CURRENT_EPOCH=$(grep -oP 'Epoch: \[\K[0-9]+' "$RSST_LOG" | tail -1)
    TOTAL_EPOCHS=20
    
    if [ -n "$CURRENT_EPOCH" ]; then
        echo "📊 训练进度: Epoch $CURRENT_EPOCH / $TOTAL_EPOCHS"
        
        # 最近的训练信息
        echo ""
        echo "最近训练记录:"
        grep -E "Epoch: \[[0-9]+\].*Accuracy" "$RSST_LOG" | tail -3
        
        # 耗时信息
        echo ""
        echo "⏱️  耗时统计:"
        
        # State耗时
        STATE_TIME=$(grep -oP 'State [0-9]+ 耗时: \K[0-9.]+' "$RSST_LOG" | tail -1)
        if [ -n "$STATE_TIME" ]; then
            echo "  - State 0 总耗时: ${STATE_TIME}秒 ($(echo "scale=2; $STATE_TIME/60" | bc)分钟)"
        fi
        
        # Epoch平均耗时
        EPOCH_TIMES=$(grep -oP 'Time \K[0-9.]+' "$RSST_LOG" | tail -5)
        if [ -n "$EPOCH_TIMES" ]; then
            AVG_TIME=$(echo "$EPOCH_TIMES" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count}')
            echo "  - 最近5个Epoch平均耗时: ${AVG_TIME}秒"
            
            # 估算剩余时间
            if [ -n "$CURRENT_EPOCH" ]; then
                REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
                REMAINING_TIME=$(echo "scale=2; $AVG_TIME * $REMAINING / 60" | bc)
                echo "  - 预计剩余时间: ${REMAINING_TIME}分钟"
            fi
        fi
        
        # RSST特有：正则化信息
        echo ""
        echo "🔧 RSST正则化:"
        REG_LOSS=$(grep -oP 'rsst.*loss.*\K[0-9.]+' "$RSST_LOG" | tail -1)
        if [ -n "$REG_LOSS" ]; then
            echo "  - 当前正则化损失: $REG_LOSS"
        fi
        
    else
        echo "⏳ 实验刚启动，等待训练开始..."
    fi
    echo ""
fi

# ==================== 快速命令 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. 快速查看命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# 实时查看Refill日志"
[ -n "$REFILL_LOG" ] && echo "tail -f $REFILL_LOG"
echo ""
echo "# 实时查看RSST日志"
[ -n "$RSST_LOG" ] && echo "tail -f $RSST_LOG"
echo ""
echo "# 查看最终结果"
echo "grep -E 'best.*SA|Test.*accuracy' $LOG_DIR/mamba_test_*_*.log"
echo ""
echo "# 查看完整剪枝统计"
echo "grep -A 30 'Pruning completed' $LOG_DIR/mamba_test_*_*.log"
echo ""
echo "# 重新运行此检查脚本"
echo "./check_mamba_test.sh"
echo ""
echo "========================================================="
