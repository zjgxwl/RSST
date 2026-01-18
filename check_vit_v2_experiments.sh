#!/bin/bash

# ============================================================================
# ViT-Small 优化版实验监控脚本
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 ViT-Small 优化版实验状态监控"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# GPU状态
echo "【GPU状态】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s: %s%% 使用率 | 显存: %s/%s MB\n", $1, $3, $4, $5}'
echo ""

# 进程状态
echo "【实验进程状态】"
PIDS=(1882967 1882968 1882969 1882970)
NAMES=("CIFAR-10-Refill[GPU0]" "CIFAR-10-RSST[GPU0]" "CIFAR-100-Refill[GPU1]" "CIFAR-100-RSST[GPU1]")

for i in {0..3}; do
    if ps -p ${PIDS[$i]} > /dev/null; then
        echo "  ✓ PID ${PIDS[$i]} (${NAMES[$i]}): 运行中"
    else
        echo "  ✗ PID ${PIDS[$i]} (${NAMES[$i]}): 已停止"
    fi
done
echo ""

# 训练进度
echo "【训练进度】"
LOG_FILES=(
    "logs_vit_small_70p_v2/cifar10_refill_70p_v2_0118_2000.log"
    "logs_vit_small_70p_v2/cifar10_rsst_70p_v2_0118_2000.log"
    "logs_vit_small_70p_v2/cifar100_refill_70p_v2_0118_2000.log"
    "logs_vit_small_70p_v2/cifar100_rsst_70p_v2_0118_2000.log"
)
EXP_NAMES=("CIFAR-10 Refill" "CIFAR-10 RSST" "CIFAR-100 Refill" "CIFAR-100 RSST")

for i in {0..3}; do
    echo ""
    echo "  【${EXP_NAMES[$i]}】"
    
    # 检查当前state
    CURRENT_STATE=$(grep -o "pruning state [0-9]*" ${LOG_FILES[$i]} 2>/dev/null | tail -1 | awk '{print $3}')
    if [ -z "$CURRENT_STATE" ]; then
        CURRENT_STATE="初始化中..."
    else
        CURRENT_STATE="State $CURRENT_STATE"
    fi
    echo "    当前阶段: $CURRENT_STATE"
    
    # 最新的验证精度
    LATEST_ACC=$(grep "valid_accuracy" ${LOG_FILES[$i]} 2>/dev/null | tail -1)
    if [ -z "$LATEST_ACC" ]; then
        echo "    最新精度: 训练中，尚未完成第一个epoch..."
    else
        echo "    最新精度: $LATEST_ACC"
    fi
    
    # 检查是否有错误
    ERROR_COUNT=$(grep -i "error\|exception\|traceback" ${LOG_FILES[$i]} 2>/dev/null | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "    ⚠️ 发现 $ERROR_COUNT 个错误，请检查日志！"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "【快捷监控命令】"
echo "  实时日志: tail -f logs_vit_small_70p_v2/cifar10_refill_70p_v2_0118_2000.log"
echo "  查看所有State: grep 'pruning state' logs_vit_small_70p_v2/*.log"
echo "  查看精度变化: grep 'valid_accuracy' logs_vit_small_70p_v2/cifar10_refill_70p_v2_0118_2000.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
