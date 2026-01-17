#!/bin/bash
# Mamba-Small 测试脚本 - 1个State，20 epochs
# 测试耗时和剪枝效果

echo "========================================================="
echo "Mamba-Small 单State测试"
echo "配置: 1 state × 20 epochs, 70%剪枝"
echo "========================================================="

# 配置
ARCH="mamba_small"
DATASET="cifar10"
PRUNE_RATE=0.7
MLP_PRUNE_RATE=0.7
PRUNING_TIMES=1  # 只跑1个state
EPOCHS=20        # 每个state 20个epoch
BATCH_SIZE=128
LR=0.01
SORTING_MODE="global"

# RSST参数
REG_GRANULARITY=1.0
RST_SCHEDULE="exp_custom_exponents"
EXPONENTS=4

# 创建日志目录
LOG_DIR="logs_mamba_test"
mkdir -p $LOG_DIR

# 获取时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

echo ""
echo "实验配置："
echo "  - 模型: ${ARCH}"
echo "  - 数据集: ${DATASET}"
echo "  - 剪枝率: SSM ${PRUNE_RATE}, MLP ${MLP_PRUNE_RATE}"
echo "  - State数: ${PRUNING_TIMES}"
echo "  - Epoch数: ${EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo ""

# ==================== 实验1: Refill ====================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实验1: CIFAR-10 + Refill (GPU 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXP_NAME="mamba_test_refill_${TIMESTAMP}"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "启动时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data datasets/$DATASET \
    --mamba_structured \
    --mamba_prune_target both \
    --rate $PRUNE_RATE \
    --mamba_mlp_prune_ratio $MLP_PRUNE_RATE \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct refill \
    --fillback_rate 0.0 \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID_REFILL=$!
echo "已启动 Refill实验，PID: $PID_REFILL"
echo ""
sleep 3

# ==================== 实验2: RSST ====================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实验2: CIFAR-10 + RSST (GPU 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXP_NAME="mamba_test_rsst_${TIMESTAMP}"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "启动时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data datasets/$DATASET \
    --mamba_structured \
    --mamba_prune_target both \
    --rate $PRUNE_RATE \
    --mamba_mlp_prune_ratio $MLP_PRUNE_RATE \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct rsst \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID_RSST=$!
echo "已启动 RSST实验，PID: $PID_RSST"
echo ""

# ==================== 总结 ====================
echo ""
echo "========================================================="
echo "两个测试实验已启动"
echo "========================================================="
echo ""
echo "进程信息："
echo "  - Refill (GPU 0): PID $PID_REFILL"
echo "  - RSST   (GPU 1): PID $PID_RSST"
echo ""
echo "日志文件："
echo "  - Refill: $LOG_DIR/mamba_test_refill_${TIMESTAMP}.log"
echo "  - RSST:   $LOG_DIR/mamba_test_rsst_${TIMESTAMP}.log"
echo ""
echo "实时监控命令："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# 查看Refill日志（实时）"
echo "tail -f $LOG_DIR/mamba_test_refill_${TIMESTAMP}.log"
echo ""
echo "# 查看RSST日志（实时）"
echo "tail -f $LOG_DIR/mamba_test_rsst_${TIMESTAMP}.log"
echo ""
echo "# 查看所有日志"
echo "tail -f $LOG_DIR/*${TIMESTAMP}.log"
echo ""
echo "# 查看耗时信息（Refill）"
echo "grep -E 'Time|Epoch:|State' $LOG_DIR/mamba_test_refill_${TIMESTAMP}.log | tail -30"
echo ""
echo "# 查看耗时信息（RSST）"
echo "grep -E 'Time|Epoch:|State' $LOG_DIR/mamba_test_rsst_${TIMESTAMP}.log | tail -30"
echo ""
echo "# 查看进程状态"
echo "ps aux | grep 'main_imp_fillback.py.*mamba_test'"
echo ""
echo "# 查看GPU使用"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# 检查是否运行中"
echo "ps -p $PID_REFILL,$PID_RSST"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "预计完成时间："
echo "  - 每个epoch约2-3分钟"
echo "  - 1个state(20 epochs)约40-60分钟"
echo "  - 预计1-1.5小时完成"
echo ""
echo "完成后检查剪枝情况："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "# 查看最终剪枝统计（Refill）"
echo "grep -A 50 'Pruning completed' $LOG_DIR/mamba_test_refill_${TIMESTAMP}.log"
echo ""
echo "# 查看最终剪枝统计（RSST）"
echo "grep -A 50 'Pruning completed' $LOG_DIR/mamba_test_rsst_${TIMESTAMP}.log"
echo ""
echo "# 查看各层保留率（Refill）"
echo "grep 'kept' $LOG_DIR/mamba_test_refill_${TIMESTAMP}.log | head -30"
echo ""
echo "# 查看各层保留率（RSST）"
echo "grep 'kept' $LOG_DIR/mamba_test_rsst_${TIMESTAMP}.log | head -30"
echo ""
echo "========================================================="
echo "启动完成！使用上述命令监控进度"
echo "========================================================="
