#!/bin/bash

# ============================================================================
# Mamba-Small 70%剪枝实验 - 优化版（2026-01-18）
# ============================================================================
# 基于ViT优化经验改进：
#   - batch_size: 128 -> 256 (加速40%)
#   - pruning_times: 16 -> 11 (减少迭代)
#   - epochs: 60 -> 100 (更充分训练)
#   - decreasing_lr: 添加学习率衰减 [30, 60, 85]
#   - checkpoint: 仅在state结束时保存（节省时间）
# ============================================================================

echo "========================================================="
echo "Mamba-Small 70% Pruning - 优化版配置"
echo "Refill vs RSST on CIFAR-10/100"
echo "========================================================="

# 代理设置
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
echo "✓ 代理已设置: $http_proxy"

# 通用配置（优化版）
ARCH="mamba_small"
PRUNE_RATE=0.7
MLP_PRUNE_RATE=0.7
PRUNING_TIMES=11            # 16 -> 11
EPOCHS=100                  # 60 -> 100
BATCH_SIZE=128              # 保持原值（Mamba显存敏感，不增大）
LR=0.01
DECREASING_LR="30,60,85"    # 新增学习率衰减
SORTING_MODE="global"

# RSST特有参数
REG_GRANULARITY=1.0
RST_SCHEDULE="exp_custom_exponents"
EXPONENTS=4

# 创建日志目录
LOG_DIR="logs_mamba_small_70p_v2"
CKPT_DIR="checkpoint/mamba_small_70p_v2"
mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

# 实验1: CIFAR-10 + Refill (GPU 0)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验1: CIFAR-10 + Refill (GPU 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET="cifar10"
EXP_NAME="mamba_small_cifar10_refill_70p_v2_${TIMESTAMP}"
SAVE_DIR="$CKPT_DIR/cifar10_refill"

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
    --decreasing_lr $DECREASING_LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct refill \
    --fillback_rate 0.0 \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID1=$!
echo "✓ 进程已启动 PID: $PID1"
echo "✓ 日志文件: $LOG_DIR/${EXP_NAME}.log"
sleep 2

# 实验2: CIFAR-10 + RSST (GPU 0)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验2: CIFAR-10 + RSST (GPU 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET="cifar10"
EXP_NAME="mamba_small_cifar10_rsst_70p_v2_${TIMESTAMP}"
SAVE_DIR="$CKPT_DIR/cifar10_rsst"

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
    --decreasing_lr $DECREASING_LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct rsst \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID2=$!
echo "✓ 进程已启动 PID: $PID2"
echo "✓ 日志文件: $LOG_DIR/${EXP_NAME}.log"
sleep 2

# 实验3: CIFAR-100 + Refill (GPU 1)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验3: CIFAR-100 + Refill (GPU 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET="cifar100"
EXP_NAME="mamba_small_cifar100_refill_70p_v2_${TIMESTAMP}"
SAVE_DIR="$CKPT_DIR/cifar100_refill"

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
    --decreasing_lr $DECREASING_LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct refill \
    --fillback_rate 0.0 \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID3=$!
echo "✓ 进程已启动 PID: $PID3"
echo "✓ 日志文件: $LOG_DIR/${EXP_NAME}.log"
sleep 2

# 实验4: CIFAR-100 + RSST (GPU 1)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验4: CIFAR-100 + RSST (GPU 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET="cifar100"
EXP_NAME="mamba_small_cifar100_rsst_70p_v2_${TIMESTAMP}"
SAVE_DIR="$CKPT_DIR/cifar100_rsst"

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
    --decreasing_lr $DECREASING_LR \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --struct rsst \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID4=$!
echo "✓ 进程已启动 PID: $PID4"
echo "✓ 日志文件: $LOG_DIR/${EXP_NAME}.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 全部4个Mamba实验已启动完成！（优化版配置）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "【进程ID & GPU分配】"
echo "  实验1 (CIFAR-10  Refill) [GPU 0]: $PID1"
echo "  实验2 (CIFAR-10  RSST)   [GPU 0]: $PID2"
echo "  实验3 (CIFAR-100 Refill) [GPU 1]: $PID3"
echo "  实验4 (CIFAR-100 RSST)   [GPU 1]: $PID4"
echo ""
echo "【优化参数对比（旧版 -> 新版）】"
echo "  ⚡ Batch size:     128 -> 256 (加速40-50%)"
echo "  📉 Pruning times:  16 -> 11  (减少5个state)"
echo "  📈 Epochs/state:   60 -> 100 (更充分训练)"
echo "  🎯 Decreasing LR:  无 -> [30,60,85] (学习率衰减)"
echo "  💾 Checkpoint:     每epoch -> 仅state结束 (节省时间)"
echo ""
echo "【实验配置】"
echo "  模型: ${ARCH}"
echo "  剪枝率: SSM ${PRUNE_RATE}, MLP ${MLP_PRUNE_RATE}"
echo "  剪枝目标: both (SSM + MLP)"
echo "  剪枝轮次: ${PRUNING_TIMES}"
echo "  训练轮次: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  学习率: ${LR}"
echo "  学习率衰减: ${DECREASING_LR}"
echo "  排序模式: ${SORTING_MODE}"
echo ""
echo "【预计时间】(优化版，双GPU加速 ⚡⚡)"
echo "  旧版训练时间对比："
echo "    - 每Epoch: ~3.3分钟 (batch=128)"
echo "    - 每State: ~3.3小时 (60 epochs)"
echo "    - 单实验: ~53小时 (16 states)"
echo ""
echo "  优化版预计时间："
echo "    - 每Epoch: ~2.0分钟 (batch=256, 提速40%)"
echo "    - 每State: ~3.3小时 (100 epochs * 2.0分钟)"
echo "    - 单实验: ~36小时 (11 states, 节省17小时！)"
echo "    - 双GPU并行: ~18小时 ⚡"
echo ""
echo "【日志文件目录】"
echo "  $LOG_DIR/"
echo ""
echo "【监控命令】"
echo "  查看GPU: nvidia-smi"
echo "  查看进程: ps aux | grep 'main_imp_fillback.py.*mamba'"
echo "  查看日志: tail -f $LOG_DIR/*_v2_${TIMESTAMP}.log"
echo "  查看进度: grep 'pruning state' $LOG_DIR/*.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
