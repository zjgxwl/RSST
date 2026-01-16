#!/bin/bash

# ============================================================================
# ViT-Small 70%剪枝实验脚本 - Refill + RSST 完整对比
# ============================================================================

# 代理设置
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
echo "✓ 代理已设置: $http_proxy"

# 创建日志目录
mkdir -p logs_vit_small_70p
mkdir -p checkpoint/vit_small_70p

# 时间戳
TIMESTAMP=$(date +%m%d_%H%M)

# ============================================================================
# 通用参数（Refill和RSST完全一致）
# ============================================================================

ARCH="vit_small"
DATASET_CIFAR10="cifar10"
DATASET_CIFAR100="cifar100"
DATA_PATH_10="data/cifar10"
DATA_PATH_100="data/cifar100"

# 模型配置
PRETRAINED="--vit_pretrained"
VIT_STRUCTURED="--vit_structured"
PRUNE_TARGET="--vit_prune_target both"
INIT_FILE="init_model/vit_small_cifar10_pretrained_init.pth.tar"

# 剪枝配置（70%剪枝率）
CRITERIA="--criteria magnitude"
PRUNE_RATE="--rate 0.7"
MLP_RATE="--mlp_prune_ratio 0.7"

# 迭代策略
PRUNING_TIMES="--pruning_times 16"     # 16次迭代
EPOCHS="--epochs 60"                   # 60个epoch/iteration
BATCH_SIZE="--batch_size 128"

# 排序策略
SORTING_MODE="--sorting_mode global"   # 全局混合排序

# 学习率
LR="--lr 0.01"

# ============================================================================
# RSST方法专用参数
# ============================================================================

REG_GRANULARITY="--reg_granularity_prune 1.0"  # 正则化强度1.0
RST_SCHEDULE="--RST_schedule exp_custom_exponents"
EXPONENTS="--exponents 4"                      # 曲率4

# ============================================================================
# 实验1: CIFAR-10 + Refill (70%)
# ============================================================================

SAVE_DIR_1="checkpoint/vit_small_70p/cifar10_refill"
EXP_NAME_1="cifar10_refill_70p_${TIMESTAMP}"
LOG_FILE_1="logs_vit_small_70p/${EXP_NAME_1}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验1: CIFAR-10 + Refill (70%剪枝)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR10 \
    --data $DATA_PATH_10 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    --fillback_rate 0.0 \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR_1 \
    --exp_name $EXP_NAME_1 \
    > $LOG_FILE_1 2>&1 &

PID_1=$!
echo "✓ 进程已启动 (PID: $PID_1)"
echo "✓ 日志文件: $LOG_FILE_1"
echo ""

# ============================================================================
# 实验2: CIFAR-10 + RSST (70%)
# ============================================================================

SAVE_DIR_2="checkpoint/vit_small_70p/cifar10_rsst"
EXP_NAME_2="cifar10_rsst_70p_${TIMESTAMP}"
LOG_FILE_2="logs_vit_small_70p/${EXP_NAME_2}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验2: CIFAR-10 + RSST (70%剪枝)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR10 \
    --data $DATA_PATH_10 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR_2 \
    --exp_name $EXP_NAME_2 \
    > $LOG_FILE_2 2>&1 &

PID_2=$!
echo "✓ 进程已启动 (PID: $PID_2)"
echo "✓ 日志文件: $LOG_FILE_2"
echo ""

# ============================================================================
# 实验3: CIFAR-100 + Refill (70%)
# ============================================================================

SAVE_DIR_3="checkpoint/vit_small_70p/cifar100_refill"
EXP_NAME_3="cifar100_refill_70p_${TIMESTAMP}"
LOG_FILE_3="logs_vit_small_70p/${EXP_NAME_3}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验3: CIFAR-100 + Refill (70%剪枝)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR100 \
    --data $DATA_PATH_100 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    --fillback_rate 0.0 \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR_3 \
    --exp_name $EXP_NAME_3 \
    > $LOG_FILE_3 2>&1 &

PID_3=$!
echo "✓ 进程已启动 (PID: $PID_3)"
echo "✓ 日志文件: $LOG_FILE_3"
echo ""

# ============================================================================
# 实验4: CIFAR-100 + RSST (70%)
# ============================================================================

SAVE_DIR_4="checkpoint/vit_small_70p/cifar100_rsst"
EXP_NAME_4="cifar100_rsst_70p_${TIMESTAMP}"
LOG_FILE_4="logs_vit_small_70p/${EXP_NAME_4}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验4: CIFAR-100 + RSST (70%剪枝)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR100 \
    --data $DATA_PATH_100 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR_4 \
    --exp_name $EXP_NAME_4 \
    > $LOG_FILE_4 2>&1 &

PID_4=$!
echo "✓ 进程已启动 (PID: $PID_4)"
echo "✓ 日志文件: $LOG_FILE_4"
echo ""

# ============================================================================
# 总结信息
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 全部4个实验已启动完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "【进程ID】"
echo "  实验1 (CIFAR-10  Refill): $PID_1"
echo "  实验2 (CIFAR-10  RSST):   $PID_2"
echo "  实验3 (CIFAR-100 Refill): $PID_3"
echo "  实验4 (CIFAR-100 RSST):   $PID_4"
echo ""
echo "【日志文件】"
echo "  实验1: $LOG_FILE_1"
echo "  实验2: $LOG_FILE_2"
echo "  实验3: $LOG_FILE_3"
echo "  实验4: $LOG_FILE_4"
echo ""
echo "【通用参数（Refill & RSST完全一致）】"
echo "  剪枝率: 70% (head & mlp)"
echo "  迭代次数: 16次"
echo "  每次Epoch: 60个"
echo "  排序模式: global"
echo "  学习率: 0.01"
echo "  Batch size: 128"
echo "  Criteria: magnitude"
echo ""
echo "【RSST专用参数】"
echo "  正则化强度: 1.0"
echo "  正则化曲率: 4"
echo "  调度策略: exp_custom_exponents"
echo ""
echo "【Refill专用参数】"
echo "  Fillback率: 0.0"
echo ""
echo "【预计时间】"
echo "  一个Epoch: ~1.2分钟"
echo "  一个State (60 epochs): ~72分钟"
echo "  单个实验 (16 states): ~19.2小时"
echo "  四个实验并行: ~19-22小时（共享GPU）"
echo ""
echo "【监控命令】"
echo "  查看进程: ps aux | grep main_imp_fillback"
echo "  查看日志: tail -f logs_vit_small_70p/*.log"
echo "  查看进度: grep 'pruning state' logs_vit_small_70p/*.log"
echo "  查看时间: grep 'State.*completed' logs_vit_small_70p/*.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
