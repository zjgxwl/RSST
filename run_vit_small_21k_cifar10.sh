#!/bin/bash

# =========================================================
# ViT-Small ImageNet-21K 预训练 - CIFAR-10 实验
# Refill vs RSST on CIFAR-10
# =========================================================

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate structlth

BASE_DIR="/workspace/ycx/RSST"
cd $BASE_DIR

# 设置代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
echo "✓ 代理已设置: http://127.0.0.1:7897"
echo ""

# =========================================================
# 实验配置
# =========================================================

DATASET="cifar10"
DATA_PATH="data/cifar10"
ARCH="vit_small"
RATE=0.7
MLP_PRUNE_RATIO=0.7

# 优化版配置
PRUNING_TIMES=11              # 减少pruning次数（16 -> 11）
EPOCHS=100                    # 增加epochs（60 -> 100）
BATCH_SIZE=256                # 大batch加速
DECREASING_LR="30,60,85"      # 学习率衰减策略
LR=0.01

# 21K预训练特定配置
# 初始化文件将自动生成
INIT_FILE_21K="init_model/vit_small_cifar10_21k_pretrained_init.pth.tar"

# 日志和保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_DIR="${BASE_DIR}/logs_vit_small_70p_21k"
CHECKPOINT_BASE="${BASE_DIR}/checkpoint/vit_small_70p_21k"

mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_BASE

echo "=========================================================
ViT-Small 70% Pruning - ImageNet-21K预训练
CIFAR-10: Refill vs RSST
========================================================="

# =========================================================
# 实验1: CIFAR-10 + Refill (GPU 0)
# =========================================================

EXP_NAME="cifar10_21k_refill_70p_${TIMESTAMP}"
SAVE_DIR="${CHECKPOINT_BASE}/cifar10_refill"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验1: CIFAR-10 + Refill + 21K预训练 (GPU 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data $DATA_PATH \
    --struct refill \
    --vit_pretrained \
    --vit_pretrained_21k \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate $RATE \
    --mlp_prune_ratio $MLP_PRUNE_RATIO \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --sorting_mode global \
    --lr $LR \
    --decreasing_lr $DECREASING_LR \
    --fillback_rate 0.0 \
    --init $INIT_FILE_21K \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID1=$!
echo "✓ 进程已启动 PID: $PID1"
echo "✓ 日志文件: $LOG_FILE"

sleep 5

# =========================================================
# 实验2: CIFAR-10 + RSST (GPU 1)
# =========================================================

EXP_NAME="cifar10_21k_rsst_70p_${TIMESTAMP}"
SAVE_DIR="${CHECKPOINT_BASE}/cifar10_rsst"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 实验2: CIFAR-10 + RSST + 21K预训练 (GPU 1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data $DATA_PATH \
    --struct rsst \
    --vit_pretrained \
    --vit_pretrained_21k \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate $RATE \
    --mlp_prune_ratio $MLP_PRUNE_RATIO \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --sorting_mode global \
    --lr $LR \
    --decreasing_lr $DECREASING_LR \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --init $INIT_FILE_21K \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID2=$!
echo "✓ 进程已启动 PID: $PID2"
echo "✓ 日志文件: $LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 全部2个21K预训练实验已启动完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "【进程ID & GPU分配】"
echo "  实验1 (CIFAR-10 Refill 21K) [GPU 0]: $PID1"
echo "  实验2 (CIFAR-10 RSST   21K) [GPU 1]: $PID2"
echo ""

echo "【预训练对比】"
echo "  原实验: ImageNet-1K  (已在运行)"
echo "  新实验: ImageNet-21K (刚启动)"
echo ""

echo "【实验配置】"
echo "  模型: vit_small"
echo "  预训练: ImageNet-21K"
echo "  数据集: CIFAR-10"
echo "  剪枝率: 70%"
echo "  Pruning times: $PRUNING_TIMES"
echo "  Epochs/state: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Decreasing LR: [$DECREASING_LR]"
echo ""

echo "【监控命令】"
echo "  查看日志: tail -f ${LOG_DIR}/*.log"
echo "  查看进程: ps aux | grep $PID1"
echo "  查看GPU: nvidia-smi"
echo ""

echo "🎯 预期：21K预训练模型应该比1K预训练有更好的表现！"
