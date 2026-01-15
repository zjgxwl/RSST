#!/bin/bash
# ViT-RSST剪枝实验脚本
# 用法: bash run_vit_rsst.sh

echo "=========================================="
echo "Running ViT-RSST Pruning Experiment"
echo "=========================================="

# 基础配置
DATASET="cifar100"
ARCH="vit_tiny"  # 可选: vit_tiny, vit_small, vit_base
STRUCT="rsst"  # 可选: rsst, refill
CRITERIA="l1"  # 可选: magnitude, l1, l2, saliency
PRETRAINED=false  # true=使用预训练模型（推荐）, false=随机初始化
SAVE_DIR="results/vit_tiny_rsst_cifar100"

# 训练参数
EPOCHS=120
BATCH_SIZE=128  # ViT通常需要较小的batch size
LR=0.001  # ViT推荐使用较小的学习率
WARMUP=20
DECREASING_LR="60,90"

# 剪枝参数
PRUNING_TIMES=15  # ViT建议使用较少的剪枝次数
RATE=0.15  # 每次剪枝15%（较温和）
PRUNE_TYPE="lt"

# RSST参数
RST_SCHEDULE="exp_custom_exponents"
REG_GRANULARITY=0.5  # ViT建议使用较小的正则化强度
EXPONENTS=3

# 其他参数
SEED=42
GPU=0

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Architecture: $ARCH"
echo "  Method: $STRUCT"
echo "  Criteria: $CRITERIA"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Pruning Times: $PRUNING_TIMES"
echo "  Pruning Rate: $RATE"
echo "  Save Directory: $SAVE_DIR"
echo "=========================================="

# 运行训练
if [ "$PRETRAINED" = true ]; then
    PRETRAINED_FLAG="--pretrained"
else
    PRETRAINED_FLAG=""
fi

python main_imp_fillback.py \
    --dataset $DATASET \
    --arch $ARCH \
    --struct $STRUCT \
    --criteria $CRITERIA \
    --save_dir $SAVE_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup $WARMUP \
    --decreasing_lr $DECREASING_LR \
    --pruning_times $PRUNING_TIMES \
    --rate $RATE \
    --prune_type $PRUNE_TYPE \
    --RST_schedule $RST_SCHEDULE \
    --reg_granularity_prune $REG_GRANULARITY \
    --exponents $EXPONENTS \
    --seed $SEED \
    --gpu $GPU \
    $PRETRAINED_FLAG

echo "=========================================="
echo "Training completed!"
echo "Results saved to: $SAVE_DIR"
echo "=========================================="

