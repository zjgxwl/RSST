#!/bin/bash
# ImageNet上的ViT-RSST剪枝实验脚本

echo "=========================================="
echo "ImageNet ViT-RSST Pruning Experiment"
echo "=========================================="

# ⚠️ 请修改此路径为您的ImageNet数据集路径
IMAGENET_PATH="/path/to/imagenet"

# 基础配置
DATASET="imagenet"
ARCH="vit_small_imagenet"  # 可选: vit_small_imagenet, vit_base_imagenet, vit_large_imagenet
PRETRAINED=true  # 推荐使用预训练
STRUCT="rsst"    # 可选: rsst, refill
CRITERIA="l1"    # 可选: magnitude, l1, l2, saliency

# 训练参数
EPOCHS=10        # ImageNet预训练模型用少量epoch即可
BATCH_SIZE=256   # 根据GPU显存调整
LR=0.0001        # ImageNet微调要用小学习率
WARMUP=2
DECREASING_LR="6,8"
WORKERS=8        # 数据加载进程数

# 剪枝参数
PRUNING_TIMES=10  # ImageNet建议较少次数
RATE=0.15         # 每次剪枝15%
PRUNE_TYPE="lt"

# RSST参数
RST_SCHEDULE="exp_custom_exponents"
REG_GRANULARITY=0.1  # ImageNet用更小的正则化强度
EXPONENTS=2

# 其他参数
SEED=42
GPU=0
SAVE_DIR="results/imagenet_vit_small_rsst"

# 检查ImageNet路径
if [ ! -d "$IMAGENET_PATH/train" ] || [ ! -d "$IMAGENET_PATH/val" ]; then
    echo "❌ 错误: ImageNet数据集路径不正确!"
    echo "   当前路径: $IMAGENET_PATH"
    echo "   请修改脚本中的 IMAGENET_PATH 变量"
    echo ""
    echo "   ImageNet目录结构应为:"
    echo "   $IMAGENET_PATH/"
    echo "   ├── train/"
    echo "   │   ├── n01440764/"
    echo "   │   └── ..."
    echo "   └── val/"
    echo "       ├── n01440764/"
    echo "       └── ..."
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Data Path: $IMAGENET_PATH"
echo "  Architecture: $ARCH"
echo "  Pretrained: $PRETRAINED"
echo "  Method: $STRUCT"
echo "  Criteria: $CRITERIA"
echo "  Epochs per round: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Workers: $WORKERS"
echo "  Pruning Times: $PRUNING_TIMES"
echo "  Pruning Rate: $RATE"
echo "  Save Directory: $SAVE_DIR"
echo "=========================================="

# 构建命令
CMD="python main_imp_fillback.py \
    --dataset $DATASET \
    --data $IMAGENET_PATH \
    --arch $ARCH \
    --struct $STRUCT \
    --criteria $CRITERIA \
    --save_dir $SAVE_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup $WARMUP \
    --decreasing_lr $DECREASING_LR \
    --workers $WORKERS \
    --pruning_times $PRUNING_TIMES \
    --rate $RATE \
    --prune_type $PRUNE_TYPE \
    --RST_schedule $RST_SCHEDULE \
    --reg_granularity_prune $REG_GRANULARITY \
    --exponents $EXPONENTS \
    --seed $SEED \
    --gpu $GPU"

# 添加预训练标志
if [ "$PRETRAINED" = true ]; then
    CMD="$CMD --pretrained"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "=========================================="
echo "开始训练... (这可能需要很长时间)"
echo "=========================================="

# 运行训练
eval $CMD

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果保存在: $SAVE_DIR"
echo "=========================================="

