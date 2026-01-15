#!/bin/bash

# ViT Head + MLP 组合剪枝测试脚本
# 使用RSST方法进行渐进式剪枝

echo "=================================="
echo "ViT Head + MLP 组合剪枝测试"
echo "=================================="

# 配置参数
DATASET="cifar100"
ARCH="vit_tiny"
STRUCT="rsst"
CRITERIA="magnitude"
HEAD_RATE=0.3
MLP_RATE=0.3
PRUNING_TIMES=3  # 快速测试，只做3次迭代
EPOCHS=5         # 快速测试，只训练5个epochs
BATCH_SIZE=128
REG_GRANULARITY=1.0
RST_SCHEDULE="exp_custom_exponents"
EXPONENTS=4

# 生成实验名称
TIMESTAMP=$(date +"%m%d_%H%M")
EXP_NAME="test_head_mlp_${CRITERIA}_h${HEAD_RATE}_m${MLP_RATE}_${TIMESTAMP}"

echo ""
echo "实验配置:"
echo "  - 数据集: $DATASET"
echo "  - 模型: $ARCH"
echo "  - 方法: $STRUCT"
echo "  - Criteria: $CRITERIA"
echo "  - Head剪枝率: $HEAD_RATE"
echo "  - MLP剪枝率: $MLP_RATE"
echo "  - 迭代次数: $PRUNING_TIMES"
echo "  - Epochs/迭代: $EPOCHS"
echo "  - 实验名: $EXP_NAME"
echo ""

# 启动训练
python main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --struct $STRUCT \
    --vit_structured \
    --vit_prune_target both \
    --criteria $CRITERIA \
    --rate $HEAD_RATE \
    --mlp_prune_ratio $MLP_RATE \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --exp_name $EXP_NAME \
    2>&1 | tee logs/${EXP_NAME}.log

echo ""
echo "=================================="
echo "测试完成！"
echo "日志保存在: logs/${EXP_NAME}.log"
echo "=================================="
