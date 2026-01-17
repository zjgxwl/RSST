#!/bin/bash
# Mamba-Small 70%剪枝实验 - RSST方法
# CIFAR-10和CIFAR-100

echo "========================================="
echo "Mamba-Small 70% Pruning - RSST Method"
echo "========================================="

# 通用配置
ARCH="mamba_small"
PRUNE_RATE=0.7
MLP_PRUNE_RATE=0.7
PRUNING_TIMES=16
EPOCHS=60
BATCH_SIZE=128
LR=0.01
SORTING_MODE="global"

# RSST特有参数
REG_GRANULARITY=1.0
RST_SCHEDULE="exp_custom_exponents"
EXPONENTS=4

# 创建日志目录
LOG_DIR="logs_mamba_small_70p"
mkdir -p $LOG_DIR

# 实验1: CIFAR-10 + RSST
echo ""
echo "实验1: CIFAR-10 + RSST"
echo "-----------------------------------------"
DATASET="cifar10"
EXP_NAME="mamba_small_cifar10_rsst_70p"

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
    --struct rsst \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

echo "已启动 PID: $!"
sleep 2

# 实验2: CIFAR-100 + RSST
echo ""
echo "实验2: CIFAR-100 + RSST"
echo "-----------------------------------------"
DATASET="cifar100"
EXP_NAME="mamba_small_cifar100_rsst_70p"

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
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

echo "已启动 PID: $!"

echo ""
echo "========================================="
echo "两个实验已启动"
echo "日志目录: $LOG_DIR"
echo "========================================="
echo ""
echo "查看日志："
echo "  tail -f $LOG_DIR/mamba_small_cifar10_rsst_70p.log"
echo "  tail -f $LOG_DIR/mamba_small_cifar100_rsst_70p.log"
echo ""
echo "查看进程："
echo "  ps aux | grep mamba"
