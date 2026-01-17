#!/bin/bash
# Mamba-Small 70%剪枝实验 - 完整对比（Refill + RSST）
# CIFAR-10和CIFAR-100，共4个实验

echo "========================================================="
echo "Mamba-Small 70% Pruning - Complete Comparison"
echo "Refill vs RSST on CIFAR-10/100"
echo "========================================================="

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

# 获取当前时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

# 实验1: CIFAR-10 + Refill (GPU 0)
echo ""
echo "实验1: CIFAR-10 + Refill (GPU 0)"
echo "---------------------------------------------------------"
DATASET="cifar10"
EXP_NAME="mamba_small_cifar10_refill_70p_${TIMESTAMP}"

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
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID1=$!
echo "已启动 PID: $PID1"
sleep 2

# 实验2: CIFAR-10 + RSST (GPU 0)
echo ""
echo "实验2: CIFAR-10 + RSST (GPU 0)"
echo "---------------------------------------------------------"
DATASET="cifar10"
EXP_NAME="mamba_small_cifar10_rsst_70p_${TIMESTAMP}"

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

PID2=$!
echo "已启动 PID: $PID2"
sleep 2

# 实验3: CIFAR-100 + Refill (GPU 1)
echo ""
echo "实验3: CIFAR-100 + Refill (GPU 1)"
echo "---------------------------------------------------------"
DATASET="cifar100"
EXP_NAME="mamba_small_cifar100_refill_70p_${TIMESTAMP}"

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
    --struct refill \
    --fillback_rate 0.0 \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID3=$!
echo "已启动 PID: $PID3"
sleep 2

# 实验4: CIFAR-100 + RSST (GPU 1)
echo ""
echo "实验4: CIFAR-100 + RSST (GPU 1)"
echo "---------------------------------------------------------"
DATASET="cifar100"
EXP_NAME="mamba_small_cifar100_rsst_70p_${TIMESTAMP}"

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

PID4=$!
echo "已启动 PID: $PID4"

echo ""
echo "========================================================="
echo "所有4个实验已启动"
echo "========================================================="
echo ""
echo "实验配置："
echo "  - 模型: ${ARCH}"
echo "  - 剪枝率: SSM ${PRUNE_RATE}, MLP ${MLP_PRUNE_RATE}"
echo "  - 剪枝目标: both (SSM + MLP)"
echo "  - 剪枝轮次: ${PRUNING_TIMES}"
echo "  - 训练轮次: ${EPOCHS}"
echo "  - GPU分配: GPU0 (CIFAR-10), GPU1 (CIFAR-100)"
echo ""
echo "进程ID："
echo "  - CIFAR-10 Refill: $PID1"
echo "  - CIFAR-10 RSST:   $PID2"
echo "  - CIFAR-100 Refill: $PID3"
echo "  - CIFAR-100 RSST:   $PID4"
echo ""
echo "日志文件："
echo "  $LOG_DIR/mamba_small_cifar10_refill_70p_${TIMESTAMP}.log"
echo "  $LOG_DIR/mamba_small_cifar10_rsst_70p_${TIMESTAMP}.log"
echo "  $LOG_DIR/mamba_small_cifar100_refill_70p_${TIMESTAMP}.log"
echo "  $LOG_DIR/mamba_small_cifar100_rsst_70p_${TIMESTAMP}.log"
echo ""
echo "监控命令："
echo "  # 查看所有日志"
echo "  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
echo ""
echo "  # 查看进程状态"
echo "  ps aux | grep 'main_imp_fillback.py.*mamba'"
echo ""
echo "  # 查看GPU使用"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "========================================================="
