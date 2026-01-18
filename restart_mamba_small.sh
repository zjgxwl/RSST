#!/bin/bash

##############################################################################
# 重启修复后的Mamba-Small实验
# Bug修复：SSM层完整同步剪枝 + RSST mask初始化
##############################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="/workspace/ycx/RSST"

# 基础参数
DATASET_CIFAR10="${BASE_DIR}/datasets/cifar10"
DATASET_CIFAR100="${BASE_DIR}/datasets/cifar100"
INIT_FILE="${BASE_DIR}/init_model/mamba_small_cifar10_init.pth.tar"
INIT_FILE_C100="${BASE_DIR}/init_model/mamba_small_cifar100_init.pth.tar"

# 剪枝参数
RATE=0.7
MLP_RATIO=0.7
PRUNING_TIMES=16
EPOCHS=60
LR=0.01
BATCH_SIZE=128
SORTING_MODE="global"

echo "=========================================================================="
echo "🔄 重启修复后的Mamba-Small实验"
echo "=========================================================================="
echo "修复内容:"
echo "  ✅ SSM层完整同步剪枝（in_proj, conv1d, x_proj, A_log, D, out_proj）"
echo "  ✅ RSST patch_embed mask跳过逻辑"
echo "=========================================================================="
echo ""

# ========== 1. CIFAR-10 Refill (GPU 0) ==========
LOG_FILE="${BASE_DIR}/logs_mamba_small_70p/mamba_small_cifar10_refill_70p_${TIMESTAMP}.log"
EXP_NAME="mamba_small_cifar10_refill_70p_${TIMESTAMP}"

echo "[1/4] 启动 Mamba-Small CIFAR-10 Refill (GPU 0)"
nohup python -u main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar10 \
    --data ${DATASET_CIFAR10} \
    --mamba_structured \
    --mamba_prune_target both \
    --rate ${RATE} \
    --mamba_mlp_prune_ratio ${MLP_RATIO} \
    --pruning_times ${PRUNING_TIMES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --sorting_mode ${SORTING_MODE} \
    --struct refill \
    --fillback_rate 0.0 \
    --init ${INIT_FILE} \
    --exp_name ${EXP_NAME} \
    > ${LOG_FILE} 2>&1 &

PID1=$!
echo "  ✓ PID: ${PID1}, Log: ${LOG_FILE}"
sleep 5

# ========== 2. CIFAR-10 RSST (GPU 0) ==========
LOG_FILE="${BASE_DIR}/logs_mamba_small_70p/mamba_small_cifar10_rsst_70p_${TIMESTAMP}.log"
EXP_NAME="mamba_small_cifar10_rsst_70p_${TIMESTAMP}"

echo "[2/4] 启动 Mamba-Small CIFAR-10 RSST (GPU 0)"
nohup python -u main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar10 \
    --data ${DATASET_CIFAR10} \
    --mamba_structured \
    --mamba_prune_target both \
    --rate ${RATE} \
    --mamba_mlp_prune_ratio ${MLP_RATIO} \
    --pruning_times ${PRUNING_TIMES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --sorting_mode ${SORTING_MODE} \
    --struct rsst \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --init ${INIT_FILE} \
    --exp_name ${EXP_NAME} \
    > ${LOG_FILE} 2>&1 &

PID2=$!
echo "  ✓ PID: ${PID2}, Log: ${LOG_FILE}"
sleep 5

# ========== 3. CIFAR-100 Refill (GPU 1) ==========
LOG_FILE="${BASE_DIR}/logs_mamba_small_70p/mamba_small_cifar100_refill_70p_${TIMESTAMP}.log"
EXP_NAME="mamba_small_cifar100_refill_70p_${TIMESTAMP}"

echo "[3/4] 启动 Mamba-Small CIFAR-100 Refill (GPU 1)"
CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar100 \
    --data ${DATASET_CIFAR100} \
    --mamba_structured \
    --mamba_prune_target both \
    --rate ${RATE} \
    --mamba_mlp_prune_ratio ${MLP_RATIO} \
    --pruning_times ${PRUNING_TIMES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --sorting_mode ${SORTING_MODE} \
    --struct refill \
    --fillback_rate 0.0 \
    --init ${INIT_FILE_C100} \
    --exp_name ${EXP_NAME} \
    > ${LOG_FILE} 2>&1 &

PID3=$!
echo "  ✓ PID: ${PID3}, Log: ${LOG_FILE}"
sleep 5

# ========== 4. CIFAR-100 RSST (GPU 1) ==========
LOG_FILE="${BASE_DIR}/logs_mamba_small_70p/mamba_small_cifar100_rsst_70p_${TIMESTAMP}.log"
EXP_NAME="mamba_small_cifar100_rsst_70p_${TIMESTAMP}"

echo "[4/4] 启动 Mamba-Small CIFAR-100 RSST (GPU 1)"
CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar100 \
    --data ${DATASET_CIFAR100} \
    --mamba_structured \
    --mamba_prune_target both \
    --rate ${RATE} \
    --mamba_mlp_prune_ratio ${MLP_RATIO} \
    --pruning_times ${PRUNING_TIMES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --sorting_mode ${SORTING_MODE} \
    --struct rsst \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --init ${INIT_FILE_C100} \
    --exp_name ${EXP_NAME} \
    > ${LOG_FILE} 2>&1 &

PID4=$!
echo "  ✓ PID: ${PID4}, Log: ${LOG_FILE}"

echo ""
echo "=========================================================================="
echo "✅ 所有实验已启动"
echo "=========================================================================="
echo "进程列表:"
echo "  [1] CIFAR-10 Refill: PID ${PID1} (GPU 0)"
echo "  [2] CIFAR-10 RSST:   PID ${PID2} (GPU 0)"
echo "  [3] CIFAR-100 Refill: PID ${PID3} (GPU 1)"
echo "  [4] CIFAR-100 RSST:   PID ${PID4} (GPU 1)"
echo ""
echo "监控命令:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f ${BASE_DIR}/logs_mamba_small_70p/*.log"
echo "=========================================================================="
