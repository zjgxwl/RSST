#!/bin/bash

##############################################################################
# 重启CIFAR-100 RSST实验
# 从State 0, Epoch 39恢复训练
##############################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="/workspace/ycx/RSST"

# 实验参数
ARCH="vit_small"
DATASET="cifar100"
DATA_PATH="${BASE_DIR}/data/cifar100"
STRUCT="rsst"
RATE=0.7
MLP_RATIO=0.7
PRUNING_TIMES=16
EPOCHS=60
BATCH_SIZE=128
SORTING_MODE="global"
LR=0.01
REG_GRANULARITY=1.0
RST_SCHEDULE="exp_custom_exponents"
EXPONENTS=4
INIT_FILE="${BASE_DIR}/init_model/vit_small_cifar100_pretrained_init.pth.tar"
SAVE_DIR="${BASE_DIR}/checkpoint/vit_small_70p/cifar100_rsst"
RESUME_FILE="${SAVE_DIR}/0checkpoint.pth.tar"
EXP_NAME="cifar100_rsst_70p_resume_${TIMESTAMP}"
LOG_FILE="${BASE_DIR}/logs_vit_small_70p/${EXP_NAME}.log"

echo "=========================================================================="
echo "🔄 重启CIFAR-100 RSST实验"
echo "=========================================================================="
echo "恢复信息:"
echo "  📍 State: 0"
echo "  📍 Epoch: 39/60 (从checkpoint恢复)"
echo "  📍 Checkpoint: $RESUME_FILE"
echo "  📍 最佳精度: 53.35%"
echo ""
echo "GPU分配:"
echo "  ✅ 强制指定: --gpu 1"
echo "  ✅ 环境变量: CUDA_VISIBLE_DEVICES=1"
echo "=========================================================================="
echo ""

# ========== 启动实验 ==========
echo "[1/1] 启动 CIFAR-100 RSST 恢复训练 (GPU 1)"
echo "命令: python -u main_imp_fillback.py \\"
echo "  --arch $ARCH --dataset $DATASET --data $DATA_PATH \\"
echo "  --struct $STRUCT \\"
echo "  --resume --checkpoint $RESUME_FILE \\"
echo "  --exp_name $EXP_NAME"
echo ""
echo "GPU设置:"
echo "  CUDA_VISIBLE_DEVICES=1 (映射到实际GPU 1)"
echo "  代码内使用默认gpu=0 (在环境变量映射后)"
echo ""

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data $DATA_PATH \
    --struct $STRUCT \
    --resume \
    --checkpoint $RESUME_FILE \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID=$!
echo "  ✓ PID: ${PID}"
echo "  ✓ GPU: 1 (强制指定)"
echo "  ✓ 日志: $LOG_FILE"
echo ""

echo "=========================================================================="
echo "✅ 实验已重启"
echo "=========================================================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi -i 1"
echo "  kill $PID  # 如需停止"
echo "=========================================================================="
