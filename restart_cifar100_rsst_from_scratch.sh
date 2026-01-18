#!/bin/bash

##############################################################################
# 从头重新启动CIFAR-100 RSST实验
# 由于checkpoint与模型结构不匹配，无法恢复
##############################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="/workspace/ycx/RSST"

# 实验参数 (与原始脚本完全一致)
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
SAVE_DIR="${BASE_DIR}/checkpoint/vit_small_70p/cifar100_rsst_restart"
EXP_NAME="cifar100_rsst_70p_restart_${TIMESTAMP}"
LOG_FILE="${BASE_DIR}/logs_vit_small_70p/${EXP_NAME}.log"

echo "=========================================================================="
echo "🔄 从头重新启动CIFAR-100 RSST实验"
echo "=========================================================================="
echo "问题诊断:"
echo "  ❌ Checkpoint保存的是已剪枝模型状态"
echo "  ❌ Resume时创建的模型是未剪枝状态"
echo "  ❌ 参数维度不匹配 (192 vs 384)"
echo ""
echo "解决方案:"
echo "  ✅ 从头重新启动实验"
echo "  ✅ 使用相同的参数配置"
echo "  ✅ 新的save_dir和exp_name避免冲突"
echo "=========================================================================="
echo ""

# ========== 启动实验 ==========
echo "[1/1] 从头启动 CIFAR-100 RSST (GPU 1)"
echo "命令参数:"
echo "  --arch $ARCH --dataset $DATASET --data $DATA_PATH"
echo "  --struct $STRUCT --vit_pretrained --vit_structured"
echo "  --vit_prune_target both --criteria magnitude"
echo "  --rate $RATE --mlp_prune_ratio $MLP_RATIO"
echo "  --pruning_times $PRUNING_TIMES --epochs $EPOCHS"
echo "  --batch_size $BATCH_SIZE --sorting_mode $SORTING_MODE"
echo "  --lr $LR --reg_granularity_prune $REG_GRANULARITY"
echo "  --RST_schedule $RST_SCHEDULE --exponents $EXPONENTS"
echo "  --init $INIT_FILE --save_dir $SAVE_DIR --exp_name $EXP_NAME"
echo ""

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data $DATA_PATH \
    --struct $STRUCT \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate $RATE \
    --mlp_prune_ratio $MLP_RATIO \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --lr $LR \
    --reg_granularity_prune $REG_GRANULARITY \
    --RST_schedule $RST_SCHEDULE \
    --exponents $EXPONENTS \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID=$!
echo "  ✓ PID: ${PID}"
echo "  ✓ GPU: 1 (CUDA_VISIBLE_DEVICES=1)"
echo "  ✓ 日志: $LOG_FILE"
echo "  ✓ 保存目录: $SAVE_DIR"
echo ""

echo "=========================================================================="
echo "✅ 实验已重新启动 (从头开始)"
echo "=========================================================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi -i 1"
echo "  kill $PID  # 如需停止"
echo ""
echo "预期进度:"
echo "  State 0: 60个epoch (约1-2小时)"
echo "  State 1-15: 每个state 60个epoch (约24-48小时)"
echo "  总计: ~24-72小时"
echo "=========================================================================="
