#!/bin/bash

##############################################################################
# 快速测试Mamba剪枝实验
# 参数: epoch=2, pruning_times=2
##############################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="/workspace/ycx/RSST"

# 快速测试参数
ARCH="mamba_small"
DATASET="cifar10"
DATA_PATH="${BASE_DIR}/datasets/cifar10"
STRUCT="refill"  # 使用Refill方法测试剪枝功能
RATE=0.7
MLP_RATIO=0.7
PRUNING_TIMES=2  # 只测试2个迭代
EPOCHS=2         # 每个state只训练2个epoch
BATCH_SIZE=128
SORTING_MODE="global"
LR=0.01
INIT_FILE="${BASE_DIR}/init_model/mamba_small_cifar10_init.pth.tar"
SAVE_DIR="${BASE_DIR}/checkpoint/test_mamba_quick"
EXP_NAME="test_mamba_quick_${TIMESTAMP}"
LOG_FILE="${BASE_DIR}/logs_test/${EXP_NAME}.log"

echo "=========================================================================="
echo "🧪 快速测试Mamba剪枝实验 (修复验证)"
echo "=========================================================================="
echo "测试参数:"
echo "  📊 模型: $ARCH"
echo "  📊 数据集: $DATASET"
echo "  📊 方法: $STRUCT"
echo "  📊 剪枝率: ${RATE} (SSM & MLP)"
echo "  📊 迭代次数: $PRUNING_TIMES"
echo "  📊 每轮Epoch: $EPOCHS"
echo "  📊 排序模式: $SORTING_MODE"
echo ""
echo "预期测试:"
echo "  ✅ State 0: 基础训练 (2 epochs)"
echo "  ✅ State 1: 剪枝 + 微调 (2 epochs)"
echo "  ✅ 验证维度同步和训练稳定性"
echo "=========================================================================="
echo ""

# ========== 启动测试实验 ==========
echo "[1/1] 启动快速测试实验"

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data $DATA_PATH \
    --struct $STRUCT \
    --mamba_structured \
    --mamba_prune_target both \
    --rate $RATE \
    --mamba_mlp_prune_ratio $MLP_RATIO \
    --pruning_times $PRUNING_TIMES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --sorting_mode $SORTING_MODE \
    --lr $LR \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID=$!
echo "  ✓ PID: ${PID}"
echo "  ✓ GPU: 1 (CUDA_VISIBLE_DEVICES=1)"
echo "  ✓ 日志: $LOG_FILE"
echo ""

echo "=========================================================================="
echo "✅ 测试实验已启动"
echo "=========================================================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi -i 1"
echo "  kill $PID  # 如需停止"
echo ""
echo "预期时间:"
echo "  State 0: ~4-6分钟 (2 epochs)"
echo "  State 1: ~4-6分钟 (剪枝 + 2 epochs)"
echo "  总计: ~8-12分钟"
echo ""
echo "成功标志:"
echo "  ✅ State 0 completed! (基础训练完成)"
echo "  ✅ State 1 completed! (剪枝训练完成)"
echo "  ✅ 无维度不匹配错误"
echo "=========================================================================="
