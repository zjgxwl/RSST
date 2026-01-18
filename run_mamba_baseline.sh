#!/bin/bash
##############################################################################
# Mamba-Small Baseline 训练脚本
# 测试 Mamba-Small 在 CIFAR-10/100 上的最佳性能（无剪枝）
##############################################################################

echo "=========================================================================="
echo "🎯 Mamba-Small Baseline 性能测试"
echo "=========================================================================="
echo ""
echo "训练策略 (基于 Gemini 建议):"
echo "  ✅ 优化器: AdamW with Cosine LR"
echo "  ✅ 学习率: 1e-3 with warmup"
echo "  ✅ Weight Decay: 0.05 (关键参数)"
echo "  ✅ 训练轮数: 300 epochs"
echo "  ✅ 数据增强: RandAugment + Mixup + Cutmix"
echo "  ✅ Label Smoothing: 0.1"
echo ""
echo "预期性能:"
echo "  📊 CIFAR-10:  94-95.5% (从零训练)"
echo "  📊 CIFAR-100: 76-81% (从零训练)"
echo "=========================================================================="
echo ""

# ============================================================================
# 配置选项
# ============================================================================

# 选择实验类型（取消注释想要运行的实验）
RUN_CIFAR10=true          # CIFAR-10 实验
RUN_CIFAR100=true         # CIFAR-100 实验
RUN_QUICK_TEST=false      # 快速测试（30 epochs）

# 模型选择
ARCH="mamba_small"        # mamba_tiny, mamba_small, mamba_base

# GPU 分配
GPU_C10=0                 # CIFAR-10 使用的 GPU
GPU_C100=1                # CIFAR-100 使用的 GPU

# ============================================================================
# 基础参数
# ============================================================================

BASE_DIR="/workspace/ycx/RSST"
PYTHON_PATH="/root/miniconda3/envs/structlth/bin/python"
DATA_PATH="${BASE_DIR}/datasets"
SAVE_DIR="${BASE_DIR}/checkpoint/mamba_baseline"
LOG_DIR="${BASE_DIR}/logs_mamba_baseline"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)

# ============================================================================
# 训练参数（Gemini 推荐配置）
# ============================================================================

if [ "$RUN_QUICK_TEST" = true ]; then
    # 快速测试配置（验证流程）
    EPOCHS=30
    BATCH_SIZE=128
    LR=1e-3
    WEIGHT_DECAY=0.05
    WARMUP=5
    EVAL_INTERVAL=5
    echo "⚡ 使用快速测试配置 (30 epochs)"
else
    # 完整训练配置（追求最佳性能）
    EPOCHS=300
    BATCH_SIZE=128
    LR=1e-3
    WEIGHT_DECAY=0.05
    WARMUP=20
    EVAL_INTERVAL=10
    echo "🏆 使用完整训练配置 (300 epochs)"
fi

# 数据增强
USE_RANDAUGMENT="--use_randaugment"
USE_MIXUP="--use_mixup"
USE_CUTMIX="--use_cutmix"
MIXUP_ALPHA=0.8
CUTMIX_ALPHA=1.0
LABEL_SMOOTHING=0.1

# ============================================================================
# 实验 1: CIFAR-10 Baseline
# ============================================================================

if [ "$RUN_CIFAR10" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 实验 1: CIFAR-10 Baseline (GPU ${GPU_C10})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    DATASET="cifar10"
    EXP_NAME="${ARCH}_${DATASET}_baseline_${TIMESTAMP}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    echo "配置信息:"
    echo "  数据集: ${DATASET}"
    echo "  模型: ${ARCH}"
    echo "  训练轮数: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  学习率: ${LR}"
    echo "  Weight Decay: ${WEIGHT_DECAY}"
    echo "  Warmup: ${WARMUP} epochs"
    echo "  数据增强: RandAugment + Mixup + Cutmix"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_C10} nohup ${PYTHON_PATH} -u train_mamba_baseline.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP} \
        ${USE_RANDAUGMENT} \
        ${USE_MIXUP} \
        ${USE_CUTMIX} \
        --mixup_alpha ${MIXUP_ALPHA} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --gpu 0 \
        --save_dir ${SAVE_DIR}/${DATASET} \
        --log_interval 50 \
        --eval_interval ${EVAL_INTERVAL} \
        > ${LOG_FILE} 2>&1 &
    
    PID_C10=$!
    echo "  ✓ 已启动 (PID: ${PID_C10})"
    echo ""
    sleep 2
fi

# ============================================================================
# 实验 2: CIFAR-100 Baseline
# ============================================================================

if [ "$RUN_CIFAR100" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 实验 2: CIFAR-100 Baseline (GPU ${GPU_C100})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    DATASET="cifar100"
    EXP_NAME="${ARCH}_${DATASET}_baseline_${TIMESTAMP}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    echo "配置信息:"
    echo "  数据集: ${DATASET}"
    echo "  模型: ${ARCH}"
    echo "  训练轮数: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  学习率: ${LR}"
    echo "  Weight Decay: ${WEIGHT_DECAY}"
    echo "  Warmup: ${WARMUP} epochs"
    echo "  数据增强: RandAugment + Mixup + Cutmix"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_C100} nohup ${PYTHON_PATH} -u train_mamba_baseline.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP} \
        ${USE_RANDAUGMENT} \
        ${USE_MIXUP} \
        ${USE_CUTMIX} \
        --mixup_alpha ${MIXUP_ALPHA} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --gpu 0 \
        --save_dir ${SAVE_DIR}/${DATASET} \
        --log_interval 50 \
        --eval_interval ${EVAL_INTERVAL} \
        > ${LOG_FILE} 2>&1 &
    
    PID_C100=$!
    echo "  ✓ 已启动 (PID: ${PID_C100})"
    echo ""
fi

# ============================================================================
# 启动总结
# ============================================================================

echo "=========================================================================="
echo "✅ 实验已启动"
echo "=========================================================================="
echo ""

if [ "$RUN_CIFAR10" = true ]; then
    echo "  [PID ${PID_C10}]  CIFAR-10  Baseline (GPU ${GPU_C10})"
fi

if [ "$RUN_CIFAR100" = true ]; then
    echo "  [PID ${PID_C100}] CIFAR-100 Baseline (GPU ${GPU_C100})"
fi

echo ""
echo "日志目录: ${LOG_DIR}"
echo "Checkpoint目录: ${SAVE_DIR}"
echo ""
echo "=========================================================================="
echo "📊 监控命令"
echo "=========================================================================="
echo ""
echo "# 实时查看日志"

if [ "$RUN_CIFAR10" = true ]; then
    echo "tail -f ${LOG_DIR}/${ARCH}_cifar10_baseline_*.log"
fi

if [ "$RUN_CIFAR100" = true ]; then
    echo "tail -f ${LOG_DIR}/${ARCH}_cifar100_baseline_*.log"
fi

echo ""
echo "# 查看所有日志"
echo "tail -f ${LOG_DIR}/*.log"
echo ""
echo "# 查看GPU使用"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# 查看进程状态"
echo "ps aux | grep 'train_mamba_baseline.py'"
echo ""
echo "# 停止实验（如需）"

if [ "$RUN_CIFAR10" = true ] && [ "$RUN_CIFAR100" = true ]; then
    echo "kill ${PID_C10} ${PID_C100}"
elif [ "$RUN_CIFAR10" = true ]; then
    echo "kill ${PID_C10}"
elif [ "$RUN_CIFAR100" = true ]; then
    echo "kill ${PID_C100}"
fi

echo ""
echo "=========================================================================="
echo "⏱️  预计时间"
echo "=========================================================================="
echo ""

if [ "$RUN_QUICK_TEST" = true ]; then
    echo "  快速测试: ~2-3 小时"
else
    echo "  CIFAR-10:  ~2-3 天 (300 epochs)"
    echo "  CIFAR-100: ~2-3 天 (300 epochs)"
    echo "  总计:      ~2-3 天（双GPU并行）"
fi

echo ""
echo "=========================================================================="
echo "🎯 预期性能（从零训练）"
echo "=========================================================================="
echo ""
echo "  CIFAR-10:  94.0-95.5%"
echo "  CIFAR-100: 76.0-81.0%"
echo ""
echo "  注: 使用 ImageNet 预训练模型微调可达到更高精度"
echo "      CIFAR-10: 98.5-99.1%"
echo "      CIFAR-100: 88.5-91.0%"
echo ""
echo "=========================================================================="
echo "💡 关键优化点"
echo "=========================================================================="
echo ""
echo "  1. Weight Decay = 0.05 (关键！Mamba 非常敏感)"
echo "  2. 训练 300 epochs (小数据集需要更多迭代)"
echo "  3. 强数据增强 (RandAugment + Mixup + Cutmix)"
echo "  4. Cosine LR + Warmup"
echo "  5. Label Smoothing (防止过拟合)"
echo ""
echo "=========================================================================="
echo "🚀 实验已启动！祝训练顺利！"
echo "=========================================================================="
