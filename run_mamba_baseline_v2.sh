#!/bin/bash
##############################################################################
# Mamba-Small Baseline V2 训练脚本 (全面优化)
# 
# V2 新增优化:
# ✅ Drop Path (Stochastic Depth)      → +0.5-1%
# ✅ EMA (Exponential Moving Average)  → +0.3-0.7%
# ✅ AutoAugment                       → +0.5-1%
# ✅ Random Erasing                    → +0.3-0.5%
# ✅ Gradient Clipping                 → 稳定性提升
# ✅ 混合精度训练 (AMP)                  → 2-3× 速度
# ✅ Layer-wise LR Decay               → +0.3-0.5%
# ✅ Test-Time Augmentation (可选)     → +0.5-1%
#
# 预期性能:
# - CIFAR-10:  97-98% (vs V1 的 94-95.5%)
# - CIFAR-100: 82-86% (vs V1 的 76-81%)
##############################################################################

echo "=========================================================================="
echo "🚀 Mamba-Small Baseline V2 训练 (全面优化)"
echo "=========================================================================="
echo ""
echo "V2 优化内容:"
echo "  ✅ Drop Path = 0.1"
echo "  ✅ EMA (decay=0.9999)"
echo "  ✅ AutoAugment (替代 RandAugment)"
echo "  ✅ Random Erasing (p=0.25)"
echo "  ✅ Gradient Clipping (max_norm=1.0)"
echo "  ✅ 混合精度训练 (AMP) → 2-3× 速度"
echo "  ✅ Layer-wise LR Decay (rate=0.65)"
echo "  ✅ 改进的 Cosine LR (指数 warmup)"
echo ""
echo "预期提升:"
echo "  📈 CIFAR-10:  94-95.5% → 97-98% (+2-3%)"
echo "  📈 CIFAR-100: 76-81% → 82-86% (+4-6%)"
echo "  ⚡ 训练速度: 2-3× 加速"
echo "=========================================================================="
echo ""

# ============================================================================
# 配置选项
# ============================================================================

# 选择实验类型
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
SAVE_DIR="${BASE_DIR}/checkpoint/mamba_baseline_v2"
LOG_DIR="${BASE_DIR}/logs_mamba_baseline_v2"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)

# ============================================================================
# 训练参数 (V2 优化配置)
# ============================================================================

if [ "$RUN_QUICK_TEST" = true ]; then
    # 快速测试配置
    EPOCHS=30
    BATCH_SIZE=128
    LR=1e-3
    WEIGHT_DECAY=0.05
    WARMUP=5
    DROP_PATH=0.1
    EVAL_INTERVAL=5
    USE_TTA="--no-use_tta"  # 快速测试不用 TTA
    echo "⚡ 使用快速测试配置 (30 epochs)"
else
    # V2 完整配置
    EPOCHS=300
    BATCH_SIZE=128
    LR=1e-3
    WEIGHT_DECAY=0.05
    WARMUP=20
    DROP_PATH=0.1              # ⭐ 新增
    EVAL_INTERVAL=10
    USE_TTA=""                 # 最终测试时使用 TTA
    echo "🏆 使用 V2 完整配置 (300 epochs + 全部优化)"
fi

# V2 优化参数
USE_EMA="--use_ema"                             # ⭐ EMA
EMA_DECAY=0.9999
USE_AMP="--use_amp"                             # ⭐ 混合精度
GRAD_CLIP=1.0                                   # ⭐ 梯度裁剪
USE_LAYERWISE_LR="--use_layerwise_lr"          # ⭐ Layer-wise LR
LAYERWISE_LR_DECAY=0.65

# 数据增强 (V2)
USE_AUTOAUGMENT="--use_autoaugment"            # ⭐ AutoAugment
USE_RANDOM_ERASING="--use_random_erasing"      # ⭐ Random Erasing
USE_MIXUP="--use_mixup"
USE_CUTMIX="--use_cutmix"
MIXUP_ALPHA=0.8
CUTMIX_ALPHA=1.0
LABEL_SMOOTHING=0.1

# ============================================================================
# 实验 1: CIFAR-10 Baseline V2
# ============================================================================

if [ "$RUN_CIFAR10" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 实验 1: CIFAR-10 Baseline V2 (GPU ${GPU_C10})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    DATASET="cifar10"
    EXP_NAME="${ARCH}_${DATASET}_baseline_v2_${TIMESTAMP}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    echo "配置信息:"
    echo "  数据集: ${DATASET}"
    echo "  模型: ${ARCH}"
    echo "  训练轮数: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  学习率: ${LR} (Layer-wise LR Decay)"
    echo "  Weight Decay: ${WEIGHT_DECAY}"
    echo "  Drop Path: ${DROP_PATH}"
    echo "  优化: EMA + AMP + AutoAugment + Random Erasing + Grad Clip"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_C10} nohup ${PYTHON_PATH} -u train_mamba_baseline_v2.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP} \
        --drop_path ${DROP_PATH} \
        ${USE_EMA} \
        --ema_decay ${EMA_DECAY} \
        ${USE_AMP} \
        --grad_clip ${GRAD_CLIP} \
        ${USE_LAYERWISE_LR} \
        --layerwise_lr_decay ${LAYERWISE_LR_DECAY} \
        ${USE_AUTOAUGMENT} \
        ${USE_RANDOM_ERASING} \
        ${USE_MIXUP} \
        ${USE_CUTMIX} \
        --mixup_alpha ${MIXUP_ALPHA} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --label_smoothing ${LABEL_SMOOTHING} \
        ${USE_TTA} \
        --tta_size 5 \
        --workers 8 \
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
# 实验 2: CIFAR-100 Baseline V2
# ============================================================================

if [ "$RUN_CIFAR100" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 实验 2: CIFAR-100 Baseline V2 (GPU ${GPU_C100})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    DATASET="cifar100"
    EXP_NAME="${ARCH}_${DATASET}_baseline_v2_${TIMESTAMP}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    echo "配置信息:"
    echo "  数据集: ${DATASET}"
    echo "  模型: ${ARCH}"
    echo "  训练轮数: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  学习率: ${LR} (Layer-wise LR Decay)"
    echo "  Weight Decay: ${WEIGHT_DECAY}"
    echo "  Drop Path: ${DROP_PATH}"
    echo "  优化: EMA + AMP + AutoAugment + Random Erasing + Grad Clip"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_C100} nohup ${PYTHON_PATH} -u train_mamba_baseline_v2.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP} \
        --drop_path ${DROP_PATH} \
        ${USE_EMA} \
        --ema_decay ${EMA_DECAY} \
        ${USE_AMP} \
        --grad_clip ${GRAD_CLIP} \
        ${USE_LAYERWISE_LR} \
        --layerwise_lr_decay ${LAYERWISE_LR_DECAY} \
        ${USE_AUTOAUGMENT} \
        ${USE_RANDOM_ERASING} \
        ${USE_MIXUP} \
        ${USE_CUTMIX} \
        --mixup_alpha ${MIXUP_ALPHA} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --label_smoothing ${LABEL_SMOOTHING} \
        ${USE_TTA} \
        --tta_size 5 \
        --workers 8 \
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
echo "✅ 实验已启动 (V2 全面优化版)"
echo "=========================================================================="
echo ""

if [ "$RUN_CIFAR10" = true ]; then
    echo "  [PID ${PID_C10}]  CIFAR-10  Baseline V2 (GPU ${GPU_C10})"
fi

if [ "$RUN_CIFAR100" = true ]; then
    echo "  [PID ${PID_C100}] CIFAR-100 Baseline V2 (GPU ${GPU_C100})"
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
    echo "tail -f ${LOG_DIR}/${ARCH}_cifar10_baseline_v2_*.log"
fi

if [ "$RUN_CIFAR100" = true ]; then
    echo "tail -f ${LOG_DIR}/${ARCH}_cifar100_baseline_v2_*.log"
fi

echo ""
echo "# 查看所有日志"
echo "tail -f ${LOG_DIR}/*.log"
echo ""
echo "# 查看GPU使用"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# 查看进程状态"
echo "ps aux | grep 'train_mamba_baseline_v2.py'"
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
    echo "  使用混合精度训练 (AMP): 速度提升 2-3×"
    echo ""
    echo "  CIFAR-10:  ~1-1.5 天 (vs V1 的 2-3 天)"
    echo "  CIFAR-100: ~1-1.5 天 (vs V1 的 2-3 天)"
    echo "  总计:      ~1-1.5 天（双GPU并行）"
fi

echo ""
echo "=========================================================================="
echo "🎯 预期性能提升"
echo "=========================================================================="
echo ""
echo "  CIFAR-10:"
echo "    V1:  94-95.5%"
echo "    V2:  97-98% (+2-3%)"
echo ""
echo "  CIFAR-100:"
echo "    V1:  76-81%"
echo "    V2:  82-86% (+4-6%)"
echo ""
echo "=========================================================================="
echo "💡 V2 优化亮点"
echo "=========================================================================="
echo ""
echo "  性能优化:"
echo "    1. Drop Path (0.1)           → +0.5-1%"
echo "    2. EMA (decay=0.9999)        → +0.3-0.7%"
echo "    3. AutoAugment               → +0.5-1%"
echo "    4. Random Erasing            → +0.3-0.5%"
echo "    5. Layer-wise LR Decay       → +0.3-0.5%"
echo "    6. Gradient Clipping         → 稳定性提升"
echo ""
echo "  工程优化:"
echo "    1. 混合精度训练 (AMP)         → 2-3× 速度"
echo "    2. DataLoader 优化           → 20-40% 加速"
echo ""
echo "  总计: 精度 +2-6%, 速度 2-3×"
echo ""
echo "=========================================================================="
echo "🚀 实验已启动！预祝训练顺利，突破 SOTA！"
echo "=========================================================================="
