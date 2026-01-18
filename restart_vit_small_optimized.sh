#!/bin/bash

# ============================================================================
# ViT-Small 70%剪枝实验脚本 - 优化版本（2026-01-18）
# ============================================================================
# 更新：
#   - batch_size: 128 -> 256
#   - pruning_times: 16 -> 11
#   - epochs: 60 -> 100
#   - decreasing_lr: 适配100 epochs
# ============================================================================

# 代理设置
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
echo "✓ 代理已设置: $http_proxy"

# 创建日志目录
mkdir -p logs_vit_small_70p_v2
mkdir -p checkpoint/vit_small_70p_v2

# 时间戳
TIMESTAMP=$(date +%m%d_%H%M)

# ============================================================================
# 通用参数（Refill和RSST完全一致）
# ============================================================================

ARCH="vit_small"
DATASET_CIFAR10="cifar10"
DATASET_CIFAR100="cifar100"
DATA_PATH_10="data/cifar10"
DATA_PATH_100="data/cifar100"

# 模型配置
PRETRAINED="--vit_pretrained"
VIT_STRUCTURED="--vit_structured"
PRUNE_TARGET="--vit_prune_target both"
INIT_FILE_CIFAR10="init_model/vit_small_cifar10_pretrained_init.pth.tar"
INIT_FILE_CIFAR100="init_model/vit_small_cifar100_pretrained_init.pth.tar"

# 剪枝配置（70%剪枝率）
CRITERIA="--criteria magnitude"
PRUNE_RATE="--rate 0.7"
MLP_RATE="--mlp_prune_ratio 0.7"

# 迭代策略（优化版）
PRUNING_TIMES="--pruning_times 11"     # 11次迭代（更少但每次更充分训练）
EPOCHS="--epochs 100"                  # 100个epoch/iteration（更充分训练）
BATCH_SIZE="--batch_size 256"          # 256 batch size（加速训练）

# 排序策略
SORTING_MODE="--sorting_mode global"   # 全局混合排序

# 学习率（适配100 epochs）
LR="--lr 0.01"
DECREASING_LR="--decreasing_lr 30,60,85"  # 在30、60、85 epoch时降低学习率

# ============================================================================
# RSST方法专用参数
# ============================================================================

REG_GRANULARITY="--reg_granularity_prune 1.0"  # 正则化强度1.0
RST_SCHEDULE="--RST_schedule exp_custom_exponents"
EXPONENTS="--exponents 4"                      # 曲率4

# ============================================================================
# 实验1: CIFAR-10 + Refill (70%)
# ============================================================================

SAVE_DIR_1="checkpoint/vit_small_70p_v2/cifar10_refill"
EXP_NAME_1="cifar10_refill_70p_v2_${TIMESTAMP}"
LOG_FILE_1="logs_vit_small_70p_v2/${EXP_NAME_1}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验1: CIFAR-10 + Refill (70%剪枝) [GPU 0]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR10 \
    --data $DATA_PATH_10 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $DECREASING_LR \
    --fillback_rate 0.0 \
    --init $INIT_FILE_CIFAR10 \
    --save_dir $SAVE_DIR_1 \
    --exp_name $EXP_NAME_1 \
    > $LOG_FILE_1 2>&1 &

PID_1=$!
echo "✓ 进程已启动 (PID: $PID_1)"
echo "✓ 日志文件: $LOG_FILE_1"
echo ""

# ============================================================================
# 实验2: CIFAR-10 + RSST (70%)
# ============================================================================

SAVE_DIR_2="checkpoint/vit_small_70p_v2/cifar10_rsst"
EXP_NAME_2="cifar10_rsst_70p_v2_${TIMESTAMP}"
LOG_FILE_2="logs_vit_small_70p_v2/${EXP_NAME_2}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验2: CIFAR-10 + RSST (70%剪枝) [GPU 0]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR10 \
    --data $DATA_PATH_10 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $DECREASING_LR \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE_CIFAR10 \
    --save_dir $SAVE_DIR_2 \
    --exp_name $EXP_NAME_2 \
    > $LOG_FILE_2 2>&1 &

PID_2=$!
echo "✓ 进程已启动 (PID: $PID_2)"
echo "✓ 日志文件: $LOG_FILE_2"
echo ""

# ============================================================================
# 实验3: CIFAR-100 + Refill (70%)
# ============================================================================

SAVE_DIR_3="checkpoint/vit_small_70p_v2/cifar100_refill"
EXP_NAME_3="cifar100_refill_70p_v2_${TIMESTAMP}"
LOG_FILE_3="logs_vit_small_70p_v2/${EXP_NAME_3}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验3: CIFAR-100 + Refill (70%剪枝) [GPU 1]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR100 \
    --data $DATA_PATH_100 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $DECREASING_LR \
    --fillback_rate 0.0 \
    --init $INIT_FILE_CIFAR100 \
    --save_dir $SAVE_DIR_3 \
    --exp_name $EXP_NAME_3 \
    > $LOG_FILE_3 2>&1 &

PID_3=$!
echo "✓ 进程已启动 (PID: $PID_3)"
echo "✓ 日志文件: $LOG_FILE_3"
echo ""

# ============================================================================
# 实验4: CIFAR-100 + RSST (70%)
# ============================================================================

SAVE_DIR_4="checkpoint/vit_small_70p_v2/cifar100_rsst"
EXP_NAME_4="cifar100_rsst_70p_v2_${TIMESTAMP}"
LOG_FILE_4="logs_vit_small_70p_v2/${EXP_NAME_4}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验4: CIFAR-100 + RSST (70%剪枝) [GPU 1]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET_CIFAR100 \
    --data $DATA_PATH_100 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $PRUNE_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $SORTING_MODE \
    $LR \
    $DECREASING_LR \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE_CIFAR100 \
    --save_dir $SAVE_DIR_4 \
    --exp_name $EXP_NAME_4 \
    > $LOG_FILE_4 2>&1 &

PID_4=$!
echo "✓ 进程已启动 (PID: $PID_4)"
echo "✓ 日志文件: $LOG_FILE_4"
echo ""

# ============================================================================
# 总结信息
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 全部4个实验已启动完成！（优化版配置）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "【进程ID & GPU分配】"
echo "  实验1 (CIFAR-10  Refill) [GPU 0]: $PID_1"
echo "  实验2 (CIFAR-10  RSST)   [GPU 0]: $PID_2"
echo "  实验3 (CIFAR-100 Refill) [GPU 1]: $PID_3"
echo "  实验4 (CIFAR-100 RSST)   [GPU 1]: $PID_4"
echo ""
echo "【日志文件】"
echo "  实验1: $LOG_FILE_1"
echo "  实验2: $LOG_FILE_2"
echo "  实验3: $LOG_FILE_3"
echo "  实验4: $LOG_FILE_4"
echo ""
echo "【优化后参数（相比旧版）】"
echo "  ⚡ Batch size: 128 -> 256 (提升训练速度)"
echo "  📉 Pruning times: 16 -> 11 (减少迭代次数)"
echo "  📈 Epochs/state: 60 -> 100 (更充分训练)"
echo "  🎯 Decreasing LR: [30, 60, 85] (适配100 epochs)"
echo ""
echo "【通用参数】"
echo "  剪枝率: 70% (head & mlp)"
echo "  迭代次数: 11次"
echo "  每次Epoch: 100个"
echo "  排序模式: global"
echo "  学习率: 0.01"
echo "  Batch size: 256"
echo "  Weight decay: 5e-4"
echo "  Criteria: magnitude"
echo ""
echo "【RSST专用参数】"
echo "  正则化强度: 1.0"
echo "  正则化曲率: 4"
echo "  调度策略: exp_custom_exponents"
echo ""
echo "【Refill专用参数】"
echo "  Fillback率: 0.0"
echo ""
echo "【预计时间】(优化版，双GPU加速 ⚡⚡)"
echo "  Batch size加倍，训练速度提升 ~40%"
echo "  一个Epoch: ~0.7分钟 (vs 旧版1.2分钟)"
echo "  一个State (100 epochs): ~70分钟 (vs 旧版72分钟)"
echo "  单个实验 (11 states): ~12.8小时 (vs 旧版19.2小时)"
echo "  双GPU并行总时间: ~6-7小时 ⚡ (vs 旧版9-11小时)"
echo ""
echo "【监控命令】"
echo "  查看GPU: nvidia-smi"
echo "  查看进程: ps aux | grep main_imp_fillback"
echo "  查看日志: tail -f logs_vit_small_70p_v2/*.log"
echo "  查看进度: grep 'pruning state' logs_vit_small_70p_v2/*.log"
echo "  查看时间: grep 'State.*completed' logs_vit_small_70p_v2/*.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
