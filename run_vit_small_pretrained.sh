#!/bin/bash

# ViT-Small Head+MLP组合剪枝实验 - ImageNet预训练版
# 使用预训练权重，减少训练时间

# 设置代理（访问HuggingFace下载预训练权重）
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

echo "=========================================="
echo "ViT-Small Head+MLP 组合剪枝实验"
echo "使用 ImageNet 预训练权重 ⚡"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  模型: vit_small (ImageNet预训练)"
echo "  剪枝目标: Head + MLP Neurons"
echo "  Criteria: magnitude"
echo "  Head剪枝率: 30%"
echo "  MLP剪枝率: 30%"
echo "  迭代次数: 20"
echo "  Epochs/迭代: 40 ⚡ (预训练加速)"
echo ""
echo "实验列表:"
echo "  1. CIFAR-10 + Refill"
echo "  2. CIFAR-10 + RSST"
echo "  3. CIFAR-100 + Refill"
echo "  4. CIFAR-100 + RSST"
echo ""
echo "=========================================="

# 创建目录
mkdir -p logs_vit_small_pretrained
mkdir -p output
mkdir -p init_model

# 通用参数
ARCH="vit_small"
PRETRAINED="--vit_pretrained"  # 使用预训练
VIT_STRUCTURED="--vit_structured"
PRUNE_TARGET="--vit_prune_target both"
CRITERIA="--criteria magnitude"
HEAD_RATE="--rate 0.3"
MLP_RATE="--mlp_prune_ratio 0.3"
PRUNING_TIMES="--pruning_times 20"
EPOCHS="--epochs 40"  # 减少到40（预训练加速）
BATCH_SIZE="--batch_size 128"
REG_GRANULARITY="--reg_granularity_prune 1.0"
RST_SCHEDULE="--RST_schedule exp_custom_exponents"
EXPONENTS="--exponents 4"

# 生成时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

echo ""
echo "==================== 开始启动实验 ===================="
echo ""

# ============================================================
# 实验1: CIFAR-10 + Refill
# ============================================================
echo "[1/4] 启动 CIFAR-10 + Refill 实验 (预训练)..."

EXP_NAME="vit_small_cifar10_refill_pretrained_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar10_refill_pretrained"
LOG_FILE="logs_vit_small_pretrained/cifar10_refill_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar10_pretrained_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $HEAD_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    --fillback_rate 0.0 \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID1=$!
echo "  ✓ 已启动，PID: $PID1"
echo "  ✓ 日志: $LOG_FILE"
echo "  ✓ 输出: $SAVE_DIR"
sleep 2

# ============================================================
# 实验2: CIFAR-10 + RSST
# ============================================================
echo ""
echo "[2/4] 启动 CIFAR-10 + RSST 实验 (预训练)..."

EXP_NAME="vit_small_cifar10_rsst_pretrained_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar10_rsst_pretrained"
LOG_FILE="logs_vit_small_pretrained/cifar10_rsst_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar10_pretrained_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $HEAD_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID2=$!
echo "  ✓ 已启动，PID: $PID2"
echo "  ✓ 日志: $LOG_FILE"
echo "  ✓ 输出: $SAVE_DIR"
sleep 2

# ============================================================
# 实验3: CIFAR-100 + Refill
# ============================================================
echo ""
echo "[3/4] 启动 CIFAR-100 + Refill 实验 (预训练)..."

EXP_NAME="vit_small_cifar100_refill_pretrained_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar100_refill_pretrained"
LOG_FILE="logs_vit_small_pretrained/cifar100_refill_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar100_pretrained_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --struct refill \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $HEAD_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    --fillback_rate 0.0 \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID3=$!
echo "  ✓ 已启动，PID: $PID3"
echo "  ✓ 日志: $LOG_FILE"
echo "  ✓ 输出: $SAVE_DIR"
sleep 2

# ============================================================
# 实验4: CIFAR-100 + RSST
# ============================================================
echo ""
echo "[4/4] 启动 CIFAR-100 + RSST 实验 (预训练)..."

EXP_NAME="vit_small_cifar100_rsst_pretrained_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar100_rsst_pretrained"
LOG_FILE="logs_vit_small_pretrained/cifar100_rsst_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar100_pretrained_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --struct rsst \
    $PRETRAINED \
    $VIT_STRUCTURED \
    $PRUNE_TARGET \
    $CRITERIA \
    $HEAD_RATE \
    $MLP_RATE \
    $PRUNING_TIMES \
    $EPOCHS \
    $BATCH_SIZE \
    $REG_GRANULARITY \
    $RST_SCHEDULE \
    $EXPONENTS \
    --init $INIT_FILE \
    --save_dir $SAVE_DIR \
    --exp_name $EXP_NAME \
    > $LOG_FILE 2>&1 &

PID4=$!
echo "  ✓ 已启动，PID: $PID4"
echo "  ✓ 日志: $LOG_FILE"
echo "  ✓ 输出: $SAVE_DIR"
sleep 2

# ============================================================
# 汇总信息
# ============================================================
echo ""
echo "=========================================="
echo "✅ 所有实验已启动！"
echo "=========================================="
echo ""
echo "进程列表:"
echo "  PID $PID1 - CIFAR-10 + Refill"
echo "  PID $PID2 - CIFAR-10 + RSST"
echo "  PID $PID3 - CIFAR-100 + Refill"
echo "  PID $PID4 - CIFAR-100 + RSST"
echo ""
echo "日志文件:"
echo "  logs_vit_small_pretrained/cifar10_refill_${TIMESTAMP}.log"
echo "  logs_vit_small_pretrained/cifar10_rsst_${TIMESTAMP}.log"
echo "  logs_vit_small_pretrained/cifar100_refill_${TIMESTAMP}.log"
echo "  logs_vit_small_pretrained/cifar100_rsst_${TIMESTAMP}.log"
echo ""
echo "配置:"
echo "  ✓ ImageNet预训练权重"
echo "  ✓ Epochs/迭代: 40 (减半加速)"
echo "  ✓ 迭代次数: 20"
echo "  ✓ 总Epochs: 800 per experiment"
echo ""
echo "预计完成时间:"
echo "  单个实验: ~2天"
echo "  全部完成: 2-3天"
echo ""
echo "查看实时日志:"
echo "  tail -f logs_vit_small_pretrained/cifar10_refill_${TIMESTAMP}.log"
echo ""
echo "查看运行状态:"
echo "  ps aux | grep 'vit_small'"
echo ""
echo "WandB项目地址: https://wandb.ai/ycx/RSST"
echo ""
echo "=========================================="

# 保存PID到文件
echo "$PID1 $PID2 $PID3 $PID4" > logs_vit_small_pretrained/experiment_pids_${TIMESTAMP}.txt
echo ""
echo "✓ 进程PID已保存到: logs_vit_small_pretrained/experiment_pids_${TIMESTAMP}.txt"
echo ""
