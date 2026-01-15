#!/bin/bash

# ViT-Small Head+MLP组合剪枝实验 - 最终版本
# 随机初始化 + 减少epochs到30（约1.5天完成）

echo "=========================================="
echo "ViT-Small Head+MLP 组合剪枝实验 - 最终版"
echo "=========================================="
echo ""
echo "⚠️  环境无外网访问，无法下载预训练权重"
echo "✓ 使用随机初始化"
echo "✓ 优化训练epochs: 30 per iteration"
echo ""
echo "配置信息:"
echo "  模型: vit_small (随机初始化)"
echo "  剪枝目标: Head + MLP Neurons"
echo "  Criteria: magnitude"
echo "  Head剪枝率: 30%"
echo "  MLP剪枝率: 30%"
echo "  迭代次数: 20"
echo "  Epochs/迭代: 30 (优化加速)"
echo ""
echo "实验列表:"
echo "  1. CIFAR-10 + Refill"
echo "  2. CIFAR-10 + RSST"
echo "  3. CIFAR-100 + Refill"
echo "  4. CIFAR-100 + RSST"
echo ""
echo "=========================================="

# 创建目录
mkdir -p logs_vit_small_final
mkdir -p output
mkdir -p init_model

# 通用参数
ARCH="vit_small"
VIT_STRUCTURED="--vit_structured"
PRUNE_TARGET="--vit_prune_target both"
CRITERIA="--criteria magnitude"
HEAD_RATE="--rate 0.3"
MLP_RATE="--mlp_prune_ratio 0.3"
PRUNING_TIMES="--pruning_times 20"
EPOCHS="--epochs 30"  # 减少到30
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
echo "[1/4] 启动 CIFAR-10 + Refill 实验..."

EXP_NAME="vit_small_cifar10_refill_final_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar10_refill_final"
LOG_FILE="logs_vit_small_final/cifar10_refill_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar10_init.pth.tar"

# 检查并生成初始化文件
if [ ! -f "$INIT_FILE" ]; then
    echo "  ⏳ 生成CIFAR-10初始化文件..."
    python -c "
import torch
import sys
sys.path.insert(0, '.')
from models.vit import vit_small
from utils import NormalizeByChannelMeanStd

model = vit_small(num_classes=10, img_size=32, pretrained=False)
normalize = NormalizeByChannelMeanStd(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]
)
model.normalize = normalize
torch.save({'state_dict': model.state_dict()}, '$INIT_FILE')
print('  ✓ CIFAR-10 初始化文件已生成')
" || { echo "生成初始化文件失败"; exit 1; }
fi

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --struct refill \
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
echo "[2/4] 启动 CIFAR-10 + RSST 实验..."

EXP_NAME="vit_small_cifar10_rsst_final_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar10_rsst_final"
LOG_FILE="logs_vit_small_final/cifar10_rsst_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar10_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --struct rsst \
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
echo "[3/4] 启动 CIFAR-100 + Refill 实验..."

EXP_NAME="vit_small_cifar100_refill_final_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar100_refill_final"
LOG_FILE="logs_vit_small_final/cifar100_refill_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar100_init.pth.tar"

# 检查并生成初始化文件
if [ ! -f "$INIT_FILE" ]; then
    echo "  ⏳ 生成CIFAR-100初始化文件..."
    python -c "
import torch
import sys
sys.path.insert(0, '.')
from models.vit import vit_small
from utils import NormalizeByChannelMeanStd

model = vit_small(num_classes=100, img_size=32, pretrained=False)
normalize = NormalizeByChannelMeanStd(
    mean=[0.5071, 0.4867, 0.4408],
    std=[0.2675, 0.2565, 0.2761]
)
model.normalize = normalize
torch.save({'state_dict': model.state_dict()}, '$INIT_FILE')
print('  ✓ CIFAR-100 初始化文件已生成')
" || { echo "生成初始化文件失败"; exit 1; }
fi

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --struct refill \
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
echo "[4/4] 启动 CIFAR-100 + RSST 实验..."

EXP_NAME="vit_small_cifar100_rsst_final_${TIMESTAMP}"
SAVE_DIR="output/vit_small_cifar100_rsst_final"
LOG_FILE="logs_vit_small_final/cifar100_rsst_${TIMESTAMP}.log"
INIT_FILE="init_model/vit_small_cifar100_init.pth.tar"

nohup python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --struct rsst \
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
echo "  logs_vit_small_final/cifar10_refill_${TIMESTAMP}.log"
echo "  logs_vit_small_final/cifar10_rsst_${TIMESTAMP}.log"
echo "  logs_vit_small_final/cifar100_refill_${TIMESTAMP}.log"
echo "  logs_vit_small_final/cifar100_rsst_${TIMESTAMP}.log"
echo ""
echo "配置:"
echo "  • 随机初始化（无预训练）"
echo "  • Epochs/迭代: 30 (优化)"
echo "  • 迭代次数: 20"
echo "  • 总Epochs: 600 per experiment"
echo ""
echo "预计完成时间:"
echo "  单个实验: ~1.5天"
echo "  全部完成: 1.5-2天"
echo ""
echo "查看实时日志:"
echo "  tail -f logs_vit_small_final/cifar10_refill_${TIMESTAMP}.log"
echo ""
echo "查看运行状态:"
echo "  ps aux | grep 'vit_small'"
echo ""
echo "停止所有实验:"
echo "  ps aux | grep 'vit_small.*main_imp_fillback' | grep -v grep | awk '{print \$2}' | xargs kill"
echo ""
echo "WandB项目地址: https://wandb.ai/ycx/RSST"
echo ""
echo "=========================================="

# 保存PID到文件
echo "$PID1 $PID2 $PID3 $PID4" > logs_vit_small_final/experiment_pids_${TIMESTAMP}.txt
echo ""
echo "✓ 进程PID已保存到: logs_vit_small_final/experiment_pids_${TIMESTAMP}.txt"
echo ""
