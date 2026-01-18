#!/bin/bash
# Mamba-Base 70%剪枝实验 - RSST方法
# CIFAR-10和CIFAR-100，共2个实验

echo "========================================================="
echo "Mamba-Base 70% Pruning - RSST Method"
echo "CIFAR-10 and CIFAR-100"
echo "========================================================="

# 通用配置
ARCH="mamba_base"
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
LOG_DIR="logs_mamba_base_70p"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

# 检查是否只启动CIFAR-10（串行模式）
SEQUENTIAL_MODE=${1:-"parallel"}  # 默认并行，可传入"sequential"改为串行

if [ "$SEQUENTIAL_MODE" = "sequential" ]; then
    echo ""
    echo "⚠️  串行模式：只启动CIFAR-10，CIFAR-100需手动启动"
    echo ""
fi

# 实验1: CIFAR-10 + RSST (GPU 0)
echo ""
echo "实验1: CIFAR-10 + RSST (GPU 0)"
echo "---------------------------------------------------------"
DATASET="cifar10"
EXP_NAME="mamba_base_cifar10_rsst_70p_${TIMESTAMP}"

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_fillback.py \
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
    --init init_model/mamba_base_random_init.pth.tar \
    --exp_name $EXP_NAME \
    > $LOG_DIR/${EXP_NAME}.log 2>&1 &

PID1=$!
echo "已启动 PID: $PID1"

# 根据模式决定是否启动第二个实验
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    sleep 2
    
    # 实验2: CIFAR-100 + RSST (GPU 0)
    echo ""
    echo "实验2: CIFAR-100 + RSST (GPU 0)"
    echo "---------------------------------------------------------"
    DATASET="cifar100"
    EXP_NAME="mamba_base_cifar100_rsst_70p_${TIMESTAMP}"
    
    CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_fillback.py \
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
        --init init_model/mamba_base_random_init.pth.tar \
        --exp_name $EXP_NAME \
        > $LOG_DIR/${EXP_NAME}.log 2>&1 &
    
    PID2=$!
    echo "已启动 PID: $PID2"
else
    echo ""
    echo "⚠️  串行模式：CIFAR-100实验未启动"
    echo "   待CIFAR-10完成后，手动启动CIFAR-100："
    echo "   bash run_mamba_base_70p_rsst_cifar100.sh"
    PID2="未启动"
fi

echo ""
echo "========================================================="
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    echo "2个实验已启动（并行模式）"
else
    echo "1个实验已启动（串行模式）"
fi
echo "========================================================="
echo ""
echo "实验配置："
echo "  - 方法: RSST (Regularization-based Structured Sparse Training)"
echo "  - 模型: ${ARCH} (~86M参数)"
echo "  - 剪枝率: SSM ${PRUNE_RATE}, MLP ${MLP_PRUNE_RATE}"
echo "  - 剪枝目标: both (SSM + MLP)"
echo "  - 排序模式: ${SORTING_MODE} (全局混合排序)"
echo "  - 剪枝轮次: ${PRUNING_TIMES}"
echo "  - 训练轮次: ${EPOCHS}"
echo "  - 学习率: ${LR}"
echo "  - 批次大小: ${BATCH_SIZE}"
echo ""
echo "RSST特有参数："
echo "  - 正则化粒度: ${REG_GRANULARITY}"
echo "  - 正则化调度: ${RST_SCHEDULE}"
echo "  - 指数参数: ${EXPONENTS}"
echo ""
echo "GPU分配："
echo "  - GPU0: 所有实验"
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    echo "  - 模式: 并行（CIFAR-10和CIFAR-100同时运行）"
else
    echo "  - 模式: 串行（先CIFAR-10，后CIFAR-100）"
fi
echo ""
echo "进程ID："
echo "  - CIFAR-10 RSST: $PID1"
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    echo "  - CIFAR-100 RSST: $PID2"
fi
echo ""
echo "日志文件："
echo "  $LOG_DIR/mamba_base_cifar10_rsst_70p_${TIMESTAMP}.log"
echo "  $LOG_DIR/mamba_base_cifar100_rsst_70p_${TIMESTAMP}.log"
echo ""
echo "监控命令："
echo "  # 查看所有日志"
echo "  tail -f $LOG_DIR/*_${TIMESTAMP}.log"
echo ""
echo "  # 查看CIFAR-10日志"
echo "  tail -f $LOG_DIR/mamba_base_cifar10_rsst_70p_${TIMESTAMP}.log"
echo ""
echo "  # 查看CIFAR-100日志"
echo "  tail -f $LOG_DIR/mamba_base_cifar100_rsst_70p_${TIMESTAMP}.log"
echo ""
echo "  # 查看进程状态"
echo "  ps aux | grep 'main_imp_fillback.py.*mamba_base'"
echo ""
echo "  # 查看GPU使用情况"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "  # 查看GPU 2和3的使用情况"
echo "  nvidia-smi -i 2,3"
echo ""
echo "========================================================="
echo ""
echo "⚠️  注意事项："
echo "  - Mamba-Base参数量大（~86M），显存占用约25-30GB/实验"
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    echo "  - 并行模式：2个实验同时运行，总显存约50-60GB"
    echo "  - 当前GPU (A800 80GB): 已用25GB + 需要50-60GB = 总计75-85GB"
    echo "  - ⚠️ 可能超过GPU容量！建议："
    echo "      1. 等待ViT实验完成后再启动（推荐）"
    echo "      2. 使用串行模式: bash run_mamba_base_70p_rsst.sh sequential"
else
    echo "  - 串行模式：1个实验，总显存约25-30GB"
    echo "  - 当前GPU (A800 80GB): 已用25GB + 需要25-30GB = 总计50-55GB ✓ 可行"
fi
echo "  - 预计完成时间:"
if [ "$SEQUENTIAL_MODE" = "parallel" ]; then
    echo "      并行: ~40-50小时（16 states × 2.5-3小时/state）"
else
    echo "      串行: ~80-100小时（2个实验 × 16 states × 2.5-3小时/state）"
fi
echo "  - RSST方法比Refill略慢（需要额外的正则化计算）"
echo ""
echo "当前GPU状态："
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU%s: %s - 已用%sMB / 总计%sMB (%.1f%%)\n", $1, $2, $3, $4, ($3/$4)*100}'
echo ""
echo "运行模式选择："
echo "  - 并行模式（默认）: bash run_mamba_base_70p_rsst.sh"
echo "  - 串行模式（推荐）: bash run_mamba_base_70p_rsst.sh sequential"
echo ""
echo "========================================================="
