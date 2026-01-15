#!/bin/bash

#=============================================================================
# RSST 实验启动脚本 (灵活版)
# 用法: 
#   ./run_experiment.sh                    # 使用默认配置
#   ./run_experiment.sh cifar10 vit_tiny   # 指定数据集和模型
#   ./run_experiment.sh cifar10 vit_tiny 80 1.0 4  # 完整参数
#=============================================================================

# ============ 参数解析 (支持命令行传入) ============
DATASET=${1:-"cifar10"}           # 第1个参数: 数据集 (默认cifar10)
MODEL=${2:-"vit_tiny"}            # 第2个参数: 模型 (默认vit_tiny)
EPOCHS=${3:-80}                   # 第3个参数: epochs (默认80)
REG_GRANULARITY=${4:-1.0}         # 第4个参数: 正则化粒度 (默认1.0)
EXPONENTS=${5:-4}                 # 第5个参数: 指数曲率 (默认4)
PRETRAINED=${6:-true}             # 第6个参数: 是否预训练 (默认true)
RATE=${7:-0.2}                    # 第7个参数: 剪枝率 (默认0.2)
AUTO_CONFIRM=${8:-"n"}            # 第8个参数: 自动确认 (y=跳过确认，默认n)
ALGORITHM=${9:-"rsst"}            # 第9个参数: 算法类型 (rsst/refill，默认rsst)
FILLBACK_RATE=${10:-0.2}          # 第10个参数: refill回填率 (默认0.2，仅refill使用)

# 固定参数（很少需要修改）
PRUNING_TIMES=16
BATCH_SIZE=128
LR=0.01
WARMUP=5
CRITERIA="l1"
RST_SCHEDULE="exp_custom_exponents"

# 根据epochs自动调整学习率衰减点
if [ "$EPOCHS" -le 60 ]; then
    DECREASING_LR="30,45"
elif [ "$EPOCHS" -le 100 ]; then
    DECREASING_LR="40,60"
else
    DECREASING_LR="91,136"
fi

# ============ 显示配置信息 ============
echo "=========================================="
echo "🚀 RSST 实验启动脚本"
echo "=========================================="
echo ""
echo "📋 当前配置:"
echo "  数据集: ${DATASET}"
echo "  模型: ${MODEL}"
echo "  算法: ${ALGORITHM}"
if [ "$ALGORITHM" = "refill" ]; then
    echo "  回填率: ${FILLBACK_RATE}"
fi
echo "  预训练: ${PRETRAINED}"
echo "  Epochs: ${EPOCHS}"
echo "  剪枝轮次: ${PRUNING_TIMES}"
echo "  剪枝率: ${RATE}"
echo "  正则化粒度: ${REG_GRANULARITY}"
echo "  指数曲率: ${EXPONENTS}"
echo "  学习率衰减: ${DECREASING_LR}"
echo ""
echo "💡 提示:"
echo "  修改配置: ./run_experiment.sh <dataset> <model> <epochs> <reg> <exp> <pretrained> <rate> <auto> <algorithm> <fillback>"
echo "  RSST示例: ./run_experiment.sh cifar10 vit_tiny 80 1.0 4 true 0.2 y rsst"
echo "  Refill示例: ./run_experiment.sh cifar10 vit_tiny 80 1.0 4 true 0.2 y refill 0.2"
echo "  (第8个参数 'y' 表示自动确认，第9个参数指定算法)"
echo "=========================================="
echo ""

# 询问是否继续（除非设置了自动确认）
if [ "$AUTO_CONFIRM" != "y" ] && [ "$AUTO_CONFIRM" != "Y" ]; then
    read -p "是否使用以上配置启动实验? (y/n) [y]: " confirm
    confirm=${confirm:-y}
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "❌ 已取消"
        exit 0
    fi
else
    echo "✅ 自动确认模式，直接启动..."
fi

# ============ 生成实验标识 ============
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PRETRAINED_TAG=""
if [ "$PRETRAINED" = "true" ]; then
    PRETRAINED_TAG="pretrained"
else
    PRETRAINED_TAG="scratch"
fi

# 实验名称（简短版，用于目录）
SHORT_NAME="${ALGORITHM}_${MODEL}_${DATASET}_${PRETRAINED_TAG}_${EPOCHS}ep"

# 实验名称（完整版，用于WandB）
FULL_NAME="${ALGORITHM}_${MODEL}_${DATASET}_${PRETRAINED_TAG}_rate${RATE}_reg${REG_GRANULARITY}_exp${EXPONENTS}_${EPOCHS}ep"

# 实验目录
EXP_DIR="experiments/${TIMESTAMP}_${SHORT_NAME}"
mkdir -p ${EXP_DIR}/{logs,checkpoints,configs,results}

echo ""
echo "📁 实验目录: ${EXP_DIR}"

# ============ 保存配置信息 ============
CONFIG_FILE="${EXP_DIR}/configs/experiment_config.txt"
cat > ${CONFIG_FILE} << EOF
===========================================
RSST 实验配置
===========================================
实验名称: ${FULL_NAME}
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
实验目录: ${EXP_DIR}

--- 数据集和模型 ---
数据集: ${DATASET}
模型架构: ${MODEL}
预训练: ${PRETRAINED_TAG}

--- 训练参数 ---
Epochs: ${EPOCHS}
Batch Size: ${BATCH_SIZE}
学习率: ${LR}
学习率衰减: ${DECREASING_LR}
预热Epochs: ${WARMUP}

--- 剪枝参数 ---
算法: ${ALGORITHM}
剪枝轮次: ${PRUNING_TIMES}
剪枝率: ${RATE}

--- RSST参数 ---
重要性标准: ${CRITERIA}
正则化策略: ${RST_SCHEDULE}
正则化粒度: ${REG_GRANULARITY}
指数曲率: ${EXPONENTS}

--- 输出位置 ---
检查点目录: ${EXP_DIR}/checkpoints
日志目录: ${EXP_DIR}/logs
结果目录: ${EXP_DIR}/results
===========================================
EOF

# ============ 保存参数为JSON ============
ARGS_JSON="${EXP_DIR}/configs/args.json"
cat > ${ARGS_JSON} << EOF
{
  "experiment": {
    "name": "${FULL_NAME}",
    "timestamp": "${TIMESTAMP}",
    "directory": "${EXP_DIR}"
  },
  "dataset": {
    "name": "${DATASET}"
  },
  "model": {
    "architecture": "${MODEL}",
    "pretrained": ${PRETRAINED}
  },
  "training": {
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "learning_rate": ${LR},
    "decreasing_lr": "${DECREASING_LR}",
    "warmup": ${WARMUP}
  },
  "pruning": {
    "algorithm": "${ALGORITHM}",
    "pruning_times": ${PRUNING_TIMES},
    "rate": ${RATE}
  },
  "rsst": {
    "criteria": "${CRITERIA}",
    "schedule": "${RST_SCHEDULE}",
    "reg_granularity": ${REG_GRANULARITY},
    "exponents": ${EXPONENTS}
  }
}
EOF

# ============ 保存环境信息 ============
ENV_FILE="${EXP_DIR}/configs/environment.txt"
cat > ${ENV_FILE} << EOF
===========================================
运行环境信息
===========================================
主机名: $(hostname)
用户: $(whoami)
工作目录: $(pwd)
Python版本: $(python --version 2>&1)
PyTorch版本: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
CUDA版本: $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "无GPU")
===========================================
EOF

# ============ 构建训练命令 ============
PRETRAINED_FLAG=""
if [ "$PRETRAINED" = "true" ]; then
    PRETRAINED_FLAG="--vit_pretrained"
fi

TRAIN_CMD="python main_imp_fillback.py \
    --dataset ${DATASET} \
    --arch ${MODEL} \
    ${PRETRAINED_FLAG} \
    --struct ${ALGORITHM} \
    --fillback_rate ${FILLBACK_RATE} \
    --epochs ${EPOCHS} \
    --pruning_times ${PRUNING_TIMES} \
    --rate ${RATE} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --decreasing_lr ${DECREASING_LR} \
    --warmup ${WARMUP} \
    --criteria ${CRITERIA} \
    --RST_schedule ${RST_SCHEDULE} \
    --reg_granularity_prune ${REG_GRANULARITY} \
    --exponents ${EXPONENTS} \
    --save_dir ${EXP_DIR}/checkpoints \
    --exp_name '${FULL_NAME}'"

# 保存完整命令
echo "${TRAIN_CMD}" > ${EXP_DIR}/configs/command.sh
chmod +x ${EXP_DIR}/configs/command.sh

# ============ 启动训练 ============
echo ""
echo "=========================================="
echo "🏃 开始训练..."
echo "=========================================="

# 后台运行
nohup ${TRAIN_CMD} \
    > ${EXP_DIR}/logs/stdout.log \
    2> ${EXP_DIR}/logs/stderr.log &

PID=$!
echo ${PID} > ${EXP_DIR}/logs/training.pid

echo "✅ 训练已在后台启动"
echo "   进程ID: ${PID}"
echo ""
echo "📊 监控命令:"
echo "   tail -f ${EXP_DIR}/logs/stdout.log"
echo ""
echo "🛑 停止命令:"
echo "   kill ${PID}"
echo ""

# 创建快捷链接
ln -sfn ${EXP_DIR} experiments/latest
echo "🔗 快捷访问: experiments/latest"
echo ""
echo "=========================================="
echo "🎉 实验启动完成！"
echo "=========================================="
