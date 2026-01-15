#!/bin/bash

#=============================================================================
# RSST 实验启动脚本 (配置文件版)
# 用法: ./run_with_config.sh [config_file]
# 示例: ./run_with_config.sh configs/default.conf
#=============================================================================

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: ./run_with_config.sh <配置文件>"
    echo "示例: ./run_with_config.sh configs/default.conf"
    echo ""
    echo "可用配置文件:"
    ls -1 configs/*.conf 2>/dev/null || echo "  (暂无配置文件)"
    exit 1
fi

CONFIG_FILE=$1

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "🚀 从配置文件启动RSST实验"
echo "=========================================="
echo "📄 配置文件: $CONFIG_FILE"
echo ""

# 加载配置
source $CONFIG_FILE

# 显示配置
echo "📋 配置内容:"
echo "  数据集: ${DATASET}"
echo "  模型: ${MODEL}"
echo "  预训练: ${PRETRAINED}"
echo "  Epochs: ${EPOCHS}"
echo "  剪枝率: ${RATE}"
echo "  正则化粒度: ${REG_GRANULARITY}"
echo "  指数曲率: ${EXPONENTS}"
if [ ! -z "$DESCRIPTION" ]; then
    echo "  描述: ${DESCRIPTION}"
fi
echo ""
echo "=========================================="
echo ""

# 询问确认
read -p "是否使用以上配置启动实验? (y/n) [y]: " confirm
confirm=${confirm:-y}
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ 已取消"
    exit 0
fi

# ============ 生成实验标识 ============
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PRETRAINED_TAG=""
if [ "$PRETRAINED" = "true" ]; then
    PRETRAINED_TAG="pretrained"
else
    PRETRAINED_TAG="scratch"
fi

SHORT_NAME="${ALGORITHM}_${MODEL}_${DATASET}_${PRETRAINED_TAG}_${EPOCHS}ep"
FULL_NAME="${ALGORITHM}_${MODEL}_${DATASET}_${PRETRAINED_TAG}_rate${RATE}_reg${REG_GRANULARITY}_exp${EXPONENTS}_${EPOCHS}ep"

# 实验目录
EXP_DIR="experiments/${TIMESTAMP}_${SHORT_NAME}"
mkdir -p ${EXP_DIR}/{logs,checkpoints,configs,results}

echo ""
echo "📁 实验目录: ${EXP_DIR}"

# ============ 复制配置文件到实验目录 ============
cp $CONFIG_FILE ${EXP_DIR}/configs/used_config.conf
echo "✓ 配置文件已备份"

# ============ 保存详细配置 ============
CONFIG_FILE_DETAIL="${EXP_DIR}/configs/experiment_config.txt"
cat > ${CONFIG_FILE_DETAIL} << EOF
===========================================
RSST 实验配置
===========================================
实验名称: ${FULL_NAME}
开始时间: $(date '+%Y-%m-%d %H:%M:%S')
实验目录: ${EXP_DIR}
配置文件: ${CONFIG_FILE}
$([ ! -z "$DESCRIPTION" ] && echo "描述: ${DESCRIPTION}")

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
    --epochs ${EPOCHS} \
    --pruning_times ${PRUNING_TIMES} \
    --rate ${RATE} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --decreasing_lr '${DECREASING_LR}' \
    --warmup ${WARMUP} \
    --criteria ${CRITERIA} \
    --RST_schedule ${RST_SCHEDULE} \
    --reg_granularity_prune ${REG_GRANULARITY} \
    --exponents ${EXPONENTS} \
    --save_dir ${EXP_DIR}/checkpoints \
    --exp_name '${FULL_NAME}'"

# 保存命令
echo "${TRAIN_CMD}" > ${EXP_DIR}/configs/command.sh
chmod +x ${EXP_DIR}/configs/command.sh

# ============ 启动训练 ============
echo ""
echo "=========================================="
echo "🏃 开始训练..."
echo "=========================================="

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
