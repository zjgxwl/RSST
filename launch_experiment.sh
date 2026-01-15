#!/bin/bash
# 实验启动辅助脚本
# 自动将启动日志保存到 launch_logs/ 目录

# 确保 launch_logs 目录存在
mkdir -p launch_logs

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 获取所有参数
DATASET=${1:-"cifar10"}
MODEL=${2:-"vit_tiny"}
ALGORITHM=${9:-"rsst"}

# 生成日志文件名
LOG_NAME="${DATASET}_${ALGORITHM}_${TIMESTAMP}.log"
LOG_PATH="launch_logs/${LOG_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动实验并保存日志"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 参数:"
echo "   数据集: $DATASET"
echo "   模型: $MODEL"
echo "   算法: $ALGORITHM"
echo ""
echo "📁 启动日志: $LOG_PATH"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 启动实验并保存日志
nohup ./run_experiment.sh "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "✅ 实验已启动"
echo "   进程ID: $PID"
echo "   启动日志: $LOG_PATH"
echo ""
echo "🔍 查看启动日志:"
echo "   cat $LOG_PATH"
echo ""
echo "📊 查看训练日志:"
echo "   # 等待几秒后执行，让实验创建目录"
echo "   tail -f experiments/latest/logs/stdout.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
