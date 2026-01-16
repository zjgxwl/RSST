#!/bin/bash

# 测试Global混合排序模式
# 快速验证整个训练链路，确保不会中途挂掉

# 设置代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
echo "✓ 代理已设置: http://127.0.0.1:7897"
echo ""

echo "========================================"
echo "🧪 测试 Global 混合排序模式"
echo "========================================"
echo ""
echo "配置："
echo "  • 数据集: CIFAR-10"
echo "  • 迭代次数: 3次 (快速测试)"
echo "  • 每次迭代: 2个epoch"
echo "  • 排序模式: global (混合排序)"
echo "  • 剪枝率: 30%"
echo ""
echo "测试目标："
echo "  ✓ 验证State 0不会因KeyError崩溃"
echo "  ✓ 验证State 1-2的RSST正则化流程"
echo "  ✓ 验证global sorting逻辑正确"
echo ""
echo "========================================"
echo ""

# 创建目录
mkdir -p logs_test
mkdir -p checkpoint/test_global_sorting

# 生成时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

echo "[1/2] 测试 Refill + Global Sorting..."
echo ""

# 测试1: Refill + Global
nohup python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct refill \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.3 \
    --mlp_prune_ratio 0.3 \
    --pruning_times 3 \
    --epochs 2 \
    --batch_size 128 \
    --sorting_mode global \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --save_dir checkpoint/test_global_sorting/refill \
    --exp_name test_refill_global_${TIMESTAMP} \
    > logs_test/refill_global_${TIMESTAMP}.log 2>&1 &

PID1=$!
echo "  ✓ Refill进程已启动，PID: $PID1"
echo "  ✓ 日志: logs_test/refill_global_${TIMESTAMP}.log"
echo ""

# 等待Refill完成State 0
echo "等待Refill完成State 0..."
sleep 5

# 检查Refill是否还在运行
if ps -p $PID1 > /dev/null; then
    echo "  ✓ Refill正在运行"
else
    echo "  ✗ Refill已停止，检查日志..."
    tail -30 logs_test/refill_global_${TIMESTAMP}.log
    exit 1
fi

echo ""
echo "[2/2] 测试 RSST + Global Sorting..."
echo ""

# 测试2: RSST + Global
nohup python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct rsst \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.3 \
    --mlp_prune_ratio 0.3 \
    --pruning_times 3 \
    --epochs 2 \
    --batch_size 128 \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --sorting_mode global \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --save_dir checkpoint/test_global_sorting/rsst \
    --exp_name test_rsst_global_${TIMESTAMP} \
    > logs_test/rsst_global_${TIMESTAMP}.log 2>&1 &

PID2=$!
echo "  ✓ RSST进程已启动，PID: $PID2"
echo "  ✓ 日志: logs_test/rsst_global_${TIMESTAMP}.log"
echo ""

echo "========================================"
echo "✅ 测试已启动"
echo "========================================"
echo ""
echo "进程列表:"
echo "  PID $PID1 - Refill + Global"
echo "  PID $PID2 - RSST + Global"
echo ""
echo "实时监控:"
echo "  tail -f logs_test/refill_global_${TIMESTAMP}.log"
echo "  tail -f logs_test/rsst_global_${TIMESTAMP}.log"
echo ""
echo "预计完成时间: ~10-15分钟"
echo ""
echo "========================================"
echo ""

# 保存PID
echo "$PID1 $PID2" > logs_test/test_pids_${TIMESTAMP}.txt

# 监控脚本
echo "启动自动监控..."
echo ""

(
    sleep 120  # 等待2分钟（State 0应该完成）
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ 2分钟检查点"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 检查Refill
    if ps -p $PID1 > /dev/null; then
        echo "✓ Refill进程运行中 (PID: $PID1)"
        echo "  最新输出:"
        tail -5 logs_test/refill_global_${TIMESTAMP}.log | sed 's/^/    /'
    else
        echo "✗ Refill进程已停止 (PID: $PID1)"
        echo "  最后30行日志:"
        tail -30 logs_test/refill_global_${TIMESTAMP}.log | sed 's/^/    /'
    fi
    
    echo ""
    
    # 检查RSST
    if ps -p $PID2 > /dev/null; then
        echo "✓ RSST进程运行中 (PID: $PID2)"
        echo "  最新输出:"
        tail -5 logs_test/rsst_global_${TIMESTAMP}.log | sed 's/^/    /'
    else
        echo "✗ RSST进程已停止 (PID: $PID2)"
        echo "  最后30行日志:"
        tail -30 logs_test/rsst_global_${TIMESTAMP}.log | sed 's/^/    /'
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
) &

echo "监控进程已启动（2分钟后自动检查）"
echo ""
