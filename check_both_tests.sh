#!/bin/bash

################################################################################
# 统一监控脚本 - 同时监控Refill和RSST测试
################################################################################

# 查找最新的日志文件
REFILL_LOG=$(ls -t logs_vit_quick_test/quick_test_0*.log 2>/dev/null | grep -v rsst | head -1)
RSST_LOG=$(ls -t logs_vit_quick_test/quick_test_rsst_*.log 2>/dev/null | head -1)

echo "========================================================================"
echo "ViT快速测试 - 双方法同步监控"
echo "========================================================================"
echo ""

# 检查GPU使用情况
echo "📊 GPU状态:"
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name util mem_used mem_total temp; do
    echo "GPU ${idx}: ${name}, 利用率: ${util}%, 显存: ${mem_used}MB/${mem_total}MB, 温度: ${temp}°C"
done
echo ""

# 检查进程状态
echo "🔄 进程状态:"
echo "------------------------------------------------------------------------"
REFILL_RUNNING=0
RSST_RUNNING=0

if ps aux | grep -v grep | grep "main_imp_fillback.py.*quick_test.*0[01]" > /dev/null; then
    echo "✓ Refill测试进程运行中"
    REFILL_RUNNING=1
else
    echo "⏸  Refill测试进程已停止"
fi

if ps aux | grep -v grep | grep "main_imp_fillback.py.*quick_test_rsst" > /dev/null; then
    echo "✓ RSST测试进程运行中"
    RSST_RUNNING=1
else
    echo "⏸  RSST测试进程已停止"
fi

echo ""
echo "========================================================================"
echo "📋 Refill测试"
echo "========================================================================"

if [ -f "${REFILL_LOG}" ]; then
    echo "日志: ${REFILL_LOG}"
    echo ""
    
    # 当前State
    CURRENT_STATE=$(grep "pruning state" ${REFILL_LOG} | tail -1)
    if [ ! -z "${CURRENT_STATE}" ]; then
        echo "当前State: ${CURRENT_STATE}"
    fi
    
    # 最近训练
    echo ""
    echo "最近训练记录:"
    grep -E "Epoch: \[[0-9]+\]\[[0-9]+/352\]" ${REFILL_LOG} | tail -5
    
    # 测试准确率
    TEST_ACC=$(grep -E "Test:.*Accuracy" ${REFILL_LOG} | tail -1)
    if [ ! -z "${TEST_ACC}" ]; then
        echo ""
        echo "最新测试: ${TEST_ACC}"
    fi
    
    # 最佳准确率
    BEST_SA=$(grep "best SA=" ${REFILL_LOG} | tail -1)
    if [ ! -z "${BEST_SA}" ]; then
        echo "最佳准确率: ${BEST_SA}"
    fi
    
    # 检查关键点
    echo ""
    echo "检查点:"
    if grep -q "pruning state 0" ${REFILL_LOG}; then
        echo "  ✓ State 0已开始"
    fi
    if grep -q "ViT Pruning" ${REFILL_LOG}; then
        echo "  ✓ 剪枝已执行"
    fi
    if grep -q "pruning state 1" ${REFILL_LOG}; then
        echo "  ✓ State 1已开始 ← 关键！"
    fi
    
    # 设备错误检查
    if grep -q "RuntimeError.*device\|Expected all tensors to be on the same device" ${REFILL_LOG}; then
        echo "  ❌ 发现设备错误！"
    else
        echo "  ✓ 无设备错误"
    fi
else
    echo "❌ 未找到Refill日志文件"
fi

echo ""
echo "========================================================================"
echo "📋 RSST测试"
echo "========================================================================"

if [ -f "${RSST_LOG}" ]; then
    echo "日志: ${RSST_LOG}"
    echo ""
    
    # 当前State
    CURRENT_STATE=$(grep "pruning state" ${RSST_LOG} | tail -1)
    if [ ! -z "${CURRENT_STATE}" ]; then
        echo "当前State: ${CURRENT_STATE}"
    fi
    
    # 最近训练
    echo ""
    echo "最近训练记录:"
    grep -E "Epoch: \[[0-9]+\]\[[0-9]+/352\]" ${RSST_LOG} | tail -5
    
    # 测试准确率
    TEST_ACC=$(grep -E "Test:.*Accuracy" ${RSST_LOG} | tail -1)
    if [ ! -z "${TEST_ACC}" ]; then
        echo ""
        echo "最新测试: ${TEST_ACC}"
    fi
    
    # 最佳准确率
    BEST_SA=$(grep "best SA=" ${RSST_LOG} | tail -1)
    if [ ! -z "${BEST_SA}" ]; then
        echo "最佳准确率: ${BEST_SA}"
    fi
    
    # 检查关键点
    echo ""
    echo "检查点:"
    if grep -q "pruning state 0" ${RSST_LOG}; then
        echo "  ✓ State 0已开始"
    fi
    if grep -q "ViT Pruning" ${RSST_LOG}; then
        echo "  ✓ 剪枝已执行"
    fi
    if grep -q "pruning state 1" ${RSST_LOG}; then
        echo "  ✓ State 1已开始 ← 关键！"
    fi
    
    # 设备错误检查
    if grep -q "RuntimeError.*device\|Expected all tensors to be on the same device" ${RSST_LOG}; then
        echo "  ❌ 发现设备错误！"
    else
        echo "  ✓ 无设备错误"
    fi
else
    echo "❌ 未找到RSST日志文件"
fi

echo ""
echo "========================================================================"
echo "📌 快速命令"
echo "========================================================================"
echo ""
echo "# 实时监控Refill"
echo "tail -f ${REFILL_LOG}"
echo ""
echo "# 实时监控RSST"
echo "tail -f ${RSST_LOG}"
echo ""
echo "# 对比两个方法的进度"
echo "watch -n 10 './check_both_tests.sh'"
echo ""
echo "# 查看GPU使用"
echo "watch -n 2 nvidia-smi"
echo ""
echo "========================================================================"
