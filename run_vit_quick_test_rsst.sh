#!/bin/bash

################################################################################
# ViT快速测试脚本 - RSST方法
################################################################################
# 
# 配置:
#   - 模型: vit_small (CIFAR-10)
#   - Epochs: 10 (每个state)
#   - Pruning times: 2 (只测试State 0 → State 1转换)
#   - 方法: RSST
#   - 目的: 快速验证修复有效性
#
################################################################################

# 实验配置
DATASET="cifar10"
MODEL="vit_small"
PRUNING_RATE=0.7
EPOCHS=10
PRUNING_TIMES=2
METHOD="rsst"
GPU=1  # 使用GPU 1，避免与Refill冲突

# 时间戳
TIMESTAMP=$(date +"%m%d_%H%M")

# 路径配置
DATA_PATH="data/cifar10"
INIT_FILE="init_model/vit_small_cifar10_pretrained_init.pth.tar"
SAVE_DIR="checkpoint/vit_quick_test_rsst_${TIMESTAMP}"
LOG_DIR="logs_vit_quick_test"
LOG_FILE="${LOG_DIR}/quick_test_rsst_${TIMESTAMP}.log"

# 创建目录
mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

echo "========================================================================"
echo "ViT快速测试 - RSST方法"
echo "========================================================================"
echo ""
echo "测试配置:"
echo "  模型: ${MODEL}"
echo "  数据集: ${DATASET}"
echo "  剪枝率: ${PRUNING_RATE}"
echo "  每个State的Epochs: ${EPOCHS}"
echo "  State数量: ${PRUNING_TIMES} (State 0 → State 1)"
echo "  方法: ${METHOD} (Regularized Structured Sparse Training)"
echo "  GPU: ${GPU}"
echo ""
echo "关键测试点:"
echo "  ✓ State 0训练完成"
echo "  ✓ RSST正则化应用"
echo "  ✓ 剪枝操作完成"
echo "  ✓ State 1开始（无RuntimeError）← 关键！"
echo "  ✓ State 1训练正常进行"
echo ""
echo "日志文件: ${LOG_FILE}"
echo "保存目录: ${SAVE_DIR}"
echo ""
echo "========================================================================"
echo "开始测试..."
echo "========================================================================"
echo ""

# 检查初始化文件是否存在
if [ ! -f "${INIT_FILE}" ]; then
    echo "❌ 错误: 初始化文件不存在: ${INIT_FILE}"
    echo "   请先运行: python generate_pretrained_init.py"
    exit 1
fi

# 运行测试
CUDA_VISIBLE_DEVICES=${GPU} /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch ${MODEL} \
    --dataset ${DATASET} \
    --data ${DATA_PATH} \
    --init ${INIT_FILE} \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --struct ${METHOD} \
    --rate ${PRUNING_RATE} \
    --mlp_prune_ratio ${PRUNING_RATE} \
    --epochs ${EPOCHS} \
    --pruning_times ${PRUNING_TIMES} \
    --save_dir ${SAVE_DIR} \
    --exp_name "quick_test_rsst_${TIMESTAMP}" \
    --print_freq 50 \
    2>&1 | tee ${LOG_FILE}

# 检查退出状态
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================================"
echo "测试完成"
echo "========================================================================"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ 测试成功完成！"
    echo ""
    echo "关键检查点:"
    
    # 检查State 0完成
    if grep -q "pruning state 0" ${LOG_FILE}; then
        echo "  ✓ State 0训练开始"
    fi
    
    # 检查RSST正则化
    if grep -q "RSST.*regularization\|reg.*lambda" ${LOG_FILE}; then
        echo "  ✓ RSST正则化应用"
    fi
    
    # 检查剪枝完成
    if grep -q "Pruning completed\|ViT Pruning" ${LOG_FILE}; then
        echo "  ✓ 剪枝操作完成"
    fi
    
    # 检查State 1开始
    if grep -q "pruning state 1" ${LOG_FILE}; then
        echo "  ✓ State 1训练开始 ← 关键！没有RuntimeError！"
    fi
    
    # 检查是否有设备错误
    if grep -q "RuntimeError.*device\|Expected all tensors to be on the same device" ${LOG_FILE}; then
        echo "  ❌ 发现设备错误！修复可能不完整"
        echo ""
        echo "错误信息:"
        grep -A 5 "RuntimeError" ${LOG_FILE}
    else
        echo "  ✓ 无设备不匹配错误"
    fi
    
    echo ""
    echo "查看完整日志:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "查看训练进度:"
    echo "  grep -E 'Epoch.*\[.*\]|pruning state|best SA' ${LOG_FILE} | tail -20"
    echo ""
    
else
    echo "❌ 测试失败！退出码: ${EXIT_CODE}"
    echo ""
    echo "查看错误信息:"
    echo "  tail -50 ${LOG_FILE}"
    echo ""
    echo "查看设备相关错误:"
    echo "  grep -i 'error\|device\|RuntimeError' ${LOG_FILE}"
    echo ""
fi

echo "========================================================================"
