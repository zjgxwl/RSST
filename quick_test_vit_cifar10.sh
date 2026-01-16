#!/bin/bash
# 快速测试ViT + RSST在CIFAR-10上的流程
# 用最小配置验证流程是否通畅

echo "=========================================="
echo "ViT + RSST on CIFAR-10 快速流程测试"
echo "=========================================="
echo ""
echo "配置说明:"
echo "  - 数据集: CIFAR-10 (自动下载)"
echo "  - 模型: ViT-Tiny + ImageNet预训练"
echo "  - 剪枝方法: RSST"
echo "  - 剪枝次数: 2轮 (快速测试)"
echo "  - 每轮epoch: 2 (快速测试)"
echo "  - Batch size: 64"
echo "  - 预计时间: 5-10分钟"
echo ""
echo "⚠️  注意: 这只是流程测试，不追求最终精度"
echo "   正式实验请使用更多epoch和剪枝次数"
echo ""
echo "=========================================="

# 创建输出目录
mkdir -p test_output

# 运行快速测试
python main_imp_fillback.py \
    --dataset cifar10 \
    --data data \
    --arch vit_tiny \
    --pretrained \
    --struct rsst \
    --criteria l1 \
    --epochs 2 \
    --batch_size 64 \
    --lr 0.001 \
    --warmup 1 \
    --decreasing_lr 1 \
    --pruning_times 2 \
    --rate 0.2 \
    --prune_type lt \
    --RST_schedule exp_custom_exponents \
    --reg_granularity_prune 0.5 \
    --exponents 3 \
    --seed 42 \
    --gpu 0 \
    --save_dir test_output/vit_rsst_flow_test

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 流程测试通过！"
    echo "=========================================="
    echo ""
    echo "现在可以运行正式实验了:"
    echo ""
    echo "# 正式训练 (80 epochs × 15轮剪枝)"
    echo "python main_imp_fillback.py \\"
    echo "    --dataset cifar10 \\"
    echo "    --arch vit_tiny \\"
    echo "    --pretrained \\"
    echo "    --struct rsst \\"
    echo "    --epochs 80 \\"
    echo "    --batch_size 128 \\"
    echo "    --lr 0.001 \\"
    echo "    --pruning_times 15 \\"
    echo "    --rate 0.15 \\"
    echo "    --save_dir results/vit_tiny_cifar10_rsst"
    echo ""
else
    echo "❌ 流程测试失败！"
    echo "=========================================="
    echo ""
    echo "请检查错误信息并修复问题"
fi

exit $EXIT_CODE

