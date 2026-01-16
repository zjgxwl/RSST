#!/bin/bash

# 简化的RSST速度测试 - 只测试正常训练，不做剪枝
# 测试优化后的训练速度

echo "================================"
echo "测试优化后的训练速度（1 epoch）"
echo "================================"

# 只训练1个epoch，不做剪枝（pruning_times=0）
python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct rsst \
    --rate 0.7 \
    --prune_type rewind_lt \
    --rewind_epoch 2 \
    --epochs 1 \
    --pruning_times 0 \
    --save_dir checkpoint/vit_small_cifar10_speed_test \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --RST_schedule exp_custom_exponents \
    --reg_granularity_prune 1e-5 \
    --exponents 4 \
    --criteria magnitude \
    --vit_pretrained \
    2>&1 | tee logs_vit_small/speed_test_simple_$(date +%m%d_%H%M).log

echo ""
echo "================================"
echo "测试完成！"
echo "================================"
