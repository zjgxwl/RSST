#!/bin/bash

# 测试RSST方法训练速度 - 只跑1个epoch
# 用于测试优化后的训练速度

echo "================================"
echo "测试RSST方法训练速度（1 epoch）"
echo "================================"

# CIFAR-10 RSST测试
python main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct rsst \
    --rate 0.7 \
    --prune_type rewind_lt \
    --rewind_epoch 2 \
    --epochs 1 \
    --save_dir checkpoint/vit_small_cifar10_rsst_speed_test \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --RST_schedule exp_custom_exponents \
    --reg_granularity_prune 1e-5 \
    --exponents 4 \
    --criteria magnitude \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --mlp_prune_ratio 0.3 \
    --sorting_mode layer-wise \
    2>&1 | tee logs_vit_small/speed_test_$(date +%m%d_%H%M).log

echo ""
echo "================================"
echo "测试完成！"
echo "================================"
