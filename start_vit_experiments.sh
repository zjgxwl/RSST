#!/bin/bash

################################################################################
# 手动启动ViT实验 - 更可控的方式
################################################################################

TIMESTAMP=$(date +%m%d_%H%M)

echo "========================================================================"
echo "🚀 启动ViT-Small 70%剪枝实验（手动模式）"
echo "========================================================================"
echo ""

# 创建目录
mkdir -p logs_vit_small_70p
mkdir -p checkpoint/vit_small_70p

# ============================================================================
# 实验1: CIFAR-10 Refill [GPU 0]
# ============================================================================

echo "启动实验1: CIFAR-10 + Refill [GPU 0]"

CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct refill \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.7 \
    --mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --batch_size 128 \
    --sorting_mode global \
    --lr 0.01 \
    --fillback_rate 0.0 \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --save_dir checkpoint/vit_small_70p/cifar10_refill \
    --exp_name cifar10_refill_70p_${TIMESTAMP} \
    > logs_vit_small_70p/cifar10_refill_70p_${TIMESTAMP}.log 2>&1 &

PID1=$!
echo "  ✓ PID: $PID1"
sleep 3

# ============================================================================
# 实验2: CIFAR-10 RSST [GPU 0]
# ============================================================================

echo "启动实验2: CIFAR-10 + RSST [GPU 0]"

CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar10 \
    --data data/cifar10 \
    --struct rsst \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.7 \
    --mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --batch_size 128 \
    --sorting_mode global \
    --lr 0.01 \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --init init_model/vit_small_cifar10_pretrained_init.pth.tar \
    --save_dir checkpoint/vit_small_70p/cifar10_rsst \
    --exp_name cifar10_rsst_70p_${TIMESTAMP} \
    > logs_vit_small_70p/cifar10_rsst_70p_${TIMESTAMP}.log 2>&1 &

PID2=$!
echo "  ✓ PID: $PID2"
sleep 3

# ============================================================================
# 实验3: CIFAR-100 Refill [GPU 1]
# ============================================================================

echo "启动实验3: CIFAR-100 + Refill [GPU 1]"

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --data data/cifar100 \
    --struct refill \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.7 \
    --mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --batch_size 128 \
    --sorting_mode global \
    --lr 0.01 \
    --fillback_rate 0.0 \
    --init init_model/vit_small_cifar100_pretrained_init.pth.tar \
    --save_dir checkpoint/vit_small_70p/cifar100_refill \
    --exp_name cifar100_refill_70p_${TIMESTAMP} \
    > logs_vit_small_70p/cifar100_refill_70p_${TIMESTAMP}.log 2>&1 &

PID3=$!
echo "  ✓ PID: $PID3"
sleep 3

# ============================================================================
# 实验4: CIFAR-100 RSST [GPU 1]
# ============================================================================

echo "启动实验4: CIFAR-100 + RSST [GPU 1]"

CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/envs/structlth/bin/python -u main_imp_fillback.py \
    --arch vit_small \
    --dataset cifar100 \
    --data data/cifar100 \
    --struct rsst \
    --vit_pretrained \
    --vit_structured \
    --vit_prune_target both \
    --criteria magnitude \
    --rate 0.7 \
    --mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --batch_size 128 \
    --sorting_mode global \
    --lr 0.01 \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --init init_model/vit_small_cifar100_pretrained_init.pth.tar \
    --save_dir checkpoint/vit_small_70p/cifar100_rsst \
    --exp_name cifar100_rsst_70p_${TIMESTAMP} \
    > logs_vit_small_70p/cifar100_rsst_70p_${TIMESTAMP}.log 2>&1 &

PID4=$!
echo "  ✓ PID: $PID4"

echo ""
echo "========================================================================"
echo "✅ 全部4个实验已启动"
echo "========================================================================"
echo ""
echo "进程ID:"
echo "  实验1 (CIFAR-10 Refill):  $PID1"
echo "  实验2 (CIFAR-10 RSST):    $PID2"
echo "  实验3 (CIFAR-100 Refill): $PID3"
echo "  实验4 (CIFAR-100 RSST):   $PID4"
echo ""
echo "日志文件:"
echo "  logs_vit_small_70p/cifar10_refill_70p_${TIMESTAMP}.log"
echo "  logs_vit_small_70p/cifar10_rsst_70p_${TIMESTAMP}.log"
echo "  logs_vit_small_70p/cifar100_refill_70p_${TIMESTAMP}.log"
echo "  logs_vit_small_70p/cifar100_rsst_70p_${TIMESTAMP}.log"
echo ""
echo "监控命令:"
echo "  ps aux | grep main_imp_fillback | grep -v grep"
echo "  nvidia-smi"
echo "  tail -f logs_vit_small_70p/*.log"
echo ""
echo "========================================================================"
