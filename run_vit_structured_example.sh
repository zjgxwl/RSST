#!/bin/bash
# ViT结构化剪枝示例脚本
# 快速启动不同配置的结构化剪枝实验

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ViT结构化剪枝实验启动脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法："
    echo "  $0 <实验编号>"
    echo ""
    echo "可用实验："
    echo "  1 - CIFAR-10 + ViT-Tiny + 50%剪枝 (基准实验)"
    echo "  2 - CIFAR-100 + ViT-Tiny + 50%剪枝"
    echo "  3 - CIFAR-10 + ViT-Small + 33%剪枝 (需要预训练)"
    echo "  4 - 对比实验：结构化 vs 非结构化"
    echo "  5 - 使用L2 criteria + 高剪枝率"
    echo "  6 - 自定义实验（交互式配置）"
    echo ""
    exit 1
fi

EXPERIMENT=$1

case $EXPERIMENT in
    1)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验1: CIFAR-10 + ViT-Tiny + 50%剪枝 (基准实验)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "配置："
        echo "  • 数据集: CIFAR-10"
        echo "  • 模型: ViT-Tiny (3 heads)"
        echo "  • 剪枝率: 50% (3→2 heads)"
        echo "  • Criteria: magnitude"
        echo "  • 算法: RSST"
        echo "  • 训练轮数: 80"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python main_imp_fillback.py \
            --arch vit_tiny \
            --dataset cifar10 \
            --vit_structured \
            --struct rsst \
            --criteria magnitude \
            --rate 0.50 \
            --epochs 80 \
            --batch_size 128 \
            --lr 0.01 \
            --gpu 0 \
            --exp_name vit_tiny_cifar10_struct50
        ;;
        
    2)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验2: CIFAR-100 + ViT-Tiny + 50%剪枝"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "配置："
        echo "  • 数据集: CIFAR-100"
        echo "  • 模型: ViT-Tiny (3 heads)"
        echo "  • 剪枝率: 50% (3→2 heads)"
        echo "  • Criteria: magnitude"
        echo "  • 算法: RSST"
        echo "  • 训练轮数: 120"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python main_imp_fillback.py \
            --arch vit_tiny \
            --dataset cifar100 \
            --vit_structured \
            --struct rsst \
            --criteria magnitude \
            --rate 0.50 \
            --epochs 120 \
            --batch_size 128 \
            --lr 0.01 \
            --gpu 0 \
            --exp_name vit_tiny_cifar100_struct50
        ;;
        
    3)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验3: CIFAR-10 + ViT-Small + 33%剪枝 (预训练)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "配置："
        echo "  • 数据集: CIFAR-10"
        echo "  • 模型: ViT-Small (6 heads)"
        echo "  • 预训练: ImageNet"
        echo "  • 剪枝率: 33% (6→4 heads)"
        echo "  • Criteria: magnitude"
        echo "  • 算法: RSST"
        echo "  • 训练轮数: 60"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python main_imp_fillback.py \
            --arch vit_small \
            --dataset cifar10 \
            --vit_pretrained \
            --vit_structured \
            --struct rsst \
            --criteria magnitude \
            --rate 0.33 \
            --epochs 60 \
            --batch_size 128 \
            --lr 0.001 \
            --gpu 0 \
            --exp_name vit_small_cifar10_pretrain_struct33
        ;;
        
    4)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验4: 对比实验 - 结构化 vs 非结构化"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "将启动两个实验进行对比："
        echo ""
        echo "实验4a: 非结构化剪枝（85%稀疏度）"
        echo "  • Element-wise pruning"
        echo "  • 高稀疏度，但计算量不减少"
        echo ""
        
        python main_imp_fillback.py \
            --arch vit_tiny \
            --dataset cifar10 \
            --struct rsst \
            --criteria magnitude \
            --rate 0.85 \
            --epochs 80 \
            --batch_size 128 \
            --gpu 0 \
            --exp_name vit_unstructured_85
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验4b: 结构化剪枝（50% heads）"
        echo "  • Head-level pruning"
        echo "  • 真正减少计算量"
        echo ""
        
        python main_imp_fillback.py \
            --arch vit_tiny \
            --dataset cifar10 \
            --vit_structured \
            --struct rsst \
            --criteria magnitude \
            --rate 0.50 \
            --epochs 80 \
            --batch_size 128 \
            --gpu 0 \
            --exp_name vit_structured_50
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✓ 对比实验完成！请在WandB中对比两个实验的结果"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
        
    5)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验5: L2 Criteria + 高剪枝率（67%）"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "配置："
        echo "  • 数据集: CIFAR-10"
        echo "  • 模型: ViT-Tiny (3 heads)"
        echo "  • 剪枝率: 67% (3→1 head)"
        echo "  • Criteria: L2 范数"
        echo "  • 算法: Refill"
        echo "  • Fillback率: 0.1"
        echo "  • 训练轮数: 100"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python main_imp_fillback.py \
            --arch vit_tiny \
            --dataset cifar10 \
            --vit_structured \
            --struct refill \
            --fillback_rate 0.1 \
            --criteria l2 \
            --rate 0.67 \
            --epochs 100 \
            --batch_size 128 \
            --lr 0.01 \
            --gpu 0 \
            --exp_name vit_tiny_l2_struct67_refill
        ;;
        
    6)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "实验6: 自定义实验配置"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # 交互式配置
        read -p "选择模型 (vit_tiny/vit_small/vit_base): " ARCH
        ARCH=${ARCH:-vit_tiny}
        
        read -p "选择数据集 (cifar10/cifar100): " DATASET
        DATASET=${DATASET:-cifar10}
        
        read -p "剪枝率 (0.0-1.0, 推荐0.33-0.50): " RATE
        RATE=${RATE:-0.50}
        
        read -p "Criteria (magnitude/l1/l2/remain/saliency): " CRITERIA
        CRITERIA=${CRITERIA:-magnitude}
        
        read -p "算法 (rsst/refill): " STRUCT
        STRUCT=${STRUCT:-rsst}
        
        read -p "训练轮数: " EPOCHS
        EPOCHS=${EPOCHS:-80}
        
        read -p "是否使用预训练模型? (y/n): " PRETRAIN
        if [ "$PRETRAIN" = "y" ]; then
            PRETRAIN_FLAG="--vit_pretrained"
        else
            PRETRAIN_FLAG=""
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "启动自定义实验："
        echo "  模型: $ARCH"
        echo "  数据集: $DATASET"
        echo "  剪枝率: $RATE"
        echo "  Criteria: $CRITERIA"
        echo "  算法: $STRUCT"
        echo "  训练轮数: $EPOCHS"
        echo "  预训练: $PRETRAIN"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python main_imp_fillback.py \
            --arch $ARCH \
            --dataset $DATASET \
            --vit_structured \
            --struct $STRUCT \
            --criteria $CRITERIA \
            --rate $RATE \
            --epochs $EPOCHS \
            --batch_size 128 \
            --gpu 0 \
            $PRETRAIN_FLAG
        ;;
        
    *)
        echo "❌ 错误: 无效的实验编号 '$EXPERIMENT'"
        echo ""
        echo "可用实验编号: 1-6"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实验启动完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
