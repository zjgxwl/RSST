#!/bin/bash
# 检查实验的剪枝率脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 实验剪枝率详细信息"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 查找所有实验目录
for exp_dir in experiments/2026*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="${exp_dir}/logs/stdout.log"
        
        if [ -f "$log_file" ]; then
            # 提取实验名称的关键信息
            dataset=$(echo "$exp_name" | grep -oP '(cifar10|cifar100|imagenet)')
            algorithm=$(echo "$exp_name" | grep -oP '(rsst|refill)')
            
            echo "【${dataset^^} + ${algorithm^^}】"
            echo "   路径: $exp_name"
            
            # 查找当前剪枝轮次
            pruning_state=$(grep "pruning state" "$log_file" | tail -1 | grep -oP 'pruning state \K\d+')
            if [ -n "$pruning_state" ]; then
                echo "   当前剪枝轮次: $pruning_state"
            else
                echo "   当前剪枝轮次: 0 (未开始剪枝)"
            fi
            
            # 提取最后一次的稀疏度信息
            last_sparsity=$(grep -E "Sparsity:" "$log_file" | tail -30 | grep -oP 'Sparsity:\s+\K[\d.]+' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "0.00"}')
            echo "   平均稀疏度: ${last_sparsity}%"
            
            # 计算剩余权重比例
            remain_weight=$(awk "BEGIN {printf \"%.2f\", 100 - $last_sparsity}")
            echo "   剩余权重: ${remain_weight}%"
            
            # 查找当前epoch
            current_epoch=$(grep -E "Epoch: \[" "$log_file" | tail -1 | grep -oP 'Epoch: \[\K\d+')
            if [ -n "$current_epoch" ]; then
                echo "   当前Epoch: $current_epoch"
            fi
            
            echo ""
        fi
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 说明:"
echo "   • 配置剪枝率: 每次迭代剪枝的比例（默认20%）"
echo "   • 平均稀疏度: 当前模型中被剪枝的权重比例"
echo "   • 剩余权重: 100% - 平均稀疏度"
echo ""
echo "💡 剪枝计算公式:"
echo "   第n轮后剩余权重 ≈ (1 - rate)^n"
echo "   例如: rate=0.2, 经过5轮后剩余 ≈ (1-0.2)^5 = 32.77%"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
