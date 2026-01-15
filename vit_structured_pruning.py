"""
ViT结构化剪枝工具
支持Attention Head级别和MLP Neuron级别的结构化剪枝

与ResNet的准结构化剪枝保持一致的接口设计
支持5种criteria: remain, magnitude, l1, l2, saliency
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Tuple


def is_vit_model(model):
    """检查模型是否为ViT"""
    return hasattr(model, 'blocks') and hasattr(model, 'patch_embed')


# ============================================================================
# Head重要性计算函数 (5种Criteria)
# ============================================================================

def compute_head_importance_remain(model, current_mask: Dict) -> Dict[int, torch.Tensor]:
    """
    Remain Criteria: 基于当前mask中head的非零权重数
    
    Args:
        model: ViT模型
        current_mask: 当前的剪枝mask字典
    
    Returns:
        head_importance: {layer_idx: importance_tensor [num_heads]}
    """
    head_importance = {}
    
    for layer_idx, block in enumerate(model.blocks):
        mask_key = f'blocks.{layer_idx}.attn.qkv.weight_mask'
        
        if mask_key not in current_mask:
            print(f"Warning: {mask_key} not found in current_mask, skipping")
            continue
        
        qkv_mask = current_mask[mask_key]
        
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        embed_dim = num_heads * head_dim
        
        # 重塑为 [3, num_heads, head_dim, embed_dim]
        qkv_mask = qkv_mask.view(3, num_heads, head_dim, embed_dim)
        
        # 计算每个head的非零权重数量 [num_heads]
        importance = qkv_mask.sum(dim=[0, 2, 3])
        
        head_importance[layer_idx] = importance
    
    return head_importance


def compute_head_importance_magnitude(model, trained_weight: Dict) -> Dict[int, torch.Tensor]:
    """
    Magnitude/L1 Criteria: 基于训练后权重的绝对值总和
    
    Args:
        model: ViT模型
        trained_weight: 训练后的权重字典
    
    Returns:
        head_importance: {layer_idx: importance_tensor [num_heads]}
    """
    head_importance = {}
    
    for layer_idx, block in enumerate(model.blocks):
        weight_key = f'blocks.{layer_idx}.attn.qkv.weight'
        
        if weight_key not in trained_weight:
            print(f"Warning: {weight_key} not found in trained_weight, skipping")
            continue
        
        qkv_weight = trained_weight[weight_key]
        
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        embed_dim = num_heads * head_dim
        
        # 重塑为 [3, num_heads, head_dim, embed_dim]
        qkv_weight = qkv_weight.view(3, num_heads, head_dim, embed_dim)
        
        # 计算每个head的权重绝对值总和 [num_heads]
        importance = qkv_weight.abs().sum(dim=[0, 2, 3])
        
        head_importance[layer_idx] = importance
    
    return head_importance


def compute_head_importance_l2(model, trained_weight: Dict) -> Dict[int, torch.Tensor]:
    """
    L2 Criteria: 基于权重的L2范数（权重平方和）
    
    Args:
        model: ViT模型
        trained_weight: 训练后的权重字典
    
    Returns:
        head_importance: {layer_idx: importance_tensor [num_heads]}
    """
    head_importance = {}
    
    for layer_idx, block in enumerate(model.blocks):
        weight_key = f'blocks.{layer_idx}.attn.qkv.weight'
        
        if weight_key not in trained_weight:
            print(f"Warning: {weight_key} not found in trained_weight, skipping")
            continue
        
        qkv_weight = trained_weight[weight_key]
        
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        embed_dim = num_heads * head_dim
        
        # 重塑为 [3, num_heads, head_dim, embed_dim]
        qkv_weight = qkv_weight.view(3, num_heads, head_dim, embed_dim)
        
        # L2范数：权重平方和 [num_heads]
        importance = (qkv_weight ** 2).sum(dim=[0, 2, 3])
        
        head_importance[layer_idx] = importance
    
    return head_importance


def compute_head_importance_saliency(model, train_loader, criterion, num_batches=10) -> Dict[int, torch.Tensor]:
    """
    Saliency Criteria: 基于Taylor展开（权重×梯度）
    
    Args:
        model: ViT模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        num_batches: 用于估计的batch数量
    
    Returns:
        head_importance: {layer_idx: importance_tensor [num_heads]}
    """
    head_importance = {}
    model.train()
    
    # 累积多个batch的梯度
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        data, target = data.cuda(), target.cuda()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 收集每个head的重要性
        for layer_idx, block in enumerate(model.blocks):
            qkv_weight = block.attn.qkv.weight.data
            qkv_grad = block.attn.qkv.weight.grad
            
            if qkv_grad is None:
                continue
            
            num_heads = block.attn.num_heads
            head_dim = block.attn.head_dim
            embed_dim = num_heads * head_dim
            
            # 重塑
            qkv_weight = qkv_weight.view(3, num_heads, head_dim, embed_dim)
            qkv_grad = qkv_grad.view(3, num_heads, head_dim, embed_dim)
            
            # Taylor展开：importance = |weight × gradient|
            importance = (qkv_weight * qkv_grad).abs().sum(dim=[0, 2, 3])
            
            if layer_idx not in head_importance:
                head_importance[layer_idx] = []
            head_importance[layer_idx].append(importance)
    
    # 平均所有batch
    for layer_idx in head_importance:
        head_importance[layer_idx] = torch.stack(
            head_importance[layer_idx]
        ).mean(0)
    
    return head_importance


def compute_vit_head_importance(
    model,
    criteria: str,
    current_mask: Optional[Dict] = None,
    trained_weight: Optional[Dict] = None,
    train_loader = None,
    criterion = None
) -> Dict[int, torch.Tensor]:
    """
    统一的ViT head重要性计算接口
    与ResNet的prune_model_custom_fillback保持一致的设计模式
    
    Args:
        model: ViT模型
        criteria: 评估标准 ['remain', 'magnitude', 'l1', 'l2', 'saliency']
        current_mask: 当前的剪枝mask（用于remain）
        trained_weight: 训练后的权重（用于magnitude/l1/l2）
        train_loader: 训练数据（用于saliency）
        criterion: 损失函数（用于saliency）
    
    Returns:
        head_importance: {layer_idx: importance_tensor [num_heads]}
    """
    print(f"[ViT结构化剪枝] 使用 {criteria} criteria 计算head重要性")
    
    if criteria == 'remain':
        if current_mask is None:
            raise ValueError("remain criteria requires current_mask")
        return compute_head_importance_remain(model, current_mask)
    
    elif criteria in ['magnitude', 'l1']:
        if trained_weight is None:
            raise ValueError(f"{criteria} criteria requires trained_weight")
        return compute_head_importance_magnitude(model, trained_weight)
    
    elif criteria == 'l2':
        if trained_weight is None:
            raise ValueError("l2 criteria requires trained_weight")
        return compute_head_importance_l2(model, trained_weight)
    
    elif criteria == 'saliency':
        if train_loader is None or criterion is None:
            raise ValueError("saliency criteria requires train_loader and criterion")
        return compute_head_importance_saliency(model, train_loader, criterion)
    
    else:
        raise ValueError(f"Unknown criteria: {criteria}. Must be one of ['remain', 'magnitude', 'l1', 'l2', 'saliency']")


# ============================================================================
# Head物理删除函数
# ============================================================================

def prune_attention_heads_hard(attn_module, heads_to_prune: List[int]):
    """
    硬剪枝：物理删除attention heads
    
    Args:
        attn_module: MultiheadAttention模块 (e.g., block.attn)
        heads_to_prune: 要删除的head索引列表，例如 [0, 2, 5]
    
    Returns:
        None (in-place修改模块)
    """
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    embed_dim = num_heads * head_dim
    
    # 确定要保留的heads
    all_heads = set(range(num_heads))
    heads_to_keep = sorted(list(all_heads - set(heads_to_prune)))
    
    if len(heads_to_keep) == 0:
        raise ValueError("Cannot prune all heads!")
    
    print(f"  原始heads: {num_heads}, 剪枝: {len(heads_to_prune)}, 保留: {len(heads_to_keep)}")
    
    # ==================== 处理QKV层 ====================
    qkv_weight = attn_module.qkv.weight.data  # [3*embed_dim, embed_dim]
    qkv_bias = attn_module.qkv.bias.data if attn_module.qkv.bias is not None else None  # [3*embed_dim]
    
    # 重塑为 [3, num_heads, head_dim, embed_dim]
    qkv_weight = qkv_weight.view(3, num_heads, head_dim, embed_dim)
    if qkv_bias is not None:
        qkv_bias = qkv_bias.view(3, num_heads, head_dim)
    
    # 选择要保留的heads
    qkv_weight = qkv_weight[:, heads_to_keep, :, :]
    if qkv_bias is not None:
        qkv_bias = qkv_bias[:, heads_to_keep, :]
    
    # 重塑回线性层格式
    new_num_heads = len(heads_to_keep)
    new_embed_dim = new_num_heads * head_dim
    
    qkv_weight = qkv_weight.reshape(3 * new_embed_dim, embed_dim)
    if qkv_bias is not None:
        qkv_bias = qkv_bias.reshape(3 * new_embed_dim)
    
    # 创建新的qkv层
    new_qkv = nn.Linear(embed_dim, 3 * new_embed_dim, bias=(qkv_bias is not None))
    new_qkv.weight.data = qkv_weight
    if qkv_bias is not None:
        new_qkv.bias.data = qkv_bias
    attn_module.qkv = new_qkv
    
    # ==================== 处理投影层 ====================
    proj_weight = attn_module.proj.weight.data  # [embed_dim, embed_dim]
    proj_bias = attn_module.proj.bias.data if attn_module.proj.bias is not None else None  # [embed_dim]
    
    # 重塑为 [embed_dim, num_heads, head_dim]
    proj_weight = proj_weight.view(embed_dim, num_heads, head_dim)
    
    # 选择要保留的heads
    proj_weight = proj_weight[:, heads_to_keep, :].reshape(embed_dim, new_embed_dim)
    
    # 创建新的投影层
    new_proj = nn.Linear(new_embed_dim, embed_dim, bias=(proj_bias is not None))
    new_proj.weight.data = proj_weight
    if proj_bias is not None:
        new_proj.bias.data = proj_bias
    attn_module.proj = new_proj
    
    # ==================== 更新模块属性 ====================
    attn_module.num_heads = new_num_heads
    
    print(f"  ✓ Head剪枝完成: {num_heads} → {new_num_heads} heads")


def structured_prune_vit_heads(
    model,
    head_importance: Dict[int, torch.Tensor],
    prune_ratio: float
):
    """
    对所有层执行结构化head剪枝
    
    Args:
        model: ViT模型
        head_importance: 每层的head重要性 {layer_idx: importance [num_heads]}
        prune_ratio: 剪枝比例 (0-1)
    """
    print(f"\n[ViT结构化剪枝] 开始剪枝，比例: {prune_ratio:.2%}")
    
    total_heads_before = 0
    total_heads_after = 0
    
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in head_importance:
            print(f"Layer {layer_idx}: 无重要性信息，跳过")
            continue
        
        importance = head_importance[layer_idx]
        num_heads = len(importance)
        num_to_prune = int(num_heads * prune_ratio)
        
        # 至少保留1个head
        if num_to_prune >= num_heads:
            num_to_prune = num_heads - 1
        
        if num_to_prune <= 0:
            print(f"Layer {layer_idx}: 无需剪枝")
            total_heads_before += num_heads
            total_heads_after += num_heads
            continue
        
        # 选择重要性最低的heads进行剪枝
        _, indices = importance.sort()
        heads_to_prune = indices[:num_to_prune].tolist()
        
        print(f"Layer {layer_idx}: 剪枝heads {heads_to_prune}, 重要性: {importance[heads_to_prune].tolist()}")
        
        # 物理删除这些heads
        prune_attention_heads_hard(block.attn, heads_to_prune)
        
        total_heads_before += num_heads
        total_heads_after += (num_heads - num_to_prune)
    
    print(f"\n[ViT结构化剪枝] 完成！")
    print(f"  总Heads: {total_heads_before} → {total_heads_after}")
    print(f"  实际剪枝率: {(1 - total_heads_after / total_heads_before):.2%}")


# ============================================================================
# MLP Neuron剪枝函数
# ============================================================================

def compute_mlp_neuron_importance(
    model,
    criteria: str,
    current_mask: Optional[Dict] = None,
    trained_weight: Optional[Dict] = None,
    train_loader = None,
    criterion = None
) -> Dict[int, torch.Tensor]:
    """
    计算MLP neuron的重要性
    
    Args:
        model: ViT模型
        criteria: 评估标准
        current_mask: 当前mask
        trained_weight: 训练后权重
        train_loader: 训练数据
        criterion: 损失函数
    
    Returns:
        neuron_importance: {layer_idx: importance [hidden_dim]}
    """
    neuron_importance = {}
    
    if criteria == 'remain':
        for layer_idx, block in enumerate(model.blocks):
            mask_key = f'blocks.{layer_idx}.mlp.fc1.weight_mask'
            if mask_key in current_mask:
                mask = current_mask[mask_key]
                # FC1输出维度即为neuron数量
                importance = mask.abs().sum(dim=1)  # [hidden_dim]
                neuron_importance[layer_idx] = importance
    
    elif criteria in ['magnitude', 'l1']:
        for layer_idx, block in enumerate(model.blocks):
            weight_key = f'blocks.{layer_idx}.mlp.fc1.weight'
            if weight_key in trained_weight:
                weight = trained_weight[weight_key]
                importance = weight.abs().sum(dim=1)  # [hidden_dim]
                neuron_importance[layer_idx] = importance
    
    elif criteria == 'l2':
        for layer_idx, block in enumerate(model.blocks):
            weight_key = f'blocks.{layer_idx}.mlp.fc1.weight'
            if weight_key in trained_weight:
                weight = trained_weight[weight_key]
                importance = (weight ** 2).sum(dim=1)  # [hidden_dim]
                neuron_importance[layer_idx] = importance
    
    elif criteria == 'saliency':
        # 类似head的saliency计算
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 10:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            
            for layer_idx, block in enumerate(model.blocks):
                weight = block.mlp.fc1.weight.data
                grad = block.mlp.fc1.weight.grad
                if grad is not None:
                    importance = (weight * grad).abs().sum(dim=1)
                    if layer_idx not in neuron_importance:
                        neuron_importance[layer_idx] = []
                    neuron_importance[layer_idx].append(importance)
        
        for layer_idx in neuron_importance:
            neuron_importance[layer_idx] = torch.stack(
                neuron_importance[layer_idx]
            ).mean(0)
    
    return neuron_importance


def prune_mlp_neurons_hard(mlp_module, neurons_to_prune: List[int]):
    """
    硬剪枝：物理删除MLP neurons
    
    Args:
        mlp_module: MLP模块 (e.g., block.mlp)
        neurons_to_prune: 要删除的neuron索引列表
    """
    hidden_dim = mlp_module.fc1.out_features
    all_neurons = set(range(hidden_dim))
    neurons_to_keep = sorted(list(all_neurons - set(neurons_to_prune)))
    
    if len(neurons_to_keep) == 0:
        raise ValueError("Cannot prune all neurons!")
    
    print(f"  原始neurons: {hidden_dim}, 剪枝: {len(neurons_to_prune)}, 保留: {len(neurons_to_keep)}")
    
    # 剪枝FC1的输出维度
    fc1_weight = mlp_module.fc1.weight.data[neurons_to_keep, :]
    fc1_bias = mlp_module.fc1.bias.data[neurons_to_keep] if mlp_module.fc1.bias is not None else None
    
    new_hidden_dim = len(neurons_to_keep)
    new_fc1 = nn.Linear(mlp_module.fc1.in_features, new_hidden_dim, bias=(fc1_bias is not None))
    new_fc1.weight.data = fc1_weight
    if fc1_bias is not None:
        new_fc1.bias.data = fc1_bias
    
    # 剪枝FC2的输入维度
    fc2_weight = mlp_module.fc2.weight.data[:, neurons_to_keep]
    fc2_bias = mlp_module.fc2.bias.data if mlp_module.fc2.bias is not None else None
    
    new_fc2 = nn.Linear(new_hidden_dim, mlp_module.fc2.out_features, bias=(fc2_bias is not None))
    new_fc2.weight.data = fc2_weight
    if fc2_bias is not None:
        new_fc2.bias.data = fc2_bias
    
    # 替换
    mlp_module.fc1 = new_fc1
    mlp_module.fc2 = new_fc2
    
    print(f"  ✓ MLP Neuron剪枝完成: {hidden_dim} → {new_hidden_dim} neurons")


# ============================================================================
# 统计和检查函数
# ============================================================================

def count_vit_parameters(model):
    """统计ViT模型的参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # 统计各部分
    attn_params = 0
    mlp_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        if 'attn' in name:
            attn_params += param.numel()
        elif 'mlp' in name:
            mlp_params += param.numel()
        else:
            other_params += param.numel()
    
    print(f"[ViT参数统计]")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Attention: {attn_params:,} ({attn_params/total_params:.1%})")
    print(f"  MLP: {mlp_params:,} ({mlp_params/total_params:.1%})")
    print(f"  Other: {other_params:,} ({other_params/total_params:.1%})")
    
    return total_params


def check_vit_structure(model):
    """检查ViT模型的结构"""
    print(f"[ViT结构检查]")
    print(f"  Layers: {len(model.blocks)}")
    
    for layer_idx, block in enumerate(model.blocks):
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim
        embed_dim = num_heads * head_dim
        hidden_dim = block.mlp.fc1.out_features
        
        if layer_idx == 0:
            print(f"  Layer {layer_idx}: {num_heads} heads × {head_dim}d = {embed_dim}d, MLP hidden: {hidden_dim}d")
        elif layer_idx == len(model.blocks) - 1:
            print(f"  Layer {layer_idx}: {num_heads} heads × {head_dim}d = {embed_dim}d, MLP hidden: {hidden_dim}d")


if __name__ == '__main__':
    print("ViT结构化剪枝工具模块")
    print("支持的criteria: remain, magnitude, l1, l2, saliency")
    print("支持的剪枝单元: Attention Head, MLP Neuron")
