"""
Vision Transformer (ViT) 专用剪枝工具
支持对Attention层和MLP层的剪枝
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


def is_vit_model(model):
    """判断模型是否是ViT"""
    from models.vit import VisionTransformer
    
    # 检查是否是自定义的ViT
    if isinstance(model, VisionTransformer):
        return True
    
    # 检查是否是timm的ViT（用于ImageNet）
    try:
        import timm.models.vision_transformer as vit_module
        if isinstance(model, vit_module.VisionTransformer):
            return True
    except:
        pass
    
    return False


def pruning_model_vit(model, px, prune_patch_embed=False):
    """
    对ViT模型进行全局L1剪枝
    
    Args:
        model: ViT模型
        px: 剪枝率
        prune_patch_embed: 是否剪枝patch embedding层
    """
    print(f'[ViT Pruning] Start unstructured L1 pruning for ViT, rate={px}')
    parameters_to_prune = []
    
    for name, m in model.named_modules():
        # Attention层的QKV和Projection
        if 'attn.qkv' in name and isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))
            print(f"  - Adding to prune: {name} (QKV)")
        elif 'attn.proj' in name and isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))
            print(f"  - Adding to prune: {name} (Attention Proj)")
        
        # MLP层的FC1和FC2
        elif 'mlp.fc1' in name and isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))
            print(f"  - Adding to prune: {name} (MLP FC1)")
        elif 'mlp.fc2' in name and isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))
            print(f"  - Adding to prune: {name} (MLP FC2)")
        
        # Patch Embedding (可选)
        elif 'patch_embed.proj' in name and isinstance(m, nn.Conv2d) and prune_patch_embed:
            parameters_to_prune.append((m, 'weight'))
            print(f"  - Adding to prune: {name} (Patch Embed)")
    
    if len(parameters_to_prune) == 0:
        print("[Warning] No layers found for pruning!")
        return
    
    parameters_to_prune = tuple(parameters_to_prune)
    
    # 全局L1剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )
    
    print(f'[ViT Pruning] Pruned {len(parameters_to_prune)} layers globally')


def prune_model_custom_vit(model, mask_dict, prune_patch_embed=False):
    """
    使用自定义mask对ViT进行剪枝
    
    Args:
        model: ViT模型
        mask_dict: 包含各层mask的字典
        prune_patch_embed: 是否剪枝patch embedding层
    """
    print('[ViT Pruning] Applying custom masks')
    
    for name, m in model.named_modules():
        mask_key = name + '.weight_mask'
        
        if mask_key in mask_dict:
            # Attention和MLP的Linear层
            if isinstance(m, nn.Linear):
                print(f'  - Pruning layer with custom mask: {name}')
                prune.CustomFromMask.apply(
                    m, 'weight', 
                    mask=mask_dict[mask_key].to(m.weight.device)
                )
            # Patch Embedding的Conv层
            elif isinstance(m, nn.Conv2d) and prune_patch_embed:
                print(f'  - Pruning layer with custom mask: {name}')
                prune.CustomFromMask.apply(
                    m, 'weight',
                    mask=mask_dict[mask_key].to(m.weight.device)
                )


def check_sparsity_vit(model, prune_patch_embed=False):
    """
    检查ViT模型的稀疏度
    
    Returns:
        remain_weight_rate: 剩余权重比例
    """
    total_zeros = 0
    total_nonzeros = 0
    
    layer_info = []
    
    for name, m in model.named_modules():
        # Attention层
        if 'attn.qkv' in name and isinstance(m, nn.Linear):
            zeros = (m.weight == 0).sum().item()
            total = m.weight.numel()
            layer_info.append((name, zeros, total))
            total_zeros += zeros
            total_nonzeros += total
            
        elif 'attn.proj' in name and isinstance(m, nn.Linear):
            zeros = (m.weight == 0).sum().item()
            total = m.weight.numel()
            layer_info.append((name, zeros, total))
            total_zeros += zeros
            total_nonzeros += total
        
        # MLP层
        elif 'mlp.fc1' in name and isinstance(m, nn.Linear):
            zeros = (m.weight == 0).sum().item()
            total = m.weight.numel()
            layer_info.append((name, zeros, total))
            total_zeros += zeros
            total_nonzeros += total
            
        elif 'mlp.fc2' in name and isinstance(m, nn.Linear):
            zeros = (m.weight == 0).sum().item()
            total = m.weight.numel()
            layer_info.append((name, zeros, total))
            total_zeros += zeros
            total_nonzeros += total
        
        # Patch Embedding
        elif 'patch_embed.proj' in name and isinstance(m, nn.Conv2d) and prune_patch_embed:
            zeros = (m.weight == 0).sum().item()
            total = m.weight.numel()
            layer_info.append((name, zeros, total))
            total_zeros += zeros
            total_nonzeros += total
    
    if total_nonzeros == 0:
        print("[Warning] No prunable layers found!")
        return 100.0
    
    remain_weight_rate = 100 * (1 - total_zeros / total_nonzeros)
    
    print('=' * 80)
    print('[ViT Sparsity Report]')
    print('-' * 80)
    for name, zeros, total in layer_info:
        sparsity = 100 * zeros / total
        print(f'{name:50s} | Sparsity: {sparsity:6.2f}% | Zeros: {zeros:8d}/{total:8d}')
    print('-' * 80)
    print(f'Total zeros: {total_zeros:,} / {total_nonzeros:,}')
    print(f'Overall sparsity: {100 * total_zeros / total_nonzeros:.2f}%')
    print(f'Remaining weights: {remain_weight_rate:.2f}%')
    print('=' * 80)
    
    return remain_weight_rate


def extract_mask_vit(model_dict):
    """
    从ViT模型的state_dict中提取mask
    
    Args:
        model_dict: 模型的state_dict
        
    Returns:
        mask_dict: 包含所有mask的字典
    """
    mask_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            mask_dict[key] = model_dict[key]
    return mask_dict


def remove_prune_vit(model, prune_patch_embed=False):
    """
    移除ViT模型的剪枝重参数化，使mask永久生效
    
    Args:
        model: ViT模型
        prune_patch_embed: 是否处理patch embedding层
    """
    print('[ViT Pruning] Removing pruning reparameterization')
    
    for name, m in model.named_modules():
        # Attention和MLP的Linear层
        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight_mask'):
                print(f'  - Removing prune from: {name}')
                prune.remove(m, 'weight')
        
        # Patch Embedding的Conv层
        elif isinstance(m, nn.Conv2d) and prune_patch_embed:
            if 'patch_embed' in name:
                if hasattr(m, 'weight_mask'):
                    print(f'  - Removing prune from: {name}')
                    prune.remove(m, 'weight')


def prune_model_custom_fillback_vit_by_head(model, mask_dict, train_loader, trained_weight,
                                             init_weight, criteria='l1', prune_ratio=0.2,
                                             return_mask_only=False):
    """
    ViT的准结构化剪枝 - Head级别的mask重组（用于RSST渐进式剪枝）
    
    类似ResNet的通道级准结构化剪枝，但针对ViT的attention heads
    
    Args:
        model: ViT模型
        mask_dict: 当前的element-wise mask字典
        train_loader: 训练数据加载器
        trained_weight: 训练后的权重
        init_weight: 初始权重
        criteria: 重要性评估标准 ('remain', 'magnitude', 'l1', 'l2', 'saliency')
        prune_ratio: 剪枝比例（针对heads）
        return_mask_only: 是否只返回mask不实际剪枝（用于RSST）
        
    Returns:
        如果return_mask_only=True，返回head级别的refill_mask字典
        否则返回剪枝后的模型
    """
    print(f'\n{"="*80}')
    print(f'[ViT Head-level Quasi-Structured Pruning]')
    print(f'  Criteria: {criteria}')
    print(f'  Prune Ratio: {prune_ratio}')
    print(f'  Mode: {"RSST (mask only)" if return_mask_only else "Refill (apply mask)"}')
    print(f'{"="*80}\n')
    
    # 收集特征图（用于l1/l2/saliency criteria）
    feature_maps = {}
    
    def make_hook(layer_name):
        def forward_hook(module, input, output):
            feature_maps[layer_name] = output.detach()
        return forward_hook
    
    # 注册hook收集特征
    if criteria in ['l1', 'l2', 'saliency']:
        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and 'attn.qkv' in name:
                hooks.append(m.register_forward_hook(make_hook(name)))
        
        # 前向传播收集特征
        model.eval()
        images, targets = next(iter(train_loader))
        images, targets = images.cuda(), targets.cuda()
        
        with torch.no_grad():
            output = model(images)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
    
    # 处理每个attention层的QKV权重
    refill_mask = {}
    
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        
        # 只处理Attention的QKV层
        if 'attn.qkv' not in name:
            continue
        
        mask_key = name + '.weight_mask'
        if mask_key not in mask_dict:
            continue
        
        # 获取当前mask
        mask = mask_dict[mask_key].clone()  # [3*embed_dim, embed_dim]
        original_shape = mask.shape
        
        # 获取模块属性
        # 从模型中找到对应的attention模块
        parts = name.split('.')
        attn_module = model
        for part in parts[:-1]:  # 去掉最后的'qkv'
            if part.isdigit():
                attn_module = attn_module[int(part)]
            else:
                attn_module = getattr(attn_module, part)
        
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
        embed_dim = num_heads * head_dim
        
        print(f'\nLayer: {name}')
        print(f'  Shape: {original_shape}')
        print(f'  Num heads: {num_heads}, Head dim: {head_dim}, Embed dim: {embed_dim}')
        
        # 重塑mask为 [3, num_heads, head_dim, embed_dim]
        # QKV的mask: [3*embed_dim, embed_dim] -> [3, num_heads, head_dim, embed_dim]
        mask_reshaped = mask.view(3, num_heads, head_dim, embed_dim)
        
        # === 计算每个head的重要性 ===
        if criteria == 'remain':
            # 方法1: 基于当前mask中非零权重数量
            # importance: [num_heads] - 每个head保留的权重数
            importance = mask_reshaped.sum(dim=[0, 2, 3])  # 对Q/K/V、head_dim、embed_dim求和
            
        elif criteria == 'magnitude':
            # 方法2: 基于训练后权重的绝对值
            weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
            importance = weight.abs().sum(dim=[0, 2, 3])
            
        elif criteria == 'l1':
            # 方法3: 基于特征图的L1范数
            if name in feature_maps:
                feat = feature_maps[name]  # [B, N, 3*embed_dim]
                # 重塑为 [B, N, 3, num_heads, head_dim]
                B, N, _ = feat.shape
                feat = feat.view(B, N, 3, num_heads, head_dim)
                # 计算每个head的L1范数
                importance = feat.abs().mean(dim=[0, 1, 2, 4])  # [num_heads]
            else:
                # fallback到magnitude
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = weight.abs().sum(dim=[0, 2, 3])
                
        elif criteria == 'l2':
            # 方法4: 基于特征图的L2范数
            if name in feature_maps:
                feat = feature_maps[name]  # [B, N, 3*embed_dim]
                B, N, _ = feat.shape
                feat = feat.view(B, N, 3, num_heads, head_dim)
                importance = (feat ** 2).mean(dim=[0, 1, 2, 4]).sqrt()  # [num_heads]
            else:
                # fallback到magnitude
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = (weight ** 2).sum(dim=[0, 2, 3]).sqrt()
                
        elif criteria == 'saliency':
            # 方法5: 基于梯度×权重（需要梯度信息，暂用magnitude替代）
            weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
            importance = weight.abs().sum(dim=[0, 2, 3])
        
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        # === 选择要保留的heads ===
        num_to_keep = max(1, int(num_heads * (1 - prune_ratio)))  # 至少保留1个head
        _, indices = importance.sort(descending=True)
        heads_to_keep = indices[:num_to_keep]
        
        print(f'  Head importance: {importance.cpu().numpy()}')
        print(f'  Keeping {num_to_keep}/{num_heads} heads: {heads_to_keep.cpu().numpy()}')
        
        # === 生成head级别的mask ===
        # 创建新mask：保留的heads全1，剪枝的heads全0
        new_mask = torch.zeros_like(mask_reshaped)
        new_mask[:, heads_to_keep, :, :] = 1  # 保留的heads全部设为1
        
        # 重塑回原始形状
        new_mask = new_mask.view(original_shape)
        
        # 统计剪枝情况
        orig_zeros = (mask == 0).sum().item()
        new_zeros = (new_mask == 0).sum().item()
        total = new_mask.numel()
        
        print(f'  Original sparsity: {100*orig_zeros/total:.2f}%')
        print(f'  New sparsity: {100*new_zeros/total:.2f}% (head-level)')
        
        # 保存mask
        refill_mask[name] = new_mask
        
        # 对应的proj层也需要处理
        proj_name = name.replace('qkv', 'proj')
        proj_mask_key = proj_name + '.weight_mask'
        if proj_mask_key in mask_dict:
            # proj: [embed_dim, embed_dim]
            # 输入维度对应heads，输出维度保持不变
            proj_mask = torch.ones_like(mask_dict[proj_mask_key])
            # 根据保留的heads设置mask
            # proj的输入维度是embed_dim，对应num_heads个heads
            for head_idx in range(num_heads):
                if head_idx not in heads_to_keep:
                    # 剪枝该head对应的输入通道
                    start_idx = head_idx * head_dim
                    end_idx = start_idx + head_dim
                    proj_mask[:, start_idx:end_idx] = 0
            
            refill_mask[proj_name] = proj_mask
            print(f'  Also updated proj mask: {proj_name}')
    
    # MLP层继续使用element-wise mask（不做结构化调整）
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'mlp' in name:
            mask_key = name + '.weight_mask'
            if mask_key in mask_dict:
                refill_mask[name] = mask_dict[mask_key].clone()
    
    if return_mask_only:
        print(f'\n[ViT Head-level Pruning] Returning masks only (for RSST)\n')
        return refill_mask
    
    # 实际应用mask并恢复权重（Refill模式）
    print(f'\n[ViT Head-level Pruning] Applying masks and restoring weights (Refill)\n')
    for name, m in model.named_modules():
        if name in refill_mask:
            mask = refill_mask[name]
            m.weight.data = init_weight[name + '.weight']
            prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
            print(f'  - Applied head-level mask to: {name}')
    
    return model


def prune_model_custom_fillback_vit(model, mask_dict, train_loader, trained_weight, 
                                    init_weight, criteria='l1', fillback_rate=0.0, 
                                    return_mask_only=False):
    """
    ViT的Refill/RSST算法 - 基于重要性恢复或标记权重（Element-wise非结构化版本）
    
    注意：如果需要head级别的准结构化剪枝，请使用prune_model_custom_fillback_vit_by_head
    
    Args:
        model: ViT模型
        mask_dict: 当前的mask字典
        train_loader: 训练数据加载器
        trained_weight: 训练后的权重
        init_weight: 初始权重
        criteria: 重要性评估标准 ('remain', 'magnitude', 'l1', 'saliency')
        fillback_rate: 恢复比例
        return_mask_only: 是否只返回mask不实际剪枝（用于RSST）
        
    Returns:
        如果return_mask_only=True，返回refill_mask字典
        否则返回剪枝后的模型
    """
    print(f'[ViT Fillback] criteria={criteria}, fillback_rate={fillback_rate}')
    
    # 存储特征图（用于l1/saliency criteria）
    feature_maps = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output.detach())
    
    # 注册hook收集特征
    if criteria in ['l1', 'l2', 'saliency']:
        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and ('attn' in name or 'mlp' in name):
                hooks.append(m.register_forward_hook(forward_hook))
        
        # 前向传播收集特征
        model.eval()
        images, targets = next(iter(train_loader))
        images, targets = images.cuda(), targets.cuda()
        
        with torch.no_grad():
            output = model(images)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
    
    # 处理每一层
    refill_mask = {}
    counter = 0
    
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        
        # 只处理Attention和MLP层
        if not ('attn' in name or 'mlp' in name):
            continue
        
        mask_key = name + '.weight_mask'
        if mask_key not in mask_dict:
            continue
        
        mask = mask_dict[mask_key].clone()
        original_shape = mask.shape
        
        # 计算需要恢复的神经元数量
        current_pruned = (mask == 0).sum().item()
        num_to_restore = int(current_pruned * fillback_rate)
        
        print(f'  Layer: {name}, current_pruned={current_pruned}, restore={num_to_restore}')
        
        if num_to_restore > 0:
            if criteria == 'remain':
                # 保持当前mask
                pass
            
            elif criteria == 'magnitude':
                # 按权重幅度恢复
                weight = trained_weight[name + '.weight']
                # 对于Linear层，按行（输出神经元）计算重要性
                importance = weight.abs().sum(dim=1)  # [out_features]
                
                # 找到被剪枝但重要性高的神经元
                pruned_indices = torch.where(mask.sum(dim=1) == 0)[0]
                if len(pruned_indices) > 0:
                    pruned_importance = importance[pruned_indices]
                    _, restore_idx = torch.topk(pruned_importance, min(num_to_restore, len(pruned_indices)))
                    restore_neurons = pruned_indices[restore_idx]
                    mask[restore_neurons] = 1
            
            elif criteria == 'l1':
                # 按特征图L1范数恢复
                if counter < len(feature_maps):
                    feat = feature_maps[counter]
                    # feat: [B, N, C] for ViT
                    importance = feat.abs().mean(dim=[0, 1])  # [C]
                    
                    pruned_indices = torch.where(mask.sum(dim=1) == 0)[0]
                    if len(pruned_indices) > 0 and len(pruned_indices) <= len(importance):
                        pruned_importance = importance[pruned_indices]
                        _, restore_idx = torch.topk(pruned_importance, min(num_to_restore, len(pruned_indices)))
                        restore_neurons = pruned_indices[restore_idx]
                        mask[restore_neurons] = 1
                
                counter += 1
            
            elif criteria == 'l2':
                # 按特征图L2范数恢复
                if counter < len(feature_maps):
                    feat = feature_maps[counter]
                    importance = (feat ** 2).mean(dim=[0, 1]).sqrt()  # [C]
                    
                    pruned_indices = torch.where(mask.sum(dim=1) == 0)[0]
                    if len(pruned_indices) > 0 and len(pruned_indices) <= len(importance):
                        pruned_importance = importance[pruned_indices]
                        _, restore_idx = torch.topk(pruned_importance, min(num_to_restore, len(pruned_indices)))
                        restore_neurons = pruned_indices[restore_idx]
                        mask[restore_neurons] = 1
                
                counter += 1
        
        # 保存mask
        refill_mask[name] = mask
    
    if return_mask_only:
        print('[ViT Fillback] Returning masks only (for RSST)')
        return refill_mask
    
    # 实际应用mask并恢复权重
    print('[ViT Fillback] Applying masks and restoring weights')
    for name, m in model.named_modules():
        if name in refill_mask:
            mask = refill_mask[name]
            m.weight.data = init_weight[name + '.weight']
            prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
            print(f'  - Applied fillback mask to: {name}')
    
    return model

