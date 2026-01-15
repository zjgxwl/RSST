"""
ViT Head + MLP Neurons 组合剪枝工具
支持同时剪枝attention heads和MLP neurons
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_model_custom_fillback_vit_head_and_mlp(
    model, mask_dict, train_loader, trained_weight,
    init_weight, criteria='l1', head_prune_ratio=0.2, mlp_prune_ratio=0.2,
    return_mask_only=False, sorting_mode='layer-wise'
):
    """
    ViT的Head + MLP组合准结构化剪枝
    
    同时剪枝：
    1. Attention heads（head级别）
    2. MLP neurons（neuron级别）
    
    排序策略（由sorting_mode参数控制）：
    - 'layer-wise' (默认): 每层独立排序，每层保持相同的剪枝率
    - 'global': 所有层混合排序，不同层可能保留不同数量的heads/neurons
    
    Args:
        model: ViT模型
        mask_dict: 当前的element-wise mask字典
        train_loader: 训练数据加载器
        trained_weight: 训练后的权重
        init_weight: 初始权重
        criteria: 重要性评估标准 ('remain', 'magnitude', 'l1', 'l2', 'saliency')
        head_prune_ratio: attention head的剪枝率
        mlp_prune_ratio: MLP neurons的剪枝率
        return_mask_only: 是否只返回mask不实际剪枝（用于RSST）
        sorting_mode: 排序方式 ('layer-wise' 或 'global')
        
    Returns:
        如果return_mask_only=True，返回refill_mask字典
        否则返回剪枝后的模型
    """
    sorting_mode = sorting_mode.lower()
    if sorting_mode not in ['layer-wise', 'global']:
        raise ValueError(f"sorting_mode must be 'layer-wise' or 'global', got {sorting_mode}")
    
    print(f'\n{"="*80}')
    print(f'[ViT Head + MLP Quasi-Structured Pruning]')
    print(f'  Criteria: {criteria}')
    print(f'  Head Prune Ratio: {head_prune_ratio}')
    print(f'  MLP Prune Ratio: {mlp_prune_ratio}')
    print(f'  Sorting Mode: {sorting_mode}')
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
            if isinstance(m, nn.Linear):
                if 'attn.qkv' in name or 'mlp.fc1' in name:
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
    
    refill_mask = {}
    
    # ========== Part 1: Attention Head剪枝 ==========
    print('─' * 80)
    if sorting_mode == 'global':
        print('Part 1: Attention Head Pruning (Global Mixed Sorting)')
    else:
        print('Part 1: Attention Head Pruning (Layer-wise Sorting)')
    print('─' * 80)
    
    # 收集所有层的信息（两种模式都需要）
    all_heads = []
    layer_info = {}  # 存储每层的元信息
    
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
        parts = name.split('.')
        attn_module = model
        for part in parts[:-1]:
            if part.isdigit():
                attn_module = attn_module[int(part)]
            else:
                attn_module = getattr(attn_module, part)
        
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
        embed_dim = num_heads * head_dim
        
        # 重塑mask为 [3, num_heads, head_dim, embed_dim]
        mask_reshaped = mask.view(3, num_heads, head_dim, embed_dim)
        
        # === 计算每个head的重要性 ===
        if criteria == 'remain':
            importance = mask_reshaped.sum(dim=[0, 2, 3])
        elif criteria == 'magnitude':
            weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
            importance = weight.abs().sum(dim=[0, 2, 3])
        elif criteria == 'l1':
            if name in feature_maps:
                feat = feature_maps[name]  # [B, N, 3*embed_dim]
                B, N, _ = feat.shape
                feat = feat.view(B, N, 3, num_heads, head_dim)
                importance = feat.abs().mean(dim=[0, 1, 2, 4])  # [num_heads]
            else:
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = weight.abs().sum(dim=[0, 2, 3])
        elif criteria == 'l2':
            if name in feature_maps:
                feat = feature_maps[name]
                B, N, _ = feat.shape
                feat = feat.view(B, N, 3, num_heads, head_dim)
                importance = (feat ** 2).mean(dim=[0, 1, 2, 4]).sqrt()
            else:
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = (weight ** 2).sum(dim=[0, 2, 3]).sqrt()
        elif criteria == 'saliency':
            weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
            importance = weight.abs().sum(dim=[0, 2, 3])
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        # 存储层信息
        layer_info[name] = {
            'num_heads': num_heads,
            'head_dim': head_dim,
            'embed_dim': embed_dim,
            'original_shape': original_shape,
            'mask_reshaped': mask_reshaped,
            'mask': mask
        }
        
        # 收集所有heads
        for head_idx in range(num_heads):
            all_heads.append({
                'layer_name': name,
                'head_idx': head_idx,
                'importance': importance[head_idx].item()
            })
        
        if sorting_mode == 'global':
            print(f'Layer: {name} - {num_heads} heads, importance range: [{importance.min().item():.4f}, {importance.max().item():.4f}]')
    
    # 根据排序模式选择不同的实现
    if sorting_mode == 'global':
        # ========== 全局混合排序 ==========
        # Step 2: 全局排序所有heads
        print(f'\nTotal heads collected: {len(all_heads)}')
        all_heads.sort(key=lambda x: x['importance'], reverse=True)
        
        # Step 3: 保留top-k（按比例）
        total_heads = len(all_heads)
        keep_count = max(1, int(total_heads * (1 - head_prune_ratio)))
        heads_to_keep = all_heads[:keep_count]
        
        print(f'Keeping {keep_count}/{total_heads} heads globally ({100*keep_count/total_heads:.1f}%)')
        
        # Step 4: 按层分组
        heads_by_layer = {}
        for item in heads_to_keep:
            layer_name = item['layer_name']
            if layer_name not in heads_by_layer:
                heads_by_layer[layer_name] = []
            heads_by_layer[layer_name].append(item['head_idx'])
    else:
        # ========== 每层独立排序 ==========
        heads_by_layer = {}
    
    # 生成每层的mask
    for name, info in layer_info.items():
        num_heads = info['num_heads']
        head_dim = info['head_dim']
        embed_dim = info['embed_dim']
        mask_reshaped = info['mask_reshaped']
        original_shape = info['original_shape']
        mask = info['mask']
        
        # 获取该层的重要性（layer-wise模式需要重新计算）
        if sorting_mode == 'layer-wise':
            if criteria == 'remain':
                importance = mask_reshaped.sum(dim=[0, 2, 3])
            elif criteria == 'magnitude':
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = weight.abs().sum(dim=[0, 2, 3])
            elif criteria == 'l1':
                if name in feature_maps:
                    feat = feature_maps[name]
                    B, N, _ = feat.shape
                    feat = feat.view(B, N, 3, num_heads, head_dim)
                    importance = feat.abs().mean(dim=[0, 1, 2, 4])
                else:
                    weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                    importance = weight.abs().sum(dim=[0, 2, 3])
            elif criteria == 'l2':
                if name in feature_maps:
                    feat = feature_maps[name]
                    B, N, _ = feat.shape
                    feat = feat.view(B, N, 3, num_heads, head_dim)
                    importance = (feat ** 2).mean(dim=[0, 1, 2, 4]).sqrt()
                else:
                    weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                    importance = (weight ** 2).sum(dim=[0, 2, 3]).sqrt()
            elif criteria == 'saliency':
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = weight.abs().sum(dim=[0, 2, 3])
        
        # 根据排序模式选择保留的heads
        if sorting_mode == 'global':
            # 全局模式：使用全局排序的结果
            layer_heads_to_keep = heads_by_layer.get(name, [])
            layer_heads_to_keep = torch.tensor(layer_heads_to_keep, dtype=torch.long)
        else:
            # 每层独立模式：在该层内排序
            num_to_keep = max(1, int(num_heads * (1 - head_prune_ratio)))
            _, indices = importance.sort(descending=True)
            layer_heads_to_keep = indices[:num_to_keep]
        
        print(f'\nLayer: {name}')
        print(f'  Keeping {len(layer_heads_to_keep)}/{num_heads} heads: {layer_heads_to_keep.cpu().numpy().tolist()}')
        
        # 生成head级别的mask
        new_mask = torch.zeros_like(mask_reshaped)
        if len(layer_heads_to_keep) > 0:
            new_mask[:, layer_heads_to_keep, :, :] = 1
        new_mask = new_mask.view(original_shape)
        
        orig_zeros = (mask == 0).sum().item()
        new_zeros = (new_mask == 0).sum().item()
        total = new_mask.numel()
        
        print(f'  Original sparsity: {100*orig_zeros/total:.2f}%')
        print(f'  New sparsity: {100*new_zeros/total:.2f}% (head-level)')
        
        refill_mask[name] = new_mask
        
        # 对应的proj层
        proj_name = name.replace('qkv', 'proj')
        proj_mask_key = proj_name + '.weight_mask'
        if proj_mask_key in mask_dict:
            proj_mask = torch.ones_like(mask_dict[proj_mask_key])
            for head_idx in range(num_heads):
                if head_idx not in layer_heads_to_keep:
                    start_idx = head_idx * head_dim
                    end_idx = start_idx + head_dim
                    proj_mask[:, start_idx:end_idx] = 0
            refill_mask[proj_name] = proj_mask
            print(f'  Also updated proj mask: {proj_name}')
    
    # ========== Part 2: MLP Neuron剪枝 ==========
    print(f'\n{"─" * 80}')
    if sorting_mode == 'global':
        print('Part 2: MLP Neuron Pruning (Global Mixed Sorting)')
    else:
        print('Part 2: MLP Neuron Pruning (Layer-wise Sorting)')
    print('─' * 80)
    
    # 收集所有层的信息（两种模式都需要）
    all_neurons = []
    mlp_layer_info = {}  # 存储每层的元信息
    
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        
        # 只处理MLP的FC1层
        if 'mlp.fc1' not in name:
            continue
        
        mask_key = name + '.weight_mask'
        if mask_key not in mask_dict:
            continue
        
        # 获取当前mask
        mask = mask_dict[mask_key].clone()  # [hidden_dim, embed_dim]
        original_shape = mask.shape
        hidden_dim, embed_dim = original_shape
        
        # === 计算每个neuron的重要性 ===
        if criteria == 'remain':
            # 每个neuron（输出通道）的非零权重数
            importance = mask.sum(dim=1)  # [hidden_dim]
        elif criteria == 'magnitude':
            # 每个neuron的权重绝对值和
            weight = trained_weight[name + '.weight']
            importance = weight.abs().sum(dim=1)  # [hidden_dim]
        elif criteria == 'l1':
            if name in feature_maps:
                feat = feature_maps[name]  # [B, N, hidden_dim]
                importance = feat.abs().mean(dim=[0, 1])  # [hidden_dim]
            else:
                weight = trained_weight[name + '.weight']
                importance = weight.abs().sum(dim=1)
        elif criteria == 'l2':
            if name in feature_maps:
                feat = feature_maps[name]
                importance = (feat ** 2).mean(dim=[0, 1]).sqrt()
            else:
                weight = trained_weight[name + '.weight']
                importance = (weight ** 2).sum(dim=1).sqrt()
        elif criteria == 'saliency':
            weight = trained_weight[name + '.weight']
            importance = weight.abs().sum(dim=1)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        # 存储层信息
        mlp_layer_info[name] = {
            'hidden_dim': hidden_dim,
            'embed_dim': embed_dim,
            'original_shape': original_shape,
            'mask': mask
        }
        
        # 收集所有neurons
        for neuron_idx in range(hidden_dim):
            all_neurons.append({
                'layer_name': name,
                'neuron_idx': neuron_idx,
                'importance': importance[neuron_idx].item()
            })
        
        if sorting_mode == 'global':
            print(f'Layer: {name} - {hidden_dim} neurons, importance range: [{importance.min().item():.4f}, {importance.max().item():.4f}]')
    
    # 根据排序模式选择不同的实现
    if sorting_mode == 'global':
        # ========== 全局混合排序 ==========
        # Step 2: 全局排序所有neurons
        print(f'\nTotal neurons collected: {len(all_neurons)}')
        all_neurons.sort(key=lambda x: x['importance'], reverse=True)
        
        # Step 3: 保留top-k（按比例）
        total_neurons = len(all_neurons)
        keep_count = max(1, int(total_neurons * (1 - mlp_prune_ratio)))
        neurons_to_keep = all_neurons[:keep_count]
        
        print(f'Keeping {keep_count}/{total_neurons} neurons globally ({100*keep_count/total_neurons:.1f}%)')
        
        # Step 4: 按层分组
        neurons_by_layer = {}
        for item in neurons_to_keep:
            layer_name = item['layer_name']
            if layer_name not in neurons_by_layer:
                neurons_by_layer[layer_name] = []
            neurons_by_layer[layer_name].append(item['neuron_idx'])
    else:
        # ========== 每层独立排序 ==========
        neurons_by_layer = {}
    
    # 生成每层的mask
    for name, info in mlp_layer_info.items():
        hidden_dim = info['hidden_dim']
        original_shape = info['original_shape']
        mask = info['mask']
        
        # 根据排序模式选择保留的neurons
        if sorting_mode == 'global':
            # 全局模式：使用全局排序的结果
            layer_neurons_to_keep = neurons_by_layer.get(name, [])
            layer_neurons_to_keep = torch.tensor(layer_neurons_to_keep, dtype=torch.long)
        else:
            # 每层独立模式：在该层内排序
            # 重新计算重要性
            if criteria == 'remain':
                importance = mask.sum(dim=1)
            elif criteria == 'magnitude':
                weight = trained_weight[name + '.weight']
                importance = weight.abs().sum(dim=1)
            elif criteria == 'l1':
                if name in feature_maps:
                    feat = feature_maps[name]
                    importance = feat.abs().mean(dim=[0, 1])
                else:
                    weight = trained_weight[name + '.weight']
                    importance = weight.abs().sum(dim=1)
            elif criteria == 'l2':
                if name in feature_maps:
                    feat = feature_maps[name]
                    importance = (feat ** 2).mean(dim=[0, 1]).sqrt()
                else:
                    weight = trained_weight[name + '.weight']
                    importance = (weight ** 2).sum(dim=1).sqrt()
            elif criteria == 'saliency':
                weight = trained_weight[name + '.weight']
                importance = weight.abs().sum(dim=1)
            
            num_to_keep = max(1, int(hidden_dim * (1 - mlp_prune_ratio)))
            _, indices = importance.sort(descending=True)
            layer_neurons_to_keep = indices[:num_to_keep]
        
        print(f'\nLayer: {name}')
        print(f'  Keeping {len(layer_neurons_to_keep)}/{hidden_dim} neurons')
        
        # 生成neuron级别的mask
        new_mask = torch.zeros_like(mask)
        if len(layer_neurons_to_keep) > 0:
            new_mask[layer_neurons_to_keep, :] = 1  # 保留的neurons全1
        
        orig_zeros = (mask == 0).sum().item()
        new_zeros = (new_mask == 0).sum().item()
        total = new_mask.numel()
        
        print(f'  Original sparsity: {100*orig_zeros/total:.2f}%')
        print(f'  New sparsity: {100*new_zeros/total:.2f}% (neuron-level)')
        
        refill_mask[name] = new_mask
        
        # 对应的FC2层
        fc2_name = name.replace('fc1', 'fc2')
        fc2_mask_key = fc2_name + '.weight_mask'
        if fc2_mask_key in mask_dict:
            fc2_mask = torch.ones_like(mask_dict[fc2_mask_key])
            # FC2的输入维度对应FC1的输出维度
            for neuron_idx in range(hidden_dim):
                if neuron_idx not in layer_neurons_to_keep:
                    fc2_mask[:, neuron_idx] = 0  # 剪枝该neuron对应的输入通道
            refill_mask[fc2_name] = fc2_mask
            print(f'  Also updated fc2 mask: {fc2_name}')
    
    # ========== 总结 ==========
    print(f'\n{"="*80}')
    print('Summary:')
    print(f'  Total masks generated: {len(refill_mask)}')
    
    attn_layers = sum(1 for k in refill_mask.keys() if 'attn' in k)
    mlp_layers = sum(1 for k in refill_mask.keys() if 'mlp' in k)
    print(f'  Attention layers: {attn_layers}')
    print(f'  MLP layers: {mlp_layers}')
    
    # 计算总体稀疏度
    total_zeros = sum((mask == 0).sum().item() for mask in refill_mask.values())
    total_weights = sum(mask.numel() for mask in refill_mask.values())
    overall_sparsity = 100 * total_zeros / total_weights if total_weights > 0 else 0
    print(f'  Overall sparsity: {overall_sparsity:.2f}%')
    print(f'{"="*80}\n')
    
    if return_mask_only:
        print('[ViT Head+MLP Pruning] Returning masks only (for RSST)\n')
        return refill_mask
    
    # 实际应用mask并恢复权重（Refill模式）
    print('[ViT Head+MLP Pruning] Applying masks and restoring weights (Refill)\n')
    for name, m in model.named_modules():
        if name in refill_mask:
            mask = refill_mask[name]
            m.weight.data = init_weight[name + '.weight']
            prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
            print(f'  - Applied mask to: {name}')
    
    return model
