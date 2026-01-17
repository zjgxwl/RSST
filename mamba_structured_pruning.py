"""
Mamba模型的结构化剪枝工具
支持RSST和Refill方法的完全结构化剪枝

剪枝策略:
1. SSM输出投影 (ssm.out_proj): 输入通道级剪枝
2. MLP神经元 (mlp.fc1/fc2): 神经元级剪枝 (与ViT相同)
3. 混合剪枝: 同时剪枝SSM和MLP

不做非结构化(element-wise)剪枝！
"""
import torch
import torch.nn as nn
import copy


# ==================== Model Identification ====================

def is_mamba_model(model):
    """
    判断模型是否是Mamba模型
    
    Args:
        model: PyTorch model
    Returns:
        bool: True if Mamba model
    """
    from models.mamba import MambaModel
    return isinstance(model, MambaModel)


# ==================== Importance Calculation ====================

def calculate_channel_importance(linear_layer, method='l1', grad_aware=False):
    """
    计算Linear层的输入通道重要性（用于SSM out_proj）
    
    Args:
        linear_layer: nn.Linear [out_features, in_features]
        method: 'l1', 'l2', or 'taylor'
        grad_aware: 是否考虑梯度
    Returns:
        importance: [in_features] tensor
    """
    weight = linear_layer.weight.data  # [out, in]
    
    if method == 'l1':
        # L1范数: 每个输入通道的绝对值之和
        importance = weight.abs().sum(dim=0)  # [in_features]
    
    elif method == 'l2':
        # L2范数
        importance = weight.pow(2).sum(dim=0).sqrt()
    
    elif method == 'taylor' and grad_aware:
        # Taylor展开: |weight * grad|
        if linear_layer.weight.grad is not None:
            grad = linear_layer.weight.grad.data
            importance = (weight * grad).abs().sum(dim=0)
        else:
            # Fallback to L1
            importance = weight.abs().sum(dim=0)
    
    else:
        importance = weight.abs().sum(dim=0)
    
    return importance


def calculate_neuron_importance(fc1, fc2, method='l1'):
    """
    计算MLP神经元重要性 (完全复用ViT的逻辑)
    
    Args:
        fc1: nn.Linear [d_model, mlp_dim]
        fc2: nn.Linear [mlp_dim, d_model]
        method: 'l1' or 'l2'
    Returns:
        importance: [mlp_dim] tensor
    """
    if method == 'l1':
        # fc1的输出权重 + fc2的输入权重
        imp_fc1 = fc1.weight.data.abs().sum(dim=1)  # [mlp_dim]
        imp_fc2 = fc2.weight.data.abs().sum(dim=0)  # [mlp_dim]
        importance = imp_fc1 + imp_fc2
    
    elif method == 'l2':
        imp_fc1 = fc1.weight.data.pow(2).sum(dim=1).sqrt()
        imp_fc2 = fc2.weight.data.pow(2).sum(dim=0).sqrt()
        importance = imp_fc1 + imp_fc2
    
    else:
        imp_fc1 = fc1.weight.data.abs().sum(dim=1)
        imp_fc2 = fc2.weight.data.abs().sum(dim=0)
        importance = imp_fc1 + imp_fc2
    
    return importance


# ==================== Structured Pruning Functions ====================

def prune_mamba_ssm_structured(model, prune_ratio=0.7, method='global', 
                               importance_method='l1'):
    """
    结构化剪枝Mamba的SSM输出投影层
    删除不重要的输入通道（完全结构化）
    
    Args:
        model: MambaModel
        prune_ratio: 剪枝率 (0.7 = 删除70%的通道)
        method: 'global' (全局排序) or 'layerwise' (逐层剪枝)
        importance_method: 'l1', 'l2', or 'taylor'
    
    Returns:
        pruned_info: dict with pruning statistics
    """
    print(f"\n{'='*60}")
    print(f"[Mamba SSM Structured Pruning]")
    print(f"  Prune ratio: {prune_ratio:.1%}")
    print(f"  Method: {method}")
    print(f"  Importance: {importance_method}")
    print(f"{'='*60}\n")
    
    importance_dict = {}
    module_dict = {}
    
    # 1. 收集所有SSM输出投影层的重要性
    for name, module in model.named_modules():
        if 'ssm.out_proj' in name and isinstance(module, nn.Linear):
            importance = calculate_channel_importance(module, importance_method)
            importance_dict[name] = importance
            module_dict[name] = module
            print(f"  Collected: {name}, in_features={module.in_features}")
    
    if len(importance_dict) == 0:
        print("⚠️  No SSM layers found for pruning!")
        return {'pruned': 0, 'total': 0}
    
    # 2. 确定每层保留的通道
    keep_channels_dict = {}
    
    if method == 'global':
        # 全局排序：所有层合并后统一阈值
        all_importance = torch.cat(list(importance_dict.values()))
        threshold = torch.quantile(all_importance, prune_ratio)
        print(f"  Global threshold: {threshold:.6f}")
        
        for name, importance in importance_dict.items():
            keep_mask = importance >= threshold
            keep_channels = keep_mask.nonzero().squeeze(-1)
            keep_channels_dict[name] = keep_channels
            
            n_total = len(importance)
            n_keep = len(keep_channels)
            print(f"    {name}: {n_keep}/{n_total} channels kept ({n_keep/n_total:.1%})")
    
    elif method == 'layerwise':
        # 逐层剪枝：每层独立保留(1-prune_ratio)的通道
        for name, importance in importance_dict.items():
            n_total = len(importance)
            n_keep = max(1, int(n_total * (1 - prune_ratio)))
            
            _, indices = torch.topk(importance, n_keep)
            keep_channels_dict[name] = indices.sort()[0]  # 保持顺序
            
            print(f"    {name}: {n_keep}/{n_total} channels kept ({n_keep/n_total:.1%})")
    
    # 3. 实际修改模型结构
    print(f"\n  Applying pruning...")
    total_params_before = sum(p.numel() for p in model.parameters())
    
    for block_idx, block in enumerate(model.blocks):
        out_proj_name = f'blocks.{block_idx}.ssm.out_proj'
        
        if out_proj_name in keep_channels_dict:
            keep_channels = keep_channels_dict[out_proj_name]
            
            # 修改out_proj
            old_out_proj = block.ssm.out_proj
            new_in_features = len(keep_channels)
            new_out_proj = nn.Linear(
                new_in_features, 
                old_out_proj.out_features, 
                bias=(old_out_proj.bias is not None)
            ).to(old_out_proj.weight.device)
            
            # 复制权重
            new_out_proj.weight.data = old_out_proj.weight.data[:, keep_channels]
            if old_out_proj.bias is not None:
                new_out_proj.bias.data = old_out_proj.bias.data
            
            block.ssm.out_proj = new_out_proj
            
            # TODO: 协同修改上游层 (in_proj, conv1d, x_proj, D)
            # 这需要修改SSM的d_inner维度
            # 暂时只修改out_proj以验证流程
    
    total_params_after = sum(p.numel() for p in model.parameters())
    
    print(f"\n  ✓ Pruning completed!")
    print(f"  Parameters: {total_params_before:,} → {total_params_after:,}")
    print(f"  Reduction: {(1 - total_params_after/total_params_before)*100:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'pruned': total_params_before - total_params_after,
        'total': total_params_before,
        'ratio': 1 - total_params_after/total_params_before,
        'keep_channels': keep_channels_dict
    }


def prune_mamba_mlp_structured(model, prune_ratio=0.7, method='global',
                               importance_method='l1'):
    """
    结构化剪枝Mamba的MLP神经元
    与ViT的MLP剪枝完全相同！
    
    Args:
        model: MambaModel
        prune_ratio: 剪枝率 (0.7 = 删除70%的神经元)
        method: 'global' or 'layerwise'
        importance_method: 'l1' or 'l2'
    
    Returns:
        pruned_info: dict with pruning statistics
    """
    print(f"\n{'='*60}")
    print(f"[Mamba MLP Structured Pruning]")
    print(f"  Prune ratio: {prune_ratio:.1%}")
    print(f"  Method: {method}")
    print(f"  Importance: {importance_method}")
    print(f"{'='*60}\n")
    
    importance_dict = {}
    mlp_dict = {}
    
    # 1. 收集所有MLP的神经元重要性
    for block_idx, block in enumerate(model.blocks):
        if hasattr(block, 'mlp') and block.use_mlp:
            fc1 = block.mlp[0]  # Linear
            fc2 = block.mlp[3]  # Linear (跳过GELU和Dropout)
            
            if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                importance = calculate_neuron_importance(fc1, fc2, importance_method)
                name = f'blocks.{block_idx}.mlp'
                importance_dict[name] = importance
                mlp_dict[name] = (fc1, fc2)
                print(f"  Collected: {name}, mlp_dim={fc1.out_features}")
    
    if len(importance_dict) == 0:
        print("⚠️  No MLP layers found for pruning!")
        return {'pruned': 0, 'total': 0}
    
    # 2. 确定每层保留的神经元
    keep_neurons_dict = {}
    
    if method == 'global':
        # 全局排序
        all_importance = torch.cat(list(importance_dict.values()))
        threshold = torch.quantile(all_importance, prune_ratio)
        print(f"  Global threshold: {threshold:.6f}")
        
        for name, importance in importance_dict.items():
            keep_mask = importance >= threshold
            keep_neurons = keep_mask.nonzero().squeeze(-1)
            keep_neurons_dict[name] = keep_neurons
            
            n_total = len(importance)
            n_keep = len(keep_neurons)
            print(f"    {name}: {n_keep}/{n_total} neurons kept ({n_keep/n_total:.1%})")
    
    elif method == 'layerwise':
        # 逐层剪枝
        for name, importance in importance_dict.items():
            n_total = len(importance)
            n_keep = max(1, int(n_total * (1 - prune_ratio)))
            
            _, indices = torch.topk(importance, n_keep)
            keep_neurons_dict[name] = indices.sort()[0]
            
            print(f"    {name}: {n_keep}/{n_total} neurons kept ({n_keep/n_total:.1%})")
    
    # 3. 实际修改MLP结构
    print(f"\n  Applying pruning...")
    total_params_before = sum(p.numel() for p in model.parameters())
    
    for name, (fc1, fc2) in mlp_dict.items():
        if name in keep_neurons_dict:
            keep_neurons = keep_neurons_dict[name]
            
            # 修改fc1的输出
            new_fc1 = nn.Linear(
                fc1.in_features,
                len(keep_neurons),
                bias=(fc1.bias is not None)
            ).to(fc1.weight.device)
            
            new_fc1.weight.data = fc1.weight.data[keep_neurons, :]
            if fc1.bias is not None:
                new_fc1.bias.data = fc1.bias.data[keep_neurons]
            
            # 修改fc2的输入
            new_fc2 = nn.Linear(
                len(keep_neurons),
                fc2.out_features,
                bias=(fc2.bias is not None)
            ).to(fc2.weight.device)
            
            new_fc2.weight.data = fc2.weight.data[:, keep_neurons]
            if fc2.bias is not None:
                new_fc2.bias.data = fc2.bias.data
            
            # 替换原有层
            block_idx = int(name.split('.')[1])
            model.blocks[block_idx].mlp[0] = new_fc1
            model.blocks[block_idx].mlp[3] = new_fc2
    
    total_params_after = sum(p.numel() for p in model.parameters())
    
    print(f"\n  ✓ Pruning completed!")
    print(f"  Parameters: {total_params_before:,} → {total_params_after:,}")
    print(f"  Reduction: {(1 - total_params_after/total_params_before)*100:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'pruned': total_params_before - total_params_after,
        'total': total_params_before,
        'ratio': 1 - total_params_after/total_params_before,
        'keep_neurons': keep_neurons_dict
    }


def prune_mamba_hybrid(model, ssm_ratio=0.7, mlp_ratio=0.7, method='global'):
    """
    混合剪枝：同时剪枝SSM和MLP
    
    Args:
        model: MambaModel
        ssm_ratio: SSM剪枝率
        mlp_ratio: MLP剪枝率
        method: 'global' or 'layerwise'
    
    Returns:
        pruned_info: dict with combined statistics
    """
    print(f"\n{'='*60}")
    print(f"[Mamba Hybrid Structured Pruning]")
    print(f"  SSM prune ratio: {ssm_ratio:.1%}")
    print(f"  MLP prune ratio: {mlp_ratio:.1%}")
    print(f"{'='*60}")
    
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # 1. 剪枝SSM
    ssm_info = prune_mamba_ssm_structured(model, ssm_ratio, method)
    
    # 2. 剪枝MLP
    mlp_info = prune_mamba_mlp_structured(model, mlp_ratio, method)
    
    total_params_after = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print(f"[Hybrid Pruning Summary]")
    print(f"  Total parameters: {total_params_before:,} → {total_params_after:,}")
    print(f"  Total reduction: {(1 - total_params_after/total_params_before)*100:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'pruned': total_params_before - total_params_after,
        'total': total_params_before,
        'ratio': 1 - total_params_after/total_params_before,
        'ssm_info': ssm_info,
        'mlp_info': mlp_info
    }


# ==================== RSST Regularization ====================

def compute_mamba_structured_regularization(model, reg_strength=1e-4, 
                                           reg_target='both'):
    """
    计算Mamba的结构化稀疏正则化（用于RSST）
    鼓励整个通道/神经元的权重趋向于0
    
    Args:
        model: MambaModel
        reg_strength: 正则化强度
        reg_target: 'ssm', 'mlp', or 'both'
    
    Returns:
        reg_loss: scalar tensor
    """
    reg_loss = 0.0
    count = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        # SSM输出投影的通道级正则化
        if reg_target in ['ssm', 'both'] and 'ssm.out_proj' in name:
            # 每个输入通道的L1范数
            channel_norms = module.weight.abs().sum(dim=0)  # [in_features]
            reg_loss = reg_loss + reg_strength * channel_norms.sum()
            count += 1
        
        # MLP神经元级正则化
        if reg_target in ['mlp', 'both'] and 'mlp.0' in name:  # fc1
            # 每个输出神经元的L1范数
            neuron_norms = module.weight.abs().sum(dim=1)  # [mlp_dim]
            reg_loss = reg_loss + reg_strength * neuron_norms.sum()
            count += 1
    
    if count > 0:
        reg_loss = reg_loss / count  # 归一化
    
    return reg_loss


def rsst_schedule_exp(epoch, total_epochs, base_strength=1e-4, exponent=4):
    """
    RSST的指数增长schedule（与ViT的RSST完全相同）
    
    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        base_strength: 基础正则化强度
        exponent: 指数
    
    Returns:
        strength: 当前的正则化强度
    """
    progress = epoch / total_epochs
    strength = base_strength * (progress ** exponent)
    return strength


# ==================== Sparsity Check ====================

def check_mamba_sparsity(model, verbose=False):
    """
    检查Mamba模型的稀疏度
    
    Args:
        model: MambaModel
        verbose: 是否打印详细信息
    
    Returns:
        sparsity_info: dict with sparsity statistics
    """
    total_params = 0
    total_zeros = 0
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'head' not in name:
            params = module.weight.numel()
            zeros = (module.weight.data == 0).sum().item()
            sparsity = zeros / params if params > 0 else 0
            
            total_params += params
            total_zeros += zeros
            
            layer_info.append({
                'name': name,
                'params': params,
                'zeros': zeros,
                'sparsity': sparsity
            })
            
            if verbose:
                print(f"{name:50s} | Sparsity: {sparsity:6.2%} | Zeros: {zeros:8d}/{params:8d}")
    
    overall_sparsity = total_zeros / total_params if total_params > 0 else 0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Overall Sparsity: {overall_sparsity:.2%}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Total Zeros: {total_zeros:,}")
        print(f"{'='*80}")
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_params': total_params,
        'total_zeros': total_zeros,
        'layer_info': layer_info
    }


# ==================== Testing ====================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing Mamba Structured Pruning")
    print("=" * 70)
    
    from models.mamba import mamba_small
    
    # Create model
    print("\n[1] Creating Mamba-Small model...")
    model = mamba_small(num_classes=10, img_size=32)
    params_init = sum(p.numel() for p in model.parameters())
    print(f"    Initial parameters: {params_init:,}")
    
    # Test forward
    print("\n[2] Testing forward pass...")
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f"    Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 10), "Output shape mismatch"
    
    # Test SSM pruning
    print("\n[3] Testing SSM structured pruning...")
    model_ssm = mamba_small(num_classes=10, img_size=32)
    info_ssm = prune_mamba_ssm_structured(model_ssm, prune_ratio=0.7)
    
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model_ssm(x)
    print(f"    After pruning - Output: {y.shape}")
    assert y.shape == (2, 10), "Output shape mismatch after pruning"
    
    # Test MLP pruning
    print("\n[4] Testing MLP structured pruning...")
    model_mlp = mamba_small(num_classes=10, img_size=32)
    info_mlp = prune_mamba_mlp_structured(model_mlp, prune_ratio=0.7)
    
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model_mlp(x)
    print(f"    After pruning - Output: {y.shape}")
    assert y.shape == (2, 10), "Output shape mismatch after pruning"
    
    # Test hybrid pruning
    print("\n[5] Testing hybrid pruning...")
    model_hybrid = mamba_small(num_classes=10, img_size=32)
    info_hybrid = prune_mamba_hybrid(model_hybrid, ssm_ratio=0.7, mlp_ratio=0.7)
    
    # Test RSST regularization
    print("\n[6] Testing RSST regularization...")
    model_rsst = mamba_small(num_classes=10, img_size=32)
    reg_loss = compute_mamba_structured_regularization(
        model_rsst, reg_strength=1e-4, reg_target='both'
    )
    print(f"    RSST reg loss: {reg_loss.item():.6f}")
    assert reg_loss.item() > 0, "Regularization loss should be positive"
    
    # Test schedule
    print("\n[7] Testing RSST schedule...")
    for epoch in [0, 30, 60, 90]:
        strength = rsst_schedule_exp(epoch, 160, 1e-4, 4)
        print(f"    Epoch {epoch:3d}: reg_strength = {strength:.8f}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
