"""
测试ViT Head+MLP剪枝的两种排序模式
- layer-wise: 每层独立排序
- global: 全局混合排序
"""
import torch
import torch.nn as nn
import sys
sys.path.append('.')

from models.vit import vit_tiny
from vit_pruning_utils_head_mlp import prune_model_custom_fillback_vit_head_and_mlp

def create_dummy_data(batch_size=2, num_classes=10):
    """创建虚拟数据"""
    images = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size,))
    return images, targets

def create_dummy_loader():
    """创建虚拟数据加载器"""
    class DummyDataset:
        def __init__(self):
            self.data = [create_dummy_data() for _ in range(5)]
        
        def __iter__(self):
            return iter(self.data)
    
    return DummyDataset()

def test_sorting_mode(model, mask_dict, train_loader, trained_weight, init_weight, 
                      sorting_mode, head_prune_ratio=0.3, mlp_prune_ratio=0.3):
    """测试指定的排序模式"""
    print(f"\n{'='*80}")
    print(f"测试排序模式: {sorting_mode}")
    print(f"{'='*80}\n")
    
    try:
        refill_mask = prune_model_custom_fillback_vit_head_and_mlp(
            model=model,
            mask_dict=mask_dict,
            train_loader=train_loader,
            trained_weight=trained_weight,
            init_weight=init_weight,
            criteria='magnitude',
            head_prune_ratio=head_prune_ratio,
            mlp_prune_ratio=mlp_prune_ratio,
            return_mask_only=True,
            sorting_mode=sorting_mode
        )
        
        print(f"\n✓ {sorting_mode}模式测试成功！")
        
        # 统计每层保留的heads和neurons
        heads_by_layer = {}
        neurons_by_layer = {}
        
        for name, mask in refill_mask.items():
            if 'attn.qkv' in name:
                # 计算该层保留的heads数量
                mask_reshaped = mask.view(3, 3, 64, 192)  # vit_tiny: 3 heads, 64 head_dim, 192 embed_dim
                heads_kept = (mask_reshaped.sum(dim=[0, 2, 3]) > 0).sum().item()
                heads_by_layer[name] = heads_kept
            elif 'mlp.fc1' in name:
                # 计算该层保留的neurons数量
                neurons_kept = (mask.sum(dim=1) > 0).sum().item()
                neurons_by_layer[name] = neurons_kept
        
        print(f"\n每层保留的Heads数量:")
        for name, count in heads_by_layer.items():
            print(f"  {name}: {count}/3")
        
        print(f"\n每层保留的MLP Neurons数量:")
        for name, count in neurons_by_layer.items():
            print(f"  {name}: {count}/768")
        
        return True, refill_mask
        
    except Exception as e:
        print(f"\n✗ {sorting_mode}模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    print("="*80)
    print("ViT Head+MLP剪枝排序模式测试")
    print("="*80)
    
    # 创建模型
    print("\n1. 创建ViT-Tiny模型...")
    model = vit_tiny(num_classes=10, img_size=32, pretrained=False)
    model = model.cuda()
    print("✓ 模型创建成功")
    
    # 创建虚拟数据
    print("\n2. 创建虚拟数据...")
    train_loader = create_dummy_loader()
    print("✓ 数据创建成功")
    
    # 创建初始mask（全1，表示所有权重都保留）
    print("\n3. 创建初始mask...")
    mask_dict = {}
    trained_weight = {}
    init_weight = {}
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'attn' in name or 'mlp' in name:
                weight_shape = m.weight.shape
                mask = torch.ones(weight_shape, device='cuda')
                mask_dict[name + '.weight_mask'] = mask
                trained_weight[name + '.weight'] = m.weight.data.clone()
                init_weight[name + '.weight'] = m.weight.data.clone()
    
    print(f"✓ 创建了{len(mask_dict)}个mask")
    
    # 测试layer-wise模式
    print("\n" + "="*80)
    print("测试1: Layer-wise模式（每层独立排序）")
    print("="*80)
    success1, mask1 = test_sorting_mode(
        model, mask_dict, train_loader, trained_weight, init_weight,
        sorting_mode='layer-wise', head_prune_ratio=0.3, mlp_prune_ratio=0.3
    )
    
    # 测试global模式
    print("\n" + "="*80)
    print("测试2: Global模式（全局混合排序）")
    print("="*80)
    success2, mask2 = test_sorting_mode(
        model, mask_dict, train_loader, trained_weight, init_weight,
        sorting_mode='global', head_prune_ratio=0.3, mlp_prune_ratio=0.3
    )
    
    # 对比结果
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    if success1 and success2:
        print("\n✓ 两种模式都测试成功！")
        
        # 统计layer-wise模式的结果
        print("\n【Layer-wise模式】")
        layer_wise_heads = {}
        layer_wise_neurons = {}
        for name, mask in mask1.items():
            if 'attn.qkv' in name:
                mask_reshaped = mask.view(3, 3, 64, 192)
                heads_kept = (mask_reshaped.sum(dim=[0, 2, 3]) > 0).sum().item()
                layer_wise_heads[name] = heads_kept
            elif 'mlp.fc1' in name:
                neurons_kept = (mask.sum(dim=1) > 0).sum().item()
                layer_wise_neurons[name] = neurons_kept
        
        # 统计global模式的结果
        print("\n【Global模式】")
        global_heads = {}
        global_neurons = {}
        for name, mask in mask2.items():
            if 'attn.qkv' in name:
                mask_reshaped = mask.view(3, 3, 64, 192)
                heads_kept = (mask_reshaped.sum(dim=[0, 2, 3]) > 0).sum().item()
                global_heads[name] = heads_kept
            elif 'mlp.fc1' in name:
                neurons_kept = (mask.sum(dim=1) > 0).sum().item()
                global_neurons[name] = neurons_kept
        
        # 对比
        print("\n【Heads对比】")
        print("Layer-wise: 每层应该保留2个heads (70%)")
        print("Global: 各层可能不同")
        for name in layer_wise_heads.keys():
            lw = layer_wise_heads.get(name, 0)
            gl = global_heads.get(name, 0)
            print(f"  {name}: Layer-wise={lw}, Global={gl}")
        
        print("\n【MLP Neurons对比】")
        print("Layer-wise: 每层应该保留约538个neurons (70%)")
        print("Global: 各层可能不同")
        for name in list(layer_wise_neurons.keys())[:3]:  # 只显示前3层
            lw = layer_wise_neurons.get(name, 0)
            gl = global_neurons.get(name, 0)
            print(f"  {name}: Layer-wise={lw}, Global={gl}")
        
        # 验证layer-wise模式的一致性
        print("\n【验证Layer-wise模式】")
        all_same_heads = len(set(layer_wise_heads.values())) == 1
        all_same_neurons = len(set(layer_wise_neurons.values())) == 1
        print(f"  所有层保留相同数量的heads: {all_same_heads} (应该为True)")
        print(f"  所有层保留相同数量的neurons: {all_same_neurons} (应该为True)")
        
        # 验证global模式的不一致性
        print("\n【验证Global模式】")
        different_heads = len(set(global_heads.values())) > 1
        different_neurons = len(set(global_neurons.values())) > 1
        print(f"  不同层保留不同数量的heads: {different_heads} (可能为True)")
        print(f"  不同层保留不同数量的neurons: {different_neurons} (可能为True)")
        
    else:
        print("\n✗ 测试失败")
        if not success1:
            print("  - Layer-wise模式失败")
        if not success2:
            print("  - Global模式失败")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)

if __name__ == '__main__':
    main()
