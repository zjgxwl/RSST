"""
测试ViT结构化剪枝功能
快速验证结构化剪枝是否正常工作
"""

import torch
import torch.nn as nn
from models.vit import vit_tiny
import vit_structured_pruning

def test_basic_structure():
    """测试基本的结构化剪枝功能"""
    print("\n" + "="*60)
    print("测试1: 基本结构化剪枝功能")
    print("="*60)
    
    # 创建一个ViT-Tiny模型
    model = vit_tiny(num_classes=10, img_size=32, pretrained=False)
    model.eval()
    
    print("\n原始模型结构:")
    vit_structured_pruning.check_vit_structure(model)
    vit_structured_pruning.count_vit_parameters(model)
    
    # 创建一些假的训练权重
    trained_weight = model.state_dict()
    
    # 测试magnitude criteria
    print("\n测试Magnitude Criteria:")
    head_importance = vit_structured_pruning.compute_vit_head_importance(
        model=model,
        criteria='magnitude',
        trained_weight=trained_weight
    )
    
    print(f"计算得到的head重要性:")
    for layer_idx, importance in head_importance.items():
        print(f"  Layer {layer_idx}: {importance.tolist()}")
    
    # 执行结构化剪枝（剪枝33%的heads）
    print("\n执行结构化剪枝 (33% heads):")
    vit_structured_pruning.structured_prune_vit_heads(
        model=model,
        head_importance=head_importance,
        prune_ratio=0.33
    )
    
    print("\n剪枝后模型结构:")
    vit_structured_pruning.check_vit_structure(model)
    vit_structured_pruning.count_vit_parameters(model)
    
    # 测试前向传播
    print("\n测试前向传播:")
    dummy_input = torch.randn(2, 3, 32, 32)
    try:
        output = model(dummy_input)
        print(f"✓ 前向传播成功！输出shape: {output.shape}")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    return True


def test_all_criteria():
    """测试所有5种criteria"""
    print("\n" + "="*60)
    print("测试2: 所有5种Criteria")
    print("="*60)
    
    model = vit_tiny(num_classes=10, img_size=32, pretrained=False)
    trained_weight = model.state_dict()
    
    # 创建假的current_mask
    current_mask = {}
    for layer_idx in range(len(model.blocks)):
        qkv_weight = trained_weight[f'blocks.{layer_idx}.attn.qkv.weight']
        current_mask[f'blocks.{layer_idx}.attn.qkv.weight_mask'] = torch.ones_like(qkv_weight)
    
    criteria_list = ['remain', 'magnitude', 'l1', 'l2']
    
    for criteria in criteria_list:
        print(f"\n测试 {criteria} criteria:")
        try:
            head_importance = vit_structured_pruning.compute_vit_head_importance(
                model=model,
                criteria=criteria,
                current_mask=current_mask if criteria == 'remain' else None,
                trained_weight=trained_weight
            )
            print(f"  ✓ {criteria} 计算成功，共 {len(head_importance)} 层")
        except Exception as e:
            print(f"  ✗ {criteria} 计算失败: {e}")
            return False
    
    return True


def test_different_prune_ratios():
    """测试不同的剪枝率"""
    print("\n" + "="*60)
    print("测试3: 不同剪枝率")
    print("="*60)
    
    prune_ratios = [0.2, 0.33, 0.5]
    
    for ratio in prune_ratios:
        print(f"\n测试剪枝率: {ratio:.0%}")
        
        model = vit_tiny(num_classes=10, img_size=32, pretrained=False)
        trained_weight = model.state_dict()
        
        # 计算重要性
        head_importance = vit_structured_pruning.compute_vit_head_importance(
            model=model,
            criteria='magnitude',
            trained_weight=trained_weight
        )
        
        # 执行剪枝
        vit_structured_pruning.structured_prune_vit_heads(
            model=model,
            head_importance=head_importance,
            prune_ratio=ratio
        )
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 32, 32)
        try:
            output = model(dummy_input)
            print(f"  ✓ 剪枝率 {ratio:.0%} 测试成功")
        except Exception as e:
            print(f"  ✗ 剪枝率 {ratio:.0%} 测试失败: {e}")
            return False
    
    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*60)
    print("测试4: 边界情况")
    print("="*60)
    
    model = vit_tiny(num_classes=10, img_size=32, pretrained=False)
    trained_weight = model.state_dict()
    
    # 测试1: 极小的剪枝率
    print("\n测试极小剪枝率 (5%):")
    head_importance = vit_structured_pruning.compute_vit_head_importance(
        model=model,
        criteria='magnitude',
        trained_weight=trained_weight
    )
    
    try:
        vit_structured_pruning.structured_prune_vit_heads(
            model=model,
            head_importance=head_importance,
            prune_ratio=0.05
        )
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model(dummy_input)
        print("  ✓ 极小剪枝率测试成功")
    except Exception as e:
        print(f"  ✗ 极小剪枝率测试失败: {e}")
        return False
    
    # 测试2: 接近100%的剪枝率（应该至少保留1个head）
    print("\n测试极大剪枝率 (90%):")
    model2 = vit_tiny(num_classes=10, img_size=32, pretrained=False)
    trained_weight2 = model2.state_dict()
    
    head_importance2 = vit_structured_pruning.compute_vit_head_importance(
        model=model2,
        criteria='magnitude',
        trained_weight=trained_weight2
    )
    
    try:
        vit_structured_pruning.structured_prune_vit_heads(
            model=model2,
            head_importance=head_importance2,
            prune_ratio=0.90
        )
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model2(dummy_input)
        print("  ✓ 极大剪枝率测试成功（自动保留至少1个head）")
    except Exception as e:
        print(f"  ✗ 极大剪枝率测试失败: {e}")
        return False
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# ViT结构化剪枝功能测试")
    print("#"*60)
    
    all_passed = True
    
    # 运行所有测试
    tests = [
        ("基本结构化剪枝", test_basic_structure),
        ("所有Criteria", test_all_criteria),
        ("不同剪枝率", test_different_prune_ratios),
        ("边界情况", test_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} 测试出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            all_passed = False
    
    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print("\n" + "#"*60)
    if all_passed:
        print("# ✓ 所有测试通过！")
    else:
        print("# ✗ 部分测试失败")
    print("#"*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
