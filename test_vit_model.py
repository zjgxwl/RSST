"""
ViT模型测试脚本
用于验证ViT模型是否正确加载和剪枝功能是否正常
"""
import torch
import argparse
from models.vit import vit_tiny, vit_small, vit_base
import vit_pruning_utils


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 80)
    print("测试1: 模型前向传播")
    print("-" * 80)
    
    # 创建模型
    model = vit_tiny(num_classes=100, img_size=32)
    model.eval()
    
    # 创建随机输入
    x = torch.randn(4, 3, 32, 32)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 预期输出形状: (4, 100)")
    
    assert output.shape == (4, 100), "输出形状不正确!"
    print("✓ 测试通过!")
    print()


def test_model_parameters():
    """测试模型参数统计"""
    print("=" * 80)
    print("测试2: 模型参数统计")
    print("-" * 80)
    
    models = {
        'vit_tiny': vit_tiny(num_classes=100, img_size=32),
        'vit_small': vit_small(num_classes=100, img_size=32),
        'vit_base': vit_base(num_classes=100, img_size=32),
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        print()
    
    print("✓ 测试通过!")
    print()


def test_pruning_functionality():
    """测试剪枝功能"""
    print("=" * 80)
    print("测试3: 剪枝功能")
    print("-" * 80)
    
    # 创建模型
    model = vit_tiny(num_classes=100, img_size=32)
    
    # 检查初始稀疏度
    print("初始状态:")
    remain_before = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
    print()
    
    # 执行剪枝
    print("执行剪枝 (rate=0.2):")
    vit_pruning_utils.pruning_model_vit(model, px=0.2, prune_patch_embed=False)
    remain_after = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
    print()
    
    # 验证剪枝效果
    expected_remain = 80.0  # 剪掉20%，剩余80%
    tolerance = 5.0  # 允许5%误差
    
    print(f"剪枝前剩余权重: {remain_before:.2f}%")
    print(f"剪枝后剩余权重: {remain_after:.2f}%")
    print(f"预期剩余权重: ~{expected_remain:.2f}%")
    
    assert abs(remain_after - expected_remain) < tolerance, f"剪枝效果不符合预期!"
    print("✓ 测试通过!")
    print()


def test_mask_extraction():
    """测试mask提取功能"""
    print("=" * 80)
    print("测试4: Mask提取功能")
    print("-" * 80)
    
    # 创建并剪枝模型
    model = vit_tiny(num_classes=100, img_size=32)
    vit_pruning_utils.pruning_model_vit(model, px=0.3, prune_patch_embed=False)
    
    # 提取mask
    mask_dict = vit_pruning_utils.extract_mask_vit(model.state_dict())
    
    print(f"提取到 {len(mask_dict)} 个mask")
    print("\nMask列表:")
    for i, (key, mask) in enumerate(mask_dict.items(), 1):
        sparsity = (mask == 0).float().mean() * 100
        print(f"  {i}. {key}: shape={mask.shape}, sparsity={sparsity:.2f}%")
    
    assert len(mask_dict) > 0, "未提取到任何mask!"
    print("\n✓ 测试通过!")
    print()


def test_model_inference_after_pruning():
    """测试剪枝后模型推理"""
    print("=" * 80)
    print("测试5: 剪枝后模型推理")
    print("-" * 80)
    
    # 创建模型并剪枝
    model = vit_tiny(num_classes=100, img_size=32)
    vit_pruning_utils.pruning_model_vit(model, px=0.5, prune_patch_embed=False)
    model.eval()
    
    # 创建输入
    x = torch.randn(2, 3, 32, 32)
    
    # 推理
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 输出包含NaN: {torch.isnan(output).any().item()}")
    print(f"✓ 输出包含Inf: {torch.isinf(output).any().item()}")
    
    assert not torch.isnan(output).any(), "输出包含NaN!"
    assert not torch.isinf(output).any(), "输出包含Inf!"
    print("✓ 测试通过!")
    print()


def test_is_vit_model():
    """测试模型类型判断"""
    print("=" * 80)
    print("测试6: 模型类型判断")
    print("-" * 80)
    
    from models.resnets import resnet20
    
    vit_model = vit_tiny(num_classes=100)
    cnn_model = resnet20(number_class=100)
    
    is_vit_1 = vit_pruning_utils.is_vit_model(vit_model)
    is_vit_2 = vit_pruning_utils.is_vit_model(cnn_model)
    
    print(f"ViT模型判断: {is_vit_1} (应该是True)")
    print(f"CNN模型判断: {is_vit_2} (应该是False)")
    
    assert is_vit_1 == True, "ViT模型判断错误!"
    assert is_vit_2 == False, "CNN模型判断错误!"
    print("✓ 测试通过!")
    print()


def main():
    print("\n" + "=" * 80)
    print("ViT模型和剪枝功能测试套件")
    print("=" * 80 + "\n")
    
    try:
        test_model_forward()
        test_model_parameters()
        test_pruning_functionality()
        test_mask_extraction()
        test_model_inference_after_pruning()
        test_is_vit_model()
        
        print("=" * 80)
        print("✓✓✓ 所有测试通过! ✓✓✓")
        print("=" * 80)
        print("\n您可以开始使用ViT进行RSST剪枝实验了!")
        print("运行命令: bash run_vit_rsst.sh")
        print()
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗✗✗ 测试失败! ✗✗✗")
        print("=" * 80)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == '__main__':
    main()

