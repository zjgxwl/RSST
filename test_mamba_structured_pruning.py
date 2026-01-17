"""
测试Mamba模型的结构化剪枝功能
"""
import torch
import torch.nn as nn
from models.mamba import mamba_small, mamba_tiny
import mamba_structured_pruning as msp


def test_basic_forward():
    """测试基本前向传播"""
    print("\n[Test 1] 基本前向传播")
    model = mamba_small(num_classes=10, img_size=32)
    x = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (2, 10), f"Output shape mismatch: {y.shape}"
    print(f"  ✓ Input: {x.shape}, Output: {y.shape}")


def test_model_identification():
    """测试模型识别"""
    print("\n[Test 2] 模型识别")
    model_mamba = mamba_small(num_classes=10)
    
    is_mamba = msp.is_mamba_model(model_mamba)
    assert is_mamba, "Failed to identify Mamba model"
    print(f"  ✓ Mamba模型识别成功")


def test_ssm_pruning():
    """测试SSM结构化剪枝"""
    print("\n[Test 3] SSM结构化剪枝")
    model = mamba_small(num_classes=10, img_size=32)
    
    # 获取原始参数量
    params_before = sum(p.numel() for p in model.parameters())
    print(f"  原始参数量: {params_before:,}")
    
    # 剪枝（70%稀疏度）
    info = msp.prune_mamba_ssm_structured(model, prune_ratio=0.7, method='global')
    
    # 获取剪枝后参数量
    params_after = sum(p.numel() for p in model.parameters())
    print(f"  剪枝后参数量: {params_after:,}")
    print(f"  参数减少: {(1 - params_after/params_before)*100:.2f}%")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (2, 10), "Output shape mismatch after pruning"
    print(f"  ✓ 剪枝后前向传播正常")


def test_mlp_pruning():
    """测试MLP结构化剪枝"""
    print("\n[Test 4] MLP结构化剪枝")
    model = mamba_small(num_classes=10, img_size=32)
    
    params_before = sum(p.numel() for p in model.parameters())
    print(f"  原始参数量: {params_before:,}")
    
    # 剪枝MLP
    info = msp.prune_mamba_mlp_structured(model, prune_ratio=0.7, method='global')
    
    params_after = sum(p.numel() for p in model.parameters())
    print(f"  剪枝后参数量: {params_after:,}")
    print(f"  参数减少: {(1 - params_after/params_before)*100:.2f}%")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (2, 10), "Output shape mismatch after pruning"
    print(f"  ✓ 剪枝后前向传播正常")


def test_hybrid_pruning():
    """测试混合剪枝（SSM + MLP）"""
    print("\n[Test 5] 混合剪枝（SSM + MLP）")
    model = mamba_small(num_classes=10, img_size=32)
    
    params_before = sum(p.numel() for p in model.parameters())
    print(f"  原始参数量: {params_before:,}")
    
    # 混合剪枝
    info = msp.prune_mamba_hybrid(model, ssm_ratio=0.7, mlp_ratio=0.7, method='global')
    
    params_after = sum(p.numel() for p in model.parameters())
    print(f"  剪枝后参数量: {params_after:,}")
    print(f"  参数减少: {(1 - params_after/params_before)*100:.2f}%")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (2, 10), "Output shape mismatch after pruning"
    print(f"  ✓ 剪枝后前向传播正常")


def test_rsst_regularization():
    """测试RSST正则化"""
    print("\n[Test 6] RSST正则化")
    model = mamba_small(num_classes=10, img_size=32)
    
    # 计算正则化损失
    reg_loss = msp.compute_mamba_structured_regularization(
        model, reg_strength=1e-4, reg_target='both'
    )
    
    assert reg_loss.item() > 0, "Regularization loss should be positive"
    print(f"  ✓ RSST正则化损失: {reg_loss.item():.6f}")
    
    # 测试不同target
    reg_ssm = msp.compute_mamba_structured_regularization(
        model, reg_strength=1e-4, reg_target='ssm'
    )
    reg_mlp = msp.compute_mamba_structured_regularization(
        model, reg_strength=1e-4, reg_target='mlp'
    )
    
    print(f"  ✓ SSM正则化: {reg_ssm.item():.6f}")
    print(f"  ✓ MLP正则化: {reg_mlp.item():.6f}")


def test_rsst_schedule():
    """测试RSST动态schedule"""
    print("\n[Test 7] RSST动态schedule")
    
    epochs_test = [0, 30, 60, 90, 160]
    total_epochs = 160
    base_strength = 1e-4
    exponent = 4
    
    print(f"  基础强度: {base_strength}, 指数: {exponent}")
    for epoch in epochs_test:
        strength = msp.rsst_schedule_exp(epoch, total_epochs, base_strength, exponent)
        print(f"  Epoch {epoch:3d}: {strength:.8f}")
    
    print(f"  ✓ Schedule测试通过")


def test_layerwise_vs_global():
    """测试layer-wise vs global剪枝策略"""
    print("\n[Test 8] Layer-wise vs Global剪枝策略对比")
    
    # Global pruning
    model_global = mamba_tiny(num_classes=10, img_size=32)
    params_before = sum(p.numel() for p in model_global.parameters())
    info_global = msp.prune_mamba_mlp_structured(model_global, prune_ratio=0.7, method='global')
    params_global = sum(p.numel() for p in model_global.parameters())
    
    # Layer-wise pruning
    model_layerwise = mamba_tiny(num_classes=10, img_size=32)
    info_layerwise = msp.prune_mamba_mlp_structured(model_layerwise, prune_ratio=0.7, method='layerwise')
    params_layerwise = sum(p.numel() for p in model_layerwise.parameters())
    
    print(f"  原始: {params_before:,}")
    print(f"  Global: {params_global:,} ({(1-params_global/params_before)*100:.2f}% 减少)")
    print(f"  Layer-wise: {params_layerwise:,} ({(1-params_layerwise/params_before)*100:.2f}% 减少)")
    print(f"  ✓ 两种策略都工作正常")


def test_gradient_flow():
    """测试剪枝后的梯度流"""
    print("\n[Test 9] 梯度流测试")
    model = mamba_small(num_classes=10, img_size=32)
    
    # 剪枝
    msp.prune_mamba_hybrid(model, ssm_ratio=0.5, mlp_ratio=0.5)
    
    # 前向+反向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients found after backward pass"
    print(f"  ✓ 梯度流正常")


if __name__ == '__main__':
    print("=" * 70)
    print("测试Mamba模型的结构化剪枝功能")
    print("=" * 70)
    
    try:
        test_basic_forward()
        test_model_identification()
        test_ssm_pruning()
        test_mlp_pruning()
        test_hybrid_pruning()
        test_rsst_regularization()
        test_rsst_schedule()
        test_layerwise_vs_global()
        test_gradient_flow()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
