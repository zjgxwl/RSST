"""
Mamba模型快速测试
验证基本功能是否正常
"""
import torch
import sys

def test_import():
    """测试导入"""
    print("\n[1/5] 测试导入模块...")
    try:
        from models.mamba import mamba_small, mamba_tiny, mamba_base
        import mamba_structured_pruning as msp
        print("  ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n[2/5] 测试模型创建...")
    try:
        from models.mamba import mamba_small
        model = mamba_small(num_classes=10, img_size=32)
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Mamba-Small创建成功")
        print(f"  ✓ 参数量: {params:,}")
        return model
    except Exception as e:
        print(f"  ✗ 创建失败: {e}")
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n[3/5] 测试前向传播...")
    try:
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        print(f"  ✓ 输入: {x.shape}")
        print(f"  ✓ 输出: {y.shape}")
        assert y.shape == (2, 10), f"输出形状错误: {y.shape}"
        print("  ✓ 前向传播正常")
        return True
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pruning(model):
    """测试剪枝功能"""
    print("\n[4/5] 测试结构化剪枝...")
    try:
        import mamba_structured_pruning as msp
        from copy import deepcopy
        
        model_copy = deepcopy(model)
        params_before = sum(p.numel() for p in model_copy.parameters())
        
        # 测试MLP剪枝（更安全，不会影响SSM）
        info = msp.prune_mamba_mlp_structured(model_copy, prune_ratio=0.5, method='global')
        
        params_after = sum(p.numel() for p in model_copy.parameters())
        reduction = (1 - params_after / params_before) * 100
        
        print(f"  ✓ 原始参数: {params_before:,}")
        print(f"  ✓ 剪枝后: {params_after:,}")
        print(f"  ✓ 减少: {reduction:.2f}%")
        
        # 测试剪枝后的前向传播
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            y = model_copy(x)
        assert y.shape == (2, 10), "剪枝后输出形状错误"
        print("  ✓ 剪枝后前向传播正常")
        
        return True
    except Exception as e:
        print(f"  ✗ 剪枝失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rsst_regularization(model):
    """测试RSST正则化"""
    print("\n[5/5] 测试RSST正则化...")
    try:
        import mamba_structured_pruning as msp
        
        # 计算正则化损失
        reg_loss = msp.compute_mamba_structured_regularization(
            model, reg_strength=1e-4, reg_target='both'
        )
        
        print(f"  ✓ 正则化损失: {reg_loss.item():.6f}")
        assert reg_loss.item() > 0, "正则化损失应该大于0"
        
        # 测试schedule
        strength = msp.rsst_schedule_exp(30, 160, 1e-4, 4)
        print(f"  ✓ Epoch 30 强度: {strength:.8f}")
        
        print("  ✓ RSST正则化正常")
        return True
    except Exception as e:
        print(f"  ✗ RSST测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Mamba模型快速测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: 导入
    if not test_import():
        print("\n❌ 导入测试失败，停止后续测试")
        return False
    results.append(True)
    
    # 测试2: 创建模型
    model = test_model_creation()
    if model is None:
        print("\n❌ 模型创建失败，停止后续测试")
        return False
    results.append(True)
    
    # 测试3: 前向传播
    if not test_forward_pass(model):
        print("\n❌ 前向传播失败，停止后续测试")
        return False
    results.append(True)
    
    # 测试4: 剪枝
    if not test_pruning(model):
        print("\n⚠️  剪枝测试失败（其他功能正常）")
        results.append(False)
    else:
        results.append(True)
    
    # 测试5: RSST
    if not test_rsst_regularization(model):
        print("\n⚠️  RSST测试失败（其他功能正常）")
        results.append(False)
    else:
        results.append(True)
    
    # 总结
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有测试通过！Mamba模型可以正常使用。")
        print("\n下一步:")
        print("  1. 运行完整测试: python test_mamba_structured_pruning.py")
        print("  2. 启动实验: ./run_mamba_small_70p_all.sh")
    elif passed >= 3:
        print("⚠️  核心功能正常，部分高级功能有问题")
        print("可以尝试运行基础实验")
    else:
        print("❌ 测试失败，请检查环境配置")
        return False
    
    print("=" * 60)
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
