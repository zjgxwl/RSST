"""
快速测试ViT + RSST在CIFAR-10上的完整流程
用于验证流程是否通畅，不追求最终精度
"""
import os
import sys
import torch


def test_flow():
    """测试完整流程"""
    
    print("=" * 80)
    print("ViT + RSST on CIFAR-10 流程测试")
    print("=" * 80)
    print()
    
    # 测试1: 导入检查
    print("【测试1/6】检查依赖库")
    print("-" * 80)
    try:
        import torch
        import torchvision
        import timm
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Torchvision: {torchvision.__version__}")
        print(f"✓ timm: {timm.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        return False
    print()
    
    # 测试2: 模型加载
    print("【测试2/6】加载预训练ViT模型")
    print("-" * 80)
    try:
        from models.vit import vit_tiny
        model = vit_tiny(num_classes=10, img_size=32, pretrained=True)
        print(f"✓ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 总参数量: {total_params:,}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    print()
    
    # 测试3: 数据加载
    print("【测试3/6】加载CIFAR-10数据集")
    print("-" * 80)
    try:
        from dataset import cifar10_dataloaders
        train_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size=32, 
            data_dir='data'
        )
        print(f"✓ 训练集batch数: {len(train_loader)}")
        print(f"✓ 验证集batch数: {len(val_loader)}")
        print(f"✓ 测试集batch数: {len(test_loader)}")
        
        # 测试一个batch
        images, labels = next(iter(train_loader))
        print(f"✓ Batch形状: {images.shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    print()
    
    # 测试4: 前向传播
    print("【测试4/6】测试前向传播")
    print("-" * 80)
    try:
        model.eval()
        with torch.no_grad():
            output = model(images)
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 预期形状: (32, 10)")
        assert output.shape == (32, 10), "输出形状不正确!"
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    print()
    
    # 测试5: 剪枝功能
    print("【测试5/6】测试剪枝功能")
    print("-" * 80)
    try:
        import vit_pruning_utils
        
        # 检查是否识别为ViT
        is_vit = vit_pruning_utils.is_vit_model(model)
        print(f"✓ 模型识别: {'ViT' if is_vit else 'CNN'}")
        assert is_vit, "模型未被识别为ViT!"
        
        # 检查初始稀疏度
        print("\n初始稀疏度:")
        remain_before = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
        
        # 执行剪枝
        print(f"\n执行20%剪枝...")
        vit_pruning_utils.pruning_model_vit(model, px=0.2, prune_patch_embed=False)
        
        # 检查剪枝后稀疏度
        print("\n剪枝后稀疏度:")
        remain_after = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
        
        print(f"\n✓ 剪枝前剩余: {remain_before:.2f}%")
        print(f"✓ 剪枝后剩余: {remain_after:.2f}%")
        print(f"✓ 实际剪掉: {remain_before - remain_after:.2f}%")
        
    except Exception as e:
        print(f"✗ 剪枝功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 测试6: 剪枝后推理
    print("【测试6/6】测试剪枝后推理")
    print("-" * 80)
    try:
        model.eval()
        with torch.no_grad():
            output = model(images)
        print(f"✓ 剪枝后输出形状: {output.shape}")
        print(f"✓ 输出无NaN: {not torch.isnan(output).any()}")
        print(f"✓ 输出无Inf: {not torch.isinf(output).any()}")
    except Exception as e:
        print(f"✗ 剪枝后推理失败: {e}")
        return False
    print()
    
    return True


def run_quick_training():
    """运行快速训练测试（1轮剪枝，2个epoch）"""
    print("=" * 80)
    print("运行快速训练测试")
    print("=" * 80)
    print()
    
    print("配置:")
    print("  - 数据集: CIFAR-10")
    print("  - 模型: ViT-Tiny (预训练)")
    print("  - 剪枝次数: 2轮")
    print("  - 每轮epoch: 2")
    print("  - Batch size: 64")
    print()
    
    cmd = """
python main_imp_fillback.py \\
    --dataset cifar10 \\
    --arch vit_tiny \\
    --pretrained \\
    --struct rsst \\
    --criteria l1 \\
    --epochs 2 \\
    --batch_size 64 \\
    --lr 0.001 \\
    --warmup 1 \\
    --pruning_times 2 \\
    --rate 0.2 \\
    --RST_schedule exp_custom_exponents \\
    --reg_granularity_prune 0.5 \\
    --exponents 3 \\
    --save_dir test_output/vit_rsst_flow_test \\
    --seed 42
    """.strip()
    
    print("运行命令:")
    print(cmd)
    print()
    print("=" * 80)
    print("开始训练... (预计5-10分钟)")
    print("=" * 80)
    
    import subprocess
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode == 0


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("ViT + RSST 流程快速测试")
    print("🧪" * 40 + "\n")
    
    # 阶段1: 组件测试
    print("阶段1: 组件功能测试")
    print("=" * 80)
    success = test_flow()
    
    if not success:
        print("\n" + "❌" * 40)
        print("流程测试失败！请检查错误信息")
        print("❌" * 40 + "\n")
        sys.exit(1)
    
    print("=" * 80)
    print("✓✓✓ 所有组件测试通过！ ✓✓✓")
    print("=" * 80)
    print()
    
    # 询问是否运行完整训练测试
    print("=" * 80)
    print("阶段2: 完整训练流程测试（可选）")
    print("=" * 80)
    print()
    print("是否运行完整训练测试？")
    print("  - 这将运行2轮剪枝，每轮2个epoch")
    print("  - 预计耗时: 5-10分钟")
    print("  - 目的: 验证训练循环无报错")
    print()
    
    response = input("运行完整测试? [y/N]: ").strip().lower()
    
    if response == 'y':
        print()
        success = run_quick_training()
        
        if success:
            print("\n" + "✅" * 40)
            print("完整流程测试通过！")
            print("✅" * 40 + "\n")
            print("现在可以运行正式实验了:")
            print()
            print("python main_imp_fillback.py \\")
            print("    --dataset cifar10 \\")
            print("    --arch vit_tiny \\")
            print("    --pretrained \\")
            print("    --struct rsst \\")
            print("    --epochs 80 \\")
            print("    --pruning_times 15")
            print()
        else:
            print("\n" + "❌" * 40)
            print("训练流程测试失败！")
            print("❌" * 40 + "\n")
            sys.exit(1)
    else:
        print()
        print("跳过完整训练测试")
        print()
        print("手动运行完整测试:")
        print()
        print("python main_imp_fillback.py \\")
        print("    --dataset cifar10 \\")
        print("    --arch vit_tiny \\")
        print("    --pretrained \\")
        print("    --struct rsst \\")
        print("    --epochs 2 \\")
        print("    --pruning_times 2 \\")
        print("    --batch_size 64")
        print()

