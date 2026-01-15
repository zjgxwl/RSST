"""
测试ViT预训练模型加载功能
"""
import torch
from models.vit import vit_tiny, vit_small, vit_base


def test_pretrained_loading():
    """测试预训练模型加载"""
    print("=" * 80)
    print("测试ViT预训练模型加载功能")
    print("=" * 80)
    print()
    
    # 测试1: 不使用预训练（默认）
    print("【测试1】随机初始化模型")
    print("-" * 80)
    model_scratch = vit_small(num_classes=100, img_size=32, pretrained=False)
    print("✓ 随机初始化成功")
    print()
    
    # 测试2: 使用预训练
    print("【测试2】加载预训练模型")
    print("-" * 80)
    try:
        model_pretrained = vit_small(num_classes=100, img_size=32, pretrained=True)
        print("✓ 预训练模型加载流程完成")
    except Exception as e:
        print(f"✗ 预训练模型加载失败: {e}")
    print()
    
    # 测试3: vit_tiny (无预训练)
    print("【测试3】ViT-Tiny（无预训练权重）")
    print("-" * 80)
    model_tiny = vit_tiny(num_classes=100, img_size=32, pretrained=True)
    print("✓ ViT-Tiny创建成功")
    print()
    
    # 测试4: 前向传播
    print("【测试4】测试前向传播")
    print("-" * 80)
    model_scratch.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model_scratch(x)
    print(f"✓ 输入: {x.shape}")
    print(f"✓ 输出: {output.shape}")
    assert output.shape == (2, 100), "输出形状错误!"
    print("✓ 前向传播测试通过")
    print()
    
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print()
    print("📝 使用说明：")
    print("  1. 不使用预训练：")
    print("     python main_imp_fillback.py --arch vit_small")
    print()
    print("  2. 使用预训练（需要先安装timm）：")
    print("     pip install timm")
    print("     python main_imp_fillback.py --arch vit_small --pretrained")
    print()
    print("详细文档请查看：ViT预训练模型使用说明.md")
    print()


if __name__ == '__main__':
    test_pretrained_loading()

