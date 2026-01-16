"""
Vision Transformer for ImageNet (224x224)
用于在ImageNet上直接进行RSST剪枝
"""
import torch
import torch.nn as nn


def vit_small_imagenet(num_classes=1000, pretrained=False):
    """
    ViT-Small for ImageNet
    输入: 224x224
    Patch size: 16x16
    """
    print('创建 ViT-Small for ImageNet (224x224, patch=16)')
    
    if not pretrained:
        print("⚠️  警告: 不使用预训练的ViT在ImageNet上从头训练需要大量资源!")
        print("   建议使用 --pretrained 加载预训练权重")
    
    try:
        import timm
        
        if pretrained:
            # 加载完整的ImageNet预训练模型
            model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
            print("✓ 成功加载ImageNet预训练的ViT-Small")
            print(f"  - 输入尺寸: 224x224")
            print(f"  - Patch size: 16x16")
            print(f"  - 类别数: {num_classes}")
        else:
            # 创建随机初始化的模型
            model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
            print("✓ 创建随机初始化的ViT-Small (不推荐用于ImageNet)")
        
        return model
        
    except ImportError:
        print("✗ 错误: 需要安装timm库才能使用ImageNet ViT")
        print("   运行: pip install timm")
        raise ImportError("请先安装timm: pip install timm")
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        raise


def vit_base_imagenet(num_classes=1000, pretrained=False):
    """
    ViT-Base for ImageNet
    输入: 224x224
    Patch size: 16x16
    """
    print('创建 ViT-Base for ImageNet (224x224, patch=16)')
    
    if not pretrained:
        print("⚠️  警告: 不使用预训练的ViT在ImageNet上从头训练需要大量资源!")
    
    try:
        import timm
        
        if pretrained:
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
            print("✓ 成功加载ImageNet预训练的ViT-Base")
            print(f"  - 输入尺寸: 224x224")
            print(f"  - Patch size: 16x16")
            print(f"  - 类别数: {num_classes}")
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            print("✓ 创建随机初始化的ViT-Base")
        
        return model
        
    except ImportError:
        print("✗ 错误: 需要安装timm库")
        print("   运行: pip install timm")
        raise ImportError("请先安装timm: pip install timm")
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        raise


def vit_large_imagenet(num_classes=1000, pretrained=False):
    """
    ViT-Large for ImageNet
    输入: 224x224
    Patch size: 16x16
    """
    print('创建 ViT-Large for ImageNet (224x224, patch=16)')
    
    if not pretrained:
        print("⚠️  警告: ViT-Large非常大，不建议从头训练!")
    
    try:
        import timm
        
        if pretrained:
            model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
            print("✓ 成功加载ImageNet预训练的ViT-Large")
            print(f"  - 输入尺寸: 224x224")
            print(f"  - Patch size: 16x16")
            print(f"  - 类别数: {num_classes}")
        else:
            model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes)
            print("✓ 创建随机初始化的ViT-Large")
        
        return model
        
    except ImportError:
        print("✗ 错误: 需要安装timm库")
        raise ImportError("请先安装timm: pip install timm")
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        raise


# 辅助函数：检查是否是ImageNet ViT
def is_imagenet_vit(model):
    """判断是否是timm的ViT模型"""
    try:
        import timm.models.vision_transformer as vit_module
        return isinstance(model, vit_module.VisionTransformer)
    except:
        return False

