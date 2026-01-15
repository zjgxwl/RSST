"""
检查vit_small模型是否支持预训练
"""
import torch
from models.vit import vit_small

print("="*60)
print("检查 vit_small 模型预训练支持")
print("="*60)

# 测试1: 不使用预训练
print("\n测试1: 不使用预训练")
try:
    model = vit_small(num_classes=10, img_size=32, pretrained=False)
    print("  ✓ 成功创建模型（无预训练）")
    print(f"  ✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试2: 使用预训练（CIFAR-10）
print("\n测试2: 使用预训练（CIFAR-10）")
try:
    model = vit_small(num_classes=10, img_size=32, pretrained=True)
    print("  ✓ 成功加载预训练模型")
    print(f"  ✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试3: 使用预训练（CIFAR-100）
print("\n测试3: 使用预训练（CIFAR-100）")
try:
    model = vit_small(num_classes=100, img_size=32, pretrained=True)
    print("  ✓ 成功加载预训练模型")
    print(f"  ✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 检查模型结构
print("\n模型结构信息:")
model = vit_small(num_classes=10, img_size=32, pretrained=False)
print(f"  - Embed dim: {model.blocks[0].attn.qkv.in_features}")
print(f"  - Num blocks: {len(model.blocks)}")
print(f"  - Num heads: {model.blocks[0].attn.num_heads}")
print(f"  - MLP hidden dim: {model.blocks[0].mlp.fc1.out_features}")

print("\n" + "="*60)
print("检查完成！")
print("="*60)
