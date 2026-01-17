"""
生成ViT-Small预训练初始化文件
使用timm库加载ImageNet预训练权重
"""
import torch
from models.vit import vit_small

print("="*80)
print("生成ViT-Small预训练初始化文件")
print("="*80)

# 生成CIFAR-10初始化文件
print("\n1. 生成 CIFAR-10 初始化文件...")
model_cifar10 = vit_small(num_classes=10, img_size=32, pretrained=True)
checkpoint = {
    'state_dict': model_cifar10.state_dict()
}
torch.save(checkpoint, 'init_model/vit_small_cifar10_pretrained_init.pth.tar')
print(f"✓ 已保存: init_model/vit_small_cifar10_pretrained_init.pth.tar")

# 验证权重std
test_weight = model_cifar10.state_dict()['blocks.0.attn.qkv.weight']
weight_std = test_weight.std().item()
print(f"  权重std: {weight_std:.6f} (预训练模型应该 > 0.05)")

# 生成CIFAR-100初始化文件
print("\n2. 生成 CIFAR-100 初始化文件...")
model_cifar100 = vit_small(num_classes=100, img_size=32, pretrained=True)
checkpoint = {
    'state_dict': model_cifar100.state_dict()
}
torch.save(checkpoint, 'init_model/vit_small_cifar100_pretrained_init.pth.tar')
print(f"✓ 已保存: init_model/vit_small_cifar100_pretrained_init.pth.tar")

# 验证权重std
test_weight = model_cifar100.state_dict()['blocks.0.attn.qkv.weight']
weight_std = test_weight.std().item()
print(f"  权重std: {weight_std:.6f} (预训练模型应该 > 0.05)")

print("\n" + "="*80)
print("✅ 预训练初始化文件生成完成！")
print("="*80)
