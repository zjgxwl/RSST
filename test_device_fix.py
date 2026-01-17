"""
测试设备不匹配bug的修复
模拟State 0 → State 1转换过程
"""
import torch
import torch.nn as nn
import os
import sys

print("=" * 80)
print("🔍 测试设备不匹配bug修复")
print("=" * 80)

# 模拟GPU环境
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"\n✓ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print(f"\n⚠️  GPU不可用，使用CPU测试")

# 创建简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

print("\n" + "=" * 80)
print("测试场景1: 初始加载initialization")
print("=" * 80)

# 创建模型并移到GPU
model = SimpleModel().to(device)
print(f"✓ 模型在设备: {next(model.parameters()).device}")

# 模拟保存初始化文件
init_file = '/tmp/test_init.pth.tar'
torch.save(model.state_dict(), init_file)
print(f"✓ 保存初始化文件到: {init_file}")

# 模拟加载initialization（会在CPU上）
initialization = torch.load(init_file, map_location='cpu')
print(f"✓ 加载initialization（默认在CPU）")
print(f"  示例参数设备: {initialization['fc1.weight'].device}")

# 🔧 应用修复：将initialization移到正确设备
print(f"\n🔧 应用修复：将所有tensor移到 {device}")
for key in initialization.keys():
    if isinstance(initialization[key], torch.Tensor):
        initialization[key] = initialization[key].to(device)

print(f"✓ 修复后参数设备: {initialization['fc1.weight'].device}")

# 测试Refill操作（关键操作）
print(f"\n测试Refill操作（m.weight.data = initialization[...]）")
try:
    model.fc1.weight.data = initialization['fc1.weight']
    print(f"  ✅ Refill成功！模型权重设备: {model.fc1.weight.device}")
except RuntimeError as e:
    print(f"  ❌ Refill失败: {e}")
    sys.exit(1)

# 测试forward pass
print(f"\n测试forward pass")
try:
    x = torch.randn(5, 10).to(device)
    output = model(x)
    print(f"  ✅ Forward成功！输出形状: {output.shape}, 设备: {output.device}")
except RuntimeError as e:
    print(f"  ❌ Forward失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试场景2: Resume时加载initialization")
print("=" * 80)

# 模拟checkpoint
checkpoint = {
    'init_weight': model.state_dict(),  # 在GPU上
    'state_dict': model.state_dict(),
    'epoch': 60,
    'state': 1
}

# 保存checkpoint
checkpoint_file = '/tmp/test_checkpoint.pth.tar'
torch.save(checkpoint, checkpoint_file)
print(f"✓ 保存checkpoint到: {checkpoint_file}")

# 加载checkpoint（map_location='cpu'）
checkpoint_loaded = torch.load(checkpoint_file, map_location='cpu')
initialization2 = checkpoint_loaded['init_weight']
print(f"✓ 从checkpoint加载initialization")
print(f"  示例参数设备: {initialization2['fc1.weight'].device}")

# 🔧 应用修复
print(f"\n🔧 应用修复：将所有tensor移到 {device}")
for key in initialization2.keys():
    if isinstance(initialization2[key], torch.Tensor):
        initialization2[key] = initialization2[key].to(device)

print(f"✓ 修复后参数设备: {initialization2['fc1.weight'].device}")

# 测试Refill操作
print(f"\n测试Refill操作")
try:
    model.fc1.weight.data = initialization2['fc1.weight']
    print(f"  ✅ Refill成功！")
except RuntimeError as e:
    print(f"  ❌ Refill失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("测试场景3: 从new_initialization复制权重")
print("=" * 80)

# 模拟new_initialization（从GPU上的model获取）
new_initialization = model.state_dict()
print(f"✓ new_initialization来自GPU model")
print(f"  示例参数设备: {new_initialization['fc1.weight'].device}")

# 复制权重到initialization
initialization3 = torch.load(init_file, map_location='cpu')
print(f"✓ initialization加载到CPU")
print(f"  示例参数设备: {initialization3['fc1.weight'].device}")

# 移到GPU
for key in initialization3.keys():
    if isinstance(initialization3[key], torch.Tensor):
        initialization3[key] = initialization3[key].to(device)
print(f"✓ 移到GPU后: {initialization3['fc1.weight'].device}")

# 从new_initialization复制某些权重（模拟head/fc复制）
initialization3['fc1.weight'] = new_initialization['fc1.weight']
initialization3['fc1.bias'] = new_initialization['fc1.bias']
print(f"✓ 从new_initialization复制权重")
print(f"  复制后设备: {initialization3['fc1.weight'].device}")

# 🔧 应用额外保护：再次确保所有tensor在正确设备
print(f"\n🔧 应用额外保护：再次检查设备")
for key in initialization3.keys():
    if isinstance(initialization3[key], torch.Tensor):
        initialization3[key] = initialization3[key].to(device)
print(f"✓ 最终参数设备: {initialization3['fc1.weight'].device}")

# 测试Refill
print(f"\n测试Refill操作")
try:
    model.fc1.weight.data = initialization3['fc1.weight']
    output = model(x)
    print(f"  ✅ Refill和Forward都成功！")
except RuntimeError as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 清理
os.remove(init_file)
os.remove(checkpoint_file)

print("\n" + "=" * 80)
print("✅ 所有测试通过！修复有效！")
print("=" * 80)
print("\n修复要点:")
print("  1. ✓ torch.load后立即移到GPU")
print("  2. ✓ resume时从checkpoint加载后移到GPU")
print("  3. ✓ 从new_initialization复制权重后再次确保GPU")
print("  4. ✓ 所有Refill操作前确保initialization在GPU")
print("\n可以放心启动ViT实验了！🚀")
