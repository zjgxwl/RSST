# 临时调试脚本：检查trained_weight的键名
import torch

# 模拟加载模型
checkpoint = torch.load('init_model/vit_small_cifar10_pretrained_init.pth.tar')
state_dict = checkpoint['state_dict']

print("========== state_dict中的attn.qkv相关键 ==========")
for key in state_dict.keys():
    if 'attn.qkv' in key:
        print(f"  {key}")
        
print("\n========== blocks.0相关的所有键 ==========")
for key in sorted(state_dict.keys()):
    if key.startswith('blocks.0'):
        print(f"  {key}")

