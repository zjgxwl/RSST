import torch
from conv_ops import conv_skip_zero_channels  # 导入自定义的 CUDA 扩展

# 生成随机输入
input_tensor = torch.randn(1, 3, 32, 32).cuda()
weights = torch.randn(3, 3, 3, 3).cuda()

# 运行自定义 CUDA 计算
output_tensor = torch.zeros_like(input_tensor).cuda()
conv_skip_zero_channels(input_tensor, output_tensor, weights, 1, 1)

print(output_tensor)

