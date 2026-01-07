import torch
from torch.utils.cpp_extension import load

# 加载自定义的 CUDA 扩展
conv_ops = load(name="conv_ops", sources=["conv_skip_zero_channels.cu"])

# 定义一个函数来调用我们自定义的 CUDA 算子
def conv_skip_zero(input, weights, stride=1, padding=0):
    output = torch.zeros_like(input)  # Initialize the output tensor
    conv_ops.conv_skip_zero_channels(input, output, weights, stride, padding)
    return output
