import argparse  # 导入解析命令行参数的库
import torch as th  # 导入PyTorch库并重命名为th
import re  # 导入正则表达式库
import io  # 导入io库
import datetime  # 导入datetime库
import os  # 导入操作系统相关的库
import ctypes  # 导入ctypes库，用于调用C语言库

# 创建解析命令行参数的对象
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument('--input_height', type=int, default=127)  # 输入图像的高度，默认为127
parser.add_argument('--input_width', type=int, default=127)  # 输入图像的宽度，默认为127
parser.add_argument('--input_channel', type=int, default=64)  # 输入图像的通道数，默认为64
parser.add_argument('--batch_size', type=int, default=64)  # 批大小，默认为64
parser.add_argument('--nkernel', type=int, default=64)  # 卷积核的数量，默认为64
parser.add_argument('--kernel_height', type=int, default=3)  # 卷积核的高度，默认为3
parser.add_argument('--kernel_width', type=int, default=3)  # 卷积核的宽度，默认为3
parser.add_argument('--vertical_stride', type=int, default=1)  # 垂直步长，默认为1
parser.add_argument('--horizontal_stride', type=int, default=1)  # 水平步长，默认为1
parser.add_argument('--vertical_dilation', type=int, default=1)  # 垂直膨胀，默认为1
parser.add_argument('--horizontal_dilation', type=int, default=1)  # 水平膨胀，默认为1
parser.add_argument('--vertical_padding', type=int, default=1)  # 垂直填充，默认为1
parser.add_argument('--horizontal_padding', type=int, default=1)  # 水平填充，默认为1
parser.add_argument('--cuda_file', default='cudnn_conv.cu')  # CUDA文件，默认为'cudnn_conv.cu'
parser.add_argument('--kernel_file', default=None)  # 卷积核文件，默认为None

# 解析命令行参数
args = parser.parse_args()

# 将命令行参数赋值给对应的变量
input_height = args.input_height
input_width = args.input_width
input_channel = args.input_channel
batch_size = args.batch_size
nkernel = args.nkernel
kernel_height = args.kernel_height
kernel_width = args.kernel_width
vertical_stride = args.vertical_stride
horizontal_stride = args.horizontal_stride
vertical_dilation = args.vertical_dilation
horizontal_dilation = args.horizontal_dilation
vertical_padding = args.vertical_padding
horizontal_padding = args.horizontal_padding

kernel_file = args.kernel_file

# 加载稀疏卷积核
sparse_kernel = th.load(kernel_file)

# 获取卷积核的形状
kernel_shape = sparse_kernel.shape
nkernel = kernel_shape[0]
input_channel = kernel_shape[1]
kernel_height = kernel_shape[2]
kernel_width = kernel_shape[3]

# 根据卷积核和输入图像的形状计算输出图像的形状
assert (kernel_height % 2 == 1 and kernel_width % 2 == 1)  # 断言卷积核的高度和宽度为奇数
tmp_kernel_height = kernel_height + (kernel_height - 1) * (vertical_dilation - 1)
tmp_kernel_width = kernel_width + (kernel_width - 1) * (horizontal_dilation - 1)
tmp = input_height - tmp_kernel_height + 2 * vertical_padding
output_height = tmp // vertical_stride + 1
tmp = input_width - tmp_kernel_width + 2 * horizontal_padding
output_width = tmp // horizontal_stride + 1
output_channels = nkernel

# 读取CUDA文件并替换其中的特定字符串
with open(cuda_file, 'r') as f:
    code = f.read()
    code = code.replace('S_kernels', str(nkernel)).replace('S_channels', str(input_channel)).replace(
        'S_kernel_height', str(kernel_height)).replace('S_kernel_width', str(kernel_width)).replace(
        'S_vertical_stride', str(vertical_stride)).replace('S_horizontal_stride', str(horizontal_stride)).replace(
        'S_dilation_height', str(vertical_dilation)).replace('S_dilation_width', str(horizontal_dilation)).replace(
        'S_padding_height', str(vertical_padding)).replace('S_padding_width', str(horizontal_padding)).replace(
        'S_batch_size', str(batch_size)).replace('S_input_height', str(input_height)).replace(
        'S_input_width', str(input_width))

# 生成临时CUDA文件
timestamp = datetime.datetime.now().time()
filename = f'.tmp/tmp_{timestamp}'
with open(filename + '.cu', 'w') as fw:
    fw.write(code)

# 编译CUDA文件并执行
os.system(f'cp Makefile .tmp/; cd .tmp; make; CUDA_VISIBLE_DEVICES=2 ./conv; cd ..')

# 删除临时文件
os.system(f'rm .tmp/*')
