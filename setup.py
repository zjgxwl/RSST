from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension
import torch

setup(
    name='conv_ops',
    ext_modules=[
        CUDAExtension(
            name='conv_ops',
            sources=['conv_skip_zero_channels.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '-arch=sm_80']  # 确保与你的 GPU 架构匹配
            }
        ),
    ],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)