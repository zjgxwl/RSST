#!/bin/bash
# 激活structlth环境的快捷脚本

source /root/miniconda3/etc/profile.d/conda.sh
conda activate structlth

echo "=================================="
echo "✅ structlth环境已激活"
echo "=================================="
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "代理设置: $http_proxy"
echo "=================================="
echo ""
echo "快速开始："
echo "  python test_vit_quasi_structured.py  # 测试功能"
echo "  python main_imp_fillback.py --help   # 查看帮助"
echo "=================================="
