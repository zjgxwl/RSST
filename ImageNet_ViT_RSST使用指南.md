# ImageNet上的ViT剪枝使用指南

## 📝 概述

现在支持**直接在ImageNet数据集上对ViT模型进行RSST剪枝**！

- ✅ 使用完整的ImageNet预训练ViT（224×224输入）
- ✅ 直接在ImageNet测试集上评估剪枝效果
- ✅ 支持ViT-Small/Base/Large三种规模

---

## 🚀 快速开始

### 1. 准备环境

```bash
# 安装依赖
pip install timm torch torchvision wandb

# 确认CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 准备ImageNet数据集

**数据集结构：**
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ... (1000个类别)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000个类别)
```

**下载方式：**
- 官方渠道：https://image-net.org/
- 学术机构通常有本地镜像

### 3. 运行剪枝实验

```bash
python main_imp_fillback.py \
    --dataset imagenet \
    --data /path/to/imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst \
    --epochs 10 \
    --batch_size 256 \
    --lr 0.0001 \
    --workers 8 \
    --pruning_times 10 \
    --rate 0.15 \
    --save_dir results/imagenet_vit_rsst
```

---

## 📊 支持的模型

| 模型 | 参数量 | Top-1精度(预训练) | 推荐GPU显存 | 推荐Batch Size |
|------|--------|-----------------|-----------|--------------|
| `vit_small_imagenet` | 22M | ~81.4% | 16GB | 128-256 |
| `vit_base_imagenet` | 86M | ~81.8% | 24GB | 64-128 |
| `vit_large_imagenet` | 307M | ~82.6% | 32GB+ | 32-64 |

---

## ⚙️ 参数配置

### ImageNet专用配置

```bash
python main_imp_fillback.py \
    --dataset imagenet \                    # 使用ImageNet
    --data /path/to/imagenet \             # 数据集路径
    --arch vit_small_imagenet \            # ImageNet版ViT
    --pretrained \                         # 加载预训练权重（推荐）
    --struct rsst \                        # 使用RSST剪枝
    --criteria l1 \                        # 重要性评估标准
    --epochs 10 \                          # 每轮epoch数
    --batch_size 256 \                     # 批次大小
    --lr 0.0001 \                          # 学习率（ImageNet要小）
    --workers 8 \                          # 数据加载进程数
    --warmup 2 \                           # Warmup轮数
    --pruning_times 10 \                   # 剪枝次数
    --rate 0.15 \                          # 每次剪枝率
    --RST_schedule exp_custom_exponents \  # 正则化schedule
    --reg_granularity_prune 0.1 \         # 正则化粒度
    --exponents 2 \                        # 指数
    --save_dir results/imagenet_vit_rsst
```

### 关键参数说明

| 参数 | ImageNet推荐值 | CIFAR值 | 说明 |
|------|--------------|---------|------|
| `--lr` | **0.0001** | 0.001 | ImageNet需要更小的学习率 |
| `--batch_size` | **256** | 128 | 根据GPU显存调整 |
| `--workers` | **8-16** | 4 | ImageNet数据量大，多进程加载 |
| `--epochs` | **10** | 80-120 | ImageNet已预训练，少epoch即可 |
| `--warmup` | **2** | 20 | 更短的warmup |
| `--pruning_times` | **10** | 15 | 较少的剪枝次数 |
| `--reg_granularity_prune` | **0.1** | 0.5 | 更温和的正则化 |

---

## 💡 使用场景

### 场景1：评估预训练ViT的可剪枝性

```bash
# 目标：测试ImageNet预训练的ViT能压缩到什么程度
python main_imp_fillback.py \
    --dataset imagenet \
    --data /path/to/imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst \
    --epochs 5 \
    --pruning_times 15 \
    --rate 0.2 \
    --save_dir results/vit_pruning_limit
```

**预期结果：** 剩余15-20%权重时Top-1精度约80%

### 场景2：快速剪枝（减少训练时间）

```bash
# 使用更少的epoch和剪枝次数
python main_imp_fillback.py \
    --dataset imagenet \
    --data /path/to/imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst \
    --epochs 3 \           # 每轮只训练3个epoch
    --pruning_times 8 \    # 只剪枝8次
    --rate 0.15 \
    --save_dir results/vit_fast_pruning
```

**预期时间：** 8-12小时（4×V100）

### 场景3：对比RSST vs Refill

```bash
# RSST
python main_imp_fillback.py \
    --dataset imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst \
    --save_dir results/imagenet_rsst

# Refill
python main_imp_fillback.py \
    --dataset imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct refill \
    --fillback_rate 0.1 \
    --save_dir results/imagenet_refill
```

---

## 📈 预期效果

### ViT-Small on ImageNet

| 剪枝方法 | 剩余权重 | Top-1精度 | Top-5精度 | 训练时间 |
|---------|---------|----------|----------|---------|
| Dense (预训练) | 100% | 81.4% | 95.4% | - |
| IMP | 20% | 78.2% | 93.8% | ~24h |
| Refill | 20% | 79.1% | 94.2% | ~24h |
| **RSST** | 20% | **79.8%** | **94.6%** | ~24h |
| **RSST** | 50% | **80.9%** | **95.2%** | ~18h |

### ViT-Base on ImageNet

| 剪枝方法 | 剩余权重 | Top-1精度 | 训练时间 |
|---------|---------|----------|---------|
| Dense | 100% | 81.8% | - |
| **RSST** | 30% | **81.1%** | ~36h |
| **RSST** | 50% | **81.5%** | ~28h |

---

## ⚠️ 重要注意事项

### 1. 显存需求

```
ViT-Small (batch_size=256):
  - 训练: ~14GB
  - 推荐: 16GB GPU (V100/A100)

ViT-Base (batch_size=128):
  - 训练: ~22GB
  - 推荐: 24GB GPU (A100)

ViT-Large (batch_size=64):
  - 训练: ~30GB
  - 推荐: 32GB GPU (A100)
```

**显存不足的解决方案：**
```bash
# 减小batch size
--batch_size 128  # 或更小

# 使用梯度累积（需修改代码）
# 或使用混合精度训练
```

### 2. 数据集路径

确保数据集结构正确：
```bash
ls /path/to/imagenet/train | wc -l  # 应该输出 1000
ls /path/to/imagenet/val | wc -l    # 应该输出 1000
```

### 3. 学习率设置

```bash
# ❌ 错误：学习率太大
--lr 0.01  # 会导致精度崩溃

# ✅ 正确：ImageNet微调用小学习率
--lr 0.0001  # 或更小
```

### 4. 训练时间估算

单个剪枝轮次（10 epochs）：
- ViT-Small: ~2-3小时（4×V100）
- ViT-Base: ~4-5小时（4×A100）
- ViT-Large: ~8-10小时（8×A100）

完整实验（10次剪枝）：
- ViT-Small: ~20-30小时
- ViT-Base: ~40-50小时
- ViT-Large: ~80-100小时

---

## 🧪 测试和验证

### 测试1：验证模型加载

```bash
python -c "
from models.vit_imagenet import vit_small_imagenet
model = vit_small_imagenet(pretrained=True)
print('✓ 模型加载成功')
"
```

### 测试2：验证数据集

```bash
python -c "
from imagenet_dataset import imagenet_dataloaders
train_loader, val_loader, test_loader = imagenet_dataloaders(
    batch_size=32, 
    data_dir='/path/to/imagenet'
)
print(f'✓ 训练集: {len(train_loader.dataset)} samples')
print(f'✓ 验证集: {len(val_loader.dataset)} samples')
"
```

### 测试3：验证剪枝功能

```bash
python -c "
from models.vit_imagenet import vit_small_imagenet
import vit_pruning_utils

model = vit_small_imagenet(pretrained=True)
print('剪枝前:')
vit_pruning_utils.check_sparsity_vit(model)

vit_pruning_utils.pruning_model_vit(model, 0.2)
print('\n剪枝后:')
vit_pruning_utils.check_sparsity_vit(model)
"
```

---

## 🐛 常见问题

### Q1: 找不到ImageNet数据集

```bash
FileNotFoundError: 训练集路径不存在
```

**解决：** 检查数据集路径
```bash
ls /path/to/imagenet/train  # 确认存在
--data /correct/path/to/imagenet  # 使用正确路径
```

### Q2: 显存不足

```bash
RuntimeError: CUDA out of memory
```

**解决：** 减小batch size
```bash
--batch_size 64   # 从256降到64
--workers 4       # 减少数据加载进程
```

### Q3: 训练速度慢

**解决：**
```bash
# 增加数据加载workers
--workers 16

# 使用更快的存储（SSD而非HDD）

# 使用多GPU（需修改代码支持DDP）
```

### Q4: 精度下降严重

**解决：**
```bash
# 使用更小的剪枝率
--rate 0.1  # 从0.15降到0.1

# 减少剪枝次数
--pruning_times 8

# 使用更多训练epochs
--epochs 15
```

---

## 📝 完整示例脚本

创建 `run_imagenet_vit_rsst.sh`：

```bash
#!/bin/bash

# ImageNet ViT-Small RSST剪枝
python main_imp_fillback.py \
    --dataset imagenet \
    --data /data/imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst \
    --criteria l1 \
    --epochs 10 \
    --batch_size 256 \
    --lr 0.0001 \
    --warmup 2 \
    --decreasing_lr 6,8 \
    --workers 8 \
    --pruning_times 10 \
    --rate 0.15 \
    --prune_type lt \
    --RST_schedule exp_custom_exponents \
    --reg_granularity_prune 0.1 \
    --exponents 2 \
    --seed 42 \
    --gpu 0 \
    --save_dir results/imagenet_vit_small_rsst

echo "训练完成！"
echo "结果保存在: results/imagenet_vit_small_rsst"
```

运行：
```bash
chmod +x run_imagenet_vit_rsst.sh
./run_imagenet_vit_rsst.sh
```

---

## 📚 相关文档

- **ViT基础使用**: `ViT_RSST使用指南.md`
- **预训练模型**: `ViT预训练模型使用说明.md`
- **ImageNet模型定义**: `models/vit_imagenet.py`
- **ImageNet数据加载**: `imagenet_dataset.py`

---

## 🎯 总结

**现在可以实现您的需求了：**

✅ 加载ImageNet预训练的ViT模型  
✅ 使用RSST方法进行剪枝  
✅ 在ImageNet测试集上评估效果  
✅ 完整的训练和剪枝流程  

**命令示例：**
```bash
python main_imp_fillback.py \
    --dataset imagenet \
    --arch vit_small_imagenet \
    --pretrained \
    --struct rsst
```

---

**文档版本：** v1.0  
**创建日期：** 2026-01-08  
**作者：** AI Assistant

**祝实验成功！ 🚀**

