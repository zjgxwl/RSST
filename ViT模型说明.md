# ViT模型完整说明

本项目支持**6种Vision Transformer (ViT)模型**，分为CIFAR专用和ImageNet专用两类。

---

## 📚 模型列表

### 🎯 CIFAR专用模型 (32x32输入)

这些模型专门为CIFAR-10/100数据集设计，使用较小的patch size (4x4)。

#### 1. `vit_tiny` ⚡

**最小最快的ViT模型，适合快速实验**

| 配置 | 值 |
|------|-----|
| **Embed Dim** | 192 |
| **Depth** | 9 layers |
| **Num Heads** | 3 |
| **MLP Ratio** | 2x |
| **参数量** | ~5M |
| **预训练** | ❌ 不支持 |
| **训练时间** | ~4-6小时 (80 epochs) |

**使用示例**：
```bash
./launch_experiment.sh cifar10 vit_tiny 80 1.0 4 true 0.2 y rsst
```

**特点**：
- ✅ 训练速度最快
- ✅ 显存占用最小 (8GB+)
- ✅ 适合快速调试和算法验证
- ❌ 无预训练权重

---

#### 2. `vit_small` 🔥

**推荐的CIFAR模型，性能与速度平衡**

| 配置 | 值 |
|------|-----|
| **Embed Dim** | 384 |
| **Depth** | 12 layers |
| **Num Heads** | 6 |
| **MLP Ratio** | 4x |
| **参数量** | ~22M |
| **预训练** | ✅ 支持ImageNet预训练 |
| **训练时间** | ~6-8小时 (80 epochs) |

**使用示例**：
```bash
# 使用预训练权重
./launch_experiment.sh cifar10 vit_small 80 1.0 4 true 0.2 y rsst

# 不使用预训练
./launch_experiment.sh cifar10 vit_small 80 1.0 4 false 0.2 y rsst
```

**特点**：
- ✅ 性能与速度平衡
- ✅ 支持ImageNet预训练
- ✅ 社区验证充分
- ✅ 推荐用于CIFAR实验

---

#### 3. `vit_base`

**最强的CIFAR模型，追求极致性能**

| 配置 | 值 |
|------|-----|
| **Embed Dim** | 768 |
| **Depth** | 12 layers |
| **Num Heads** | 12 |
| **MLP Ratio** | 4x |
| **参数量** | ~86M |
| **预训练** | ✅ 支持ImageNet预训练 |
| **训练时间** | ~10-15小时 (80 epochs) |

**使用示例**：
```bash
./launch_experiment.sh cifar100 vit_base 80 1.0 4 true 0.2 y rsst
```

**特点**：
- ✅ 最强性能
- ✅ 支持ImageNet预训练
- ⚠️  需要更多显存 (16GB+)
- ⚠️  训练时间较长

---

### 🌐 ImageNet专用模型 (224x224输入)

这些模型专门为ImageNet数据集设计，使用标准的patch size (16x16)。

#### 4. `vit_small_imagenet`

**推荐的ImageNet实验模型**

| 配置 | 值 |
|------|-----|
| **输入尺寸** | 224x224 |
| **Patch Size** | 16x16 |
| **参数量** | ~22M |
| **预训练** | ✅ 强烈推荐 |

**使用示例**：
```bash
./launch_experiment.sh imagenet vit_small_imagenet 100 1.0 4 true 0.2 y rsst
```

---

#### 5. `vit_base_imagenet`

**标准的ImageNet ViT配置**

| 配置 | 值 |
|------|-----|
| **输入尺寸** | 224x224 |
| **Patch Size** | 16x16 |
| **参数量** | ~86M |
| **预训练** | ✅ 强烈推荐 |

**使用示例**：
```bash
./launch_experiment.sh imagenet vit_base_imagenet 100 1.0 4 true 0.2 y rsst
```

---

#### 6. `vit_large_imagenet`

**最大的ViT模型，追求极致性能**

| 配置 | 值 |
|------|-----|
| **输入尺寸** | 224x224 |
| **Patch Size** | 16x16 |
| **参数量** | ~307M |
| **预训练** | ✅ 必须使用 |

**使用示例**：
```bash
./launch_experiment.sh imagenet vit_large_imagenet 100 1.0 4 true 0.2 y rsst
```

**特点**：
- ✅ 最强性能
- ⚠️  需要大量显存 (24GB+)
- ⚠️  训练时间非常长
- ⚠️  必须使用预训练权重

---

## 📊 模型对比表

| 模型 | 数据集 | 参数量 | 层数 | 输入尺寸 | 预训练 | 显存需求 | 速度 |
|------|--------|--------|------|----------|--------|----------|------|
| **vit_tiny** | CIFAR | ~5M | 9 | 32x32 | ❌ | 8GB+ | ⚡⚡⚡ |
| **vit_small** | CIFAR | ~22M | 12 | 32x32 | ✅ | 8GB+ | ⚡⚡ |
| **vit_base** | CIFAR | ~86M | 12 | 32x32 | ✅ | 16GB+ | ⚡ |
| **vit_small_imagenet** | ImageNet | ~22M | 12 | 224x224 | ✅ | 16GB+ | ⚡⚡ |
| **vit_base_imagenet** | ImageNet | ~86M | 12 | 224x224 | ✅ | 24GB+ | ⚡ |
| **vit_large_imagenet** | ImageNet | ~307M | 24 | 224x224 | ✅ | 32GB+ | 🐢 |

---

## 💡 选择建议

### 🎯 根据目标选择

#### 快速实验和调试
**推荐：`vit_tiny`**
- 训练速度最快
- 适合快速验证算法
- 资源需求最低

#### CIFAR数据集最佳实践
**推荐：`vit_small`**
- 性能与速度平衡
- 支持预训练
- 社区验证充分

#### ImageNet数据集
**推荐：`vit_small_imagenet` 或 `vit_base_imagenet`**
- 标准配置
- 预训练效果好
- 适合研究和实验

#### 追求最高精度
**推荐：`vit_base` (CIFAR) 或 `vit_large_imagenet` (ImageNet)**
- 最强性能
- 需要更多资源和时间

---

## ⚙️ 使用方法

### 基本命令格式

```bash
./launch_experiment.sh <dataset> <model> <epochs> <reg> <exp> <pretrained> <rate> <auto> <algorithm> <fillback>
```

### 参数说明

- `<dataset>`: 数据集名称 (`cifar10`, `cifar100`, `imagenet`)
- `<model>`: 模型名称（见下方列表）
- `<epochs>`: 训练轮数（推荐80）
- `<reg>`: 正则化粒度（推荐1.0）
- `<exp>`: 指数曲率（推荐4）
- `<pretrained>`: 是否使用预训练 (`true`/`false`)
- `<rate>`: 剪枝率（推荐0.2）
- `<auto>`: 自动确认 (`y`自动，`n`手动)
- `<algorithm>`: 算法类型 (`rsst`/`refill`)
- `<fillback>`: refill回填率（仅refill使用，推荐0.2）

### 模型参数值

```
# CIFAR专用
vit_tiny
vit_small
vit_base

# ImageNet专用
vit_small_imagenet
vit_base_imagenet
vit_large_imagenet
```

---

## 📝 完整示例

### CIFAR-10 实验

```bash
# 1. 快速实验 (vit_tiny, 无预训练)
./launch_experiment.sh cifar10 vit_tiny 80 1.0 4 false 0.2 y rsst

# 2. 标准实验 (vit_small, 预训练)
./launch_experiment.sh cifar10 vit_small 80 1.0 4 true 0.2 y rsst

# 3. 高性能实验 (vit_base, 预训练)
./launch_experiment.sh cifar10 vit_base 80 1.0 4 true 0.2 y rsst
```

### CIFAR-100 实验

```bash
# 使用Refill算法
./launch_experiment.sh cifar100 vit_small 80 1.0 4 true 0.2 y refill 0.2

# 使用RSST算法
./launch_experiment.sh cifar100 vit_small 80 1.0 4 true 0.2 y rsst
```

### ImageNet 实验

```bash
# ViT-Small (推荐)
./launch_experiment.sh imagenet vit_small_imagenet 100 1.0 4 true 0.2 y rsst

# ViT-Base
./launch_experiment.sh imagenet vit_base_imagenet 100 1.0 4 true 0.2 y rsst
```

---

## ⚠️ 注意事项

### 预训练权重

1. **vit_tiny**: 不支持预训练，始终使用随机初始化
2. **其他模型**: 需要安装 `timm` 库才能使用预训练
   ```bash
   pip install timm
   ```

### 显存需求

| 模型 | 最小显存 | 推荐显存 |
|------|----------|----------|
| vit_tiny | 6GB | 8GB |
| vit_small | 8GB | 12GB |
| vit_base | 12GB | 16GB |
| vit_small_imagenet | 12GB | 16GB |
| vit_base_imagenet | 16GB | 24GB |
| vit_large_imagenet | 24GB | 32GB |

### 训练时间估算

基于单张A800 80GB GPU，80 epochs：

- **vit_tiny**: ~4-6小时 (CIFAR)
- **vit_small**: ~6-8小时 (CIFAR)
- **vit_base**: ~10-15小时 (CIFAR)
- **vit_small_imagenet**: ~20-30小时 (ImageNet)
- **vit_base_imagenet**: ~40-60小时 (ImageNet)
- **vit_large_imagenet**: ~80-120小时 (ImageNet)

---

## 🔍 模型架构细节

### CIFAR模型 vs ImageNet模型

| 特性 | CIFAR模型 | ImageNet模型 |
|------|-----------|--------------|
| **输入尺寸** | 32x32 | 224x224 |
| **Patch Size** | 4x4 | 16x16 |
| **Num Patches** | 64 | 196 |
| **位置编码** | 64 | 196 |
| **优化目标** | 小图像分类 | 大图像分类 |

### 层结构

所有ViT模型都包含：
- **Patch Embedding**: 将图像分割成patches
- **Transformer Blocks**: 多层自注意力和MLP
- **Classification Head**: 最终分类层

可剪枝的层：
- ✅ `attn.qkv` - Attention的Q/K/V投影
- ✅ `attn.proj` - Attention的输出投影
- ✅ `mlp.fc1` - MLP第一层
- ✅ `mlp.fc2` - MLP第二层
- ⚠️  `patch_embed` - 通常不剪枝

---

## 📚 相关文档

- **ViT_RSST使用指南.md** - ViT与RSST集成的详细说明
- **ViT预训练模型使用说明.md** - 预训练权重使用指南
- **实验启动指南.md** - 实验管理完整指南
- **launch_logs/使用指南.md** - 日志管理指南

---

## 🎓 参考文献

- **ViT原始论文**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **timm库**: https://github.com/huggingface/pytorch-image-models
