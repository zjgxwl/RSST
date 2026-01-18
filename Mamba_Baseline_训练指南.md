# Mamba-Small Baseline 训练指南

**创建时间**: 2026-01-19  
**用途**: 测试 Mamba-Small 在 CIFAR-10/100 上的最佳性能（无剪枝）

---

## 📋 概述

这套脚本用于训练 Mamba-Small 的 **baseline 性能**，与 RSST/Refill 剪枝方法无关。

### 核心特点

基于 **Gemini 建议的现代化训练方案**：
- ✅ **优化器**: AdamW with Cosine LR Schedule
- ✅ **强数据增强**: RandAugment + Mixup + Cutmix
- ✅ **训练轮数**: 300 epochs（小数据集需要更多迭代）
- ✅ **Weight Decay**: 0.05（关键参数，Mamba 非常敏感）
- ✅ **Label Smoothing**: 0.1

### 预期性能

| 数据集 | 训练方式 | 预期准确率 |
|--------|---------|-----------|
| **CIFAR-10** | 从零训练 | **94.0-95.5%** |
| **CIFAR-100** | 从零训练 | **76.0-81.0%** |
| **CIFAR-10** | ImageNet 预训练微调 | 98.5-99.1% |
| **CIFAR-100** | ImageNet 预训练微调 | 88.5-91.0% |

---

## 🚀 快速开始

### 1. 运行完整训练（300 epochs）

```bash
cd /workspace/ycx/RSST

# 运行 CIFAR-10 + CIFAR-100（推荐）
./run_mamba_baseline.sh
```

**预计时间**: 2-3 天（双 GPU 并行）

---

### 2. 快速测试（30 epochs）

如果只想验证流程是否正常：

```bash
# 编辑脚本，修改配置
vim run_mamba_baseline.sh

# 将 RUN_QUICK_TEST 改为 true
RUN_QUICK_TEST=true

# 运行
./run_mamba_baseline.sh
```

**预计时间**: 2-3 小时

---

### 3. 单独运行某个数据集

```bash
# 编辑脚本
vim run_mamba_baseline.sh

# 选择想运行的实验
RUN_CIFAR10=true      # CIFAR-10
RUN_CIFAR100=false    # 不运行 CIFAR-100

# 运行
./run_mamba_baseline.sh
```

---

## 📊 监控训练

### 查看实时日志

```bash
# CIFAR-10
tail -f logs_mamba_baseline/mamba_small_cifar10_baseline_*.log

# CIFAR-100
tail -f logs_mamba_baseline/mamba_small_cifar100_baseline_*.log

# 同时查看所有日志
tail -f logs_mamba_baseline/*.log
```

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### 查看进程

```bash
ps aux | grep train_mamba_baseline
```

### 停止训练

```bash
# 找到 PID（启动时会显示）
# 或者用 ps 查找
kill <PID>
```

---

## 📁 文件结构

```
RSST/
├── train_mamba_baseline.py          # 主训练脚本
├── run_mamba_baseline.sh            # 启动脚本
├── Mamba_Baseline_训练指南.md       # 本文档
├── logs_mamba_baseline/             # 训练日志
│   ├── mamba_small_cifar10_baseline_*.log
│   └── mamba_small_cifar100_baseline_*.log
└── checkpoint/mamba_baseline/       # 模型保存
    ├── cifar10/
    │   ├── mamba_small_cifar10_best.pth
    │   └── mamba_small_cifar10_epoch*.pth
    └── cifar100/
        ├── mamba_small_cifar100_best.pth
        └── mamba_small_cifar100_epoch*.pth
```

---

## ⚙️ 参数详解

### 核心参数（训练脚本）

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--dataset` | cifar10 | 数据集（cifar10/cifar100）|
| `--arch` | mamba_small | 模型（mamba_tiny/small/base）|
| `--epochs` | 300 | 训练轮数 |
| `--batch_size` | 128 | Batch size |
| `--lr` | 1e-3 | 初始学习率 |
| `--weight_decay` | 0.05 | 权重衰减（关键！）|
| `--warmup_epochs` | 20 | Warmup 轮数 |

### 数据增强参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use_randaugment` | True | RandAugment (2, 9) |
| `--use_mixup` | True | Mixup |
| `--use_cutmix` | True | Cutmix |
| `--mixup_alpha` | 0.8 | Mixup alpha |
| `--cutmix_alpha` | 1.0 | Cutmix alpha |
| `--label_smoothing` | 0.1 | Label smoothing |

---

## 🔧 自定义训练

### 使用命令行直接运行

```bash
cd /workspace/ycx/RSST

python train_mamba_baseline.py \
    --dataset cifar10 \
    --arch mamba_small \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --use_randaugment \
    --use_mixup \
    --use_cutmix \
    --mixup_alpha 0.8 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.1 \
    --gpu 0 \
    --save_dir ./checkpoint/mamba_baseline/cifar10
```

### 调整学习率

```bash
# 尝试更小的学习率（如果训练不稳定）
--lr 5e-4

# 或更大的学习率（如果收敛太慢）
--lr 2e-3
```

### 调整 Weight Decay

```bash
# Gemini 建议：0.05
# 如果过拟合严重，可以增大
--weight_decay 0.1

# 如果欠拟合，可以减小
--weight_decay 0.03
```

### 使用更小的模型（快速实验）

```bash
--arch mamba_tiny  # 参数量 ~5M
```

---

## 📈 预期训练曲线

### CIFAR-10

```
Epoch   50:  ~85%
Epoch  100:  ~90%
Epoch  150:  ~92%
Epoch  200:  ~93.5%
Epoch  250:  ~94.5%
Epoch  300:  ~95%
```

### CIFAR-100

```
Epoch   50:  ~55%
Epoch  100:  ~65%
Epoch  150:  ~72%
Epoch  200:  ~76%
Epoch  250:  ~78%
Epoch  300:  ~79-80%
```

---

## 💡 关键优化建议

### 1. Weight Decay = 0.05 是关键

Mamba 模型对 Weight Decay 非常敏感：
- 太小（如 1e-4）：严重过拟合
- 太大（如 0.2）：欠拟合
- **推荐 0.05**

### 2. 需要训练足够长

CIFAR 数据集小，但 Mamba 模型大（16.5M 参数）：
- 至少 **300 epochs**
- SSM 需要更多迭代来学习空间结构

### 3. 强数据增强必不可少

不使用数据增强，精度会下降 **5-10%**：
- RandAugment
- Mixup
- Cutmix
- Label Smoothing

### 4. Cosine LR + Warmup

- Warmup 帮助 SSM 层稳定初始化
- Cosine Decay 比 Step Decay 效果更好

---

## 🔍 常见问题

### Q1: 训练太慢，如何加速？

**方案 1**: 使用更小的模型
```bash
--arch mamba_tiny  # 参数量 5M vs 16.5M
```

**方案 2**: 减少训练轮数（快速测试）
```bash
--epochs 100  # 可能只能达到 90% (CIFAR-10)
```

**方案 3**: 增大 batch size（如果 GPU 内存充足）
```bash
--batch_size 256  # 需要 ~16GB GPU 内存
```

---

### Q2: 精度达不到预期怎么办？

**检查清单**:
1. ✅ Weight Decay 是否设置为 0.05？
2. ✅ 是否启用了所有数据增强？
3. ✅ 是否训练了足够的轮数（300 epochs）？
4. ✅ 学习率是否合适（1e-3 或 5e-4）？

**尝试调整**:
```bash
# 增大 Weight Decay
--weight_decay 0.08

# 增加训练轮数
--epochs 400

# 调整学习率
--lr 5e-4
```

---

### Q3: GPU 内存不足怎么办？

```bash
# 方案 1: 减小 batch size
--batch_size 64

# 方案 2: 使用更小的模型
--arch mamba_tiny

# 方案 3: 梯度累积（需要修改代码）
# 暂不支持，可以后续添加
```

---

### Q4: 如何恢复中断的训练？

目前脚本会定期保存 checkpoint（每 50 epochs），但恢复功能尚未实现。

**临时方案**: 重新训练（因为脚本会保存最佳模型）

---

## 📊 与剪枝方法对比

| 方法 | 参数量 | CIFAR-10 | CIFAR-100 | 训练时间 |
|------|--------|----------|-----------|---------|
| **Baseline（本脚本）** | 16.5M | ~95% | ~80% | 2-3天 |
| **RSST 70%剪枝** | 6.6M | ~90-91% | ~70% | 4-5天 |
| **Refill 70%剪枝** | 6.6M | ~89-90% | ~68-69% | 4-5天 |

**结论**: Baseline 提供了性能上限，剪枝方法在 **3.3× 参数压缩**下保持了 **90-95%** 的性能。

---

## 📚 参考资料

### 相关论文

1. **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. **Vim (Vision Mamba)**: 将 Mamba 应用到视觉任务
3. **DeiT**: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) - 现代训练策略

### 项目中的其他文档

- `Mamba_RSST使用指南.md`: Mamba 剪枝方法
- `Mamba测试报告.md`: 剪枝功能测试
- `Mamba可剪枝组件详细分析.md`: 技术分析

---

## ✅ 检查清单

使用前确认：

- [ ] 已激活 conda 环境：`conda activate structlth`
- [ ] 数据集已准备：`datasets/cifar10`, `datasets/cifar100`
- [ ] GPU 可用：`nvidia-smi`
- [ ] 脚本有执行权限：`chmod +x run_mamba_baseline.sh`
- [ ] 磁盘空间充足（checkpoint ~500MB，日志 ~100MB）

---

## 📞 问题反馈

如遇到问题，请检查：

1. **日志文件**: `logs_mamba_baseline/*.log`
2. **GPU 状态**: `nvidia-smi`
3. **进程状态**: `ps aux | grep train_mamba_baseline`
4. **磁盘空间**: `df -h`

---

**祝实验顺利！** 🎉

**预期结果**: 
- CIFAR-10: **94-95.5%** 
- CIFAR-100: **76-81%**
