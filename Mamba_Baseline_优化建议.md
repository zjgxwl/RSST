# Mamba-Small Baseline 优化建议

**分析时间**: 2026-01-19  
**当前性能**: CIFAR-10 94-95.5%, CIFAR-100 76-81%  
**优化目标**: CIFAR-10 96%+, CIFAR-100 82-85%+

---

## 📊 优化空间分析总览

| 优化类别 | 当前状态 | 潜在提升 | 难度 | 优先级 |
|---------|---------|---------|------|--------|
| **1. 数据增强** | 基础 | +1-2% | 低 | ⭐⭐⭐ |
| **2. 模型正则化** | 部分 | +0.5-1% | 低 | ⭐⭐⭐ |
| **3. 训练技巧** | 基础 | +1-1.5% | 中 | ⭐⭐ |
| **4. 模型架构** | 标准 | +0.5-1% | 高 | ⭐⭐ |
| **5. 推理优化** | 无 | +0.5-1% | 低 | ⭐⭐ |
| **6. 工程优化** | 基础 | 2-3× 速度 | 中 | ⭐ |
| **7. Mamba特定** | 标准 | +1-2% | 高 | ⭐⭐⭐ |

---

## 🎯 优先级 1: 高性价比优化（立即可做）

### 1.1 ⭐⭐⭐ 添加 Drop Path (Stochastic Depth)

**问题**: 当前代码有 `--drop_path` 参数，但**模型中没有实际使用**！

**修改位置**: `models/mamba.py` 的 `MambaBlock`

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, 
                 use_mlp=True, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):  # 添加 drop_path
        super().__init__()
        self.d_model = d_model
        self.use_mlp = use_mlp
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()  # 新增
        
        # ... 其他代码不变 ...
    
    def forward(self, x):
        # SSM路径 (with residual + drop_path)
        x = x + self.drop_path(self.ssm(self.norm1(x)))  # 修改
        
        # MLP路径 (with residual + drop_path)
        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # 修改
        
        return x

# 添加 DropPath 类
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
```

**预期提升**: +0.5-1%  
**推荐值**: 0.1-0.2

---

### 1.2 ⭐⭐⭐ 添加更多数据增强

**当前**: 只有 RandAugment + Mixup + Cutmix

**推荐添加**:

#### A. Random Erasing (擦除增强)

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25),  # 新增！25% 概率擦除
])
```

**预期提升**: +0.3-0.5%

#### B. AutoAugment (比 RandAugment 更强)

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # 替换 RandAugment
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25),
])
```

**预期提升**: +0.5-1%（替代 RandAugment）

---

### 1.3 ⭐⭐⭐ EMA (Exponential Moving Average)

**原理**: 使用参数的指数移动平均，提升测试性能

```python
class ModelEMA:
    """模型参数的指数移动平均"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 EMA 参数（测试时）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数（训练时）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# 使用方法
ema = ModelEMA(model, decay=0.9999)

# 训练循环中
for batch in train_loader:
    # ... 前向传播和反向传播 ...
    optimizer.step()
    ema.update()  # 更新 EMA

# 验证时
ema.apply_shadow()  # 使用 EMA 参数
val_acc = validate(model, test_loader, criterion, args)
ema.restore()  # 恢复训练参数
```

**预期提升**: +0.3-0.7%  
**推荐值**: decay=0.9999

---

### 1.4 ⭐⭐ Gradient Clipping

**问题**: 训练可能不稳定，特别是 SSM 层

```python
# 在 optimizer.step() 之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**预期提升**: +0.2-0.5%（主要提升稳定性）  
**推荐值**: max_norm=1.0

---

### 1.5 ⭐⭐ Test-Time Augmentation (TTA)

**原理**: 测试时使用多个增强版本投票

```python
def validate_with_tta(model, test_loader, criterion, args, n_augment=5):
    """带 TTA 的验证"""
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    # TTA transforms
    tta_transforms = [
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=p),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
        for p in [0, 0.5, 1.0]  # 3种翻转策略
    ]
    
    with torch.no_grad():
        for images, targets in test_loader:
            batch_outputs = []
            
            # 对每个样本应用多个增强
            for transform in tta_transforms:
                aug_images = torch.stack([transform(img) for img in images])
                aug_images = aug_images.cuda()
                outputs = model(aug_images)
                batch_outputs.append(outputs)
            
            # 平均所有增强的输出
            avg_output = torch.stack(batch_outputs).mean(dim=0)
            all_outputs.append(avg_output)
            all_targets.append(targets)
    
    # 计算准确率
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets).cuda()
    acc = accuracy(all_outputs, all_targets, topk=(1,))[0]
    
    return acc.item()
```

**预期提升**: +0.5-1%  
**成本**: 推理时间增加 3-5×

---

## 🚀 优先级 2: 进阶优化（中等难度）

### 2.1 ⭐⭐⭐ 混合精度训练 (AMP)

**好处**: 
- 训练速度提升 **2-3×**
- 显存节省 **30-40%**
- 精度基本不变（甚至略有提升）

```python
from torch.cuda.amp import autocast, GradScaler

# 初始化
scaler = GradScaler()

# 训练循环
for images, targets in train_loader:
    optimizer.zero_grad()
    
    # 使用混合精度
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, targets)
    
    # 缩放梯度
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

**预期提升**: 
- 速度: +100-200%
- 精度: +0.0-0.2%

---

### 2.2 ⭐⭐ Layer-wise Learning Rate Decay

**原理**: 不同层使用不同学习率（浅层小，深层大）

```python
def get_layer_wise_lr_params(model, lr, decay_rate=0.65):
    """
    为不同层设置不同的学习率
    深层学习率大，浅层学习率小
    """
    parameter_groups = []
    
    # Patch embedding (最小学习率)
    parameter_groups.append({
        'params': model.patch_embed.parameters(),
        'lr': lr * (decay_rate ** 24)
    })
    
    # Blocks (逐层递增)
    for i, block in enumerate(model.blocks):
        parameter_groups.append({
            'params': block.parameters(),
            'lr': lr * (decay_rate ** (24 - i))
        })
    
    # Head (最大学习率)
    parameter_groups.append({
        'params': model.head.parameters(),
        'lr': lr
    })
    
    return parameter_groups

# 使用
param_groups = get_layer_wise_lr_params(model, lr=1e-3, decay_rate=0.65)
optimizer = optim.AdamW(param_groups, weight_decay=0.05)
```

**预期提升**: +0.3-0.5%  
**推荐**: decay_rate=0.65-0.75

---

### 2.3 ⭐⭐ 更好的学习率 Warmup

**当前**: 线性 warmup  
**推荐**: Exponential warmup（更平滑）

```python
def get_cosine_schedule_with_exp_warmup(optimizer, num_warmup_steps, 
                                         num_training_steps, min_lr=0):
    """
    Exponential warmup + Cosine decay
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Exponential warmup: 更平滑
            progress = current_step / num_warmup_steps
            return (1 - np.exp(-5 * progress)) / (1 - np.exp(-5))
        
        # Cosine decay
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**预期提升**: +0.1-0.3%（主要提升稳定性）

---

### 2.4 ⭐⭐ Knowledge Distillation (自蒸馏)

**原理**: 使用训练过程中的最佳 checkpoint 作为教师

```python
# 步骤 1: 先训练一个教师模型（300 epochs）
# 步骤 2: 使用教师模型蒸馏学生模型

def distillation_loss(student_outputs, teacher_outputs, targets, 
                      temperature=4.0, alpha=0.5):
    """
    蒸馏损失 = α * KL散度 + (1-α) * CE损失
    """
    # KL 散度
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 交叉熵
    hard_loss = F.cross_entropy(student_outputs, targets)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

# 训练循环
teacher_model.eval()  # 教师模型冻结
student_model.train()

with torch.no_grad():
    teacher_outputs = teacher_model(images)

student_outputs = student_model(images)
loss = distillation_loss(student_outputs, teacher_outputs, targets)
```

**预期提升**: +0.5-1%  
**推荐**: temperature=4.0, alpha=0.5

---

## 🔬 优先级 3: Mamba 特定优化（高难度）

### 3.1 ⭐⭐⭐ Bidirectional SSM (双向扫描)

**问题**: 当前 SSM 是单向的，图像信息可能丢失

**方案**: 实现双向 SSM

```python
class BidirectionalSSM(nn.Module):
    """双向 SSM"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)  # 融合前向和后向
    
    def forward(self, x):
        # 前向扫描
        forward_out = self.forward_ssm(x)
        
        # 后向扫描（翻转序列）
        x_reversed = torch.flip(x, dims=[1])  # 翻转序列维度
        backward_out = self.backward_ssm(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])  # 翻转回来
        
        # 融合
        merged = self.merge(torch.cat([forward_out, backward_out], dim=-1))
        return merged
```

**预期提升**: +1-2%  
**成本**: 参数量 +100%, 速度 -20%

---

### 3.2 ⭐⭐ 更好的 Patch Embedding

**当前**: 简单的 4×4 卷积  
**推荐**: 重叠的卷积 + 更深的 stem

```python
class AdvancedPatchEmbed(nn.Module):
    """
    更强的 Patch Embedding
    使用重叠卷积 + 更深的 stem
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.stem = nn.Sequential(
            # Stage 1: 3→64, 3×3 conv, stride=1
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Stage 2: 64→128, 3×3 conv, stride=2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Stage 3: 128→embed_dim, 3×3 conv, stride=2
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        
        self.num_patches = (img_size // 4) ** 2  # 32/4 = 8, 8*8=64
    
    def forward(self, x):
        x = self.stem(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x
```

**预期提升**: +0.3-0.7%  
**成本**: 参数量 +5%, 速度基本不变

---

### 3.3 ⭐⭐ Multi-scale SSM

**原理**: 在不同尺度上应用 SSM（类似 FPN）

```python
class MultiScaleSSM(nn.Module):
    """多尺度 SSM"""
    def __init__(self, d_model, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.ssms = nn.ModuleList([
            SelectiveSSM(d_model) for _ in scales
        ])
        self.merge = nn.Linear(d_model * len(scales), d_model)
    
    def forward(self, x):
        # x: [B, L, D]
        outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                out = self.ssms[i](x)
            else:
                # 下采样 → SSM → 上采样
                x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale).transpose(1, 2)
                out_down = self.ssms[i](x_down)
                out = F.interpolate(out_down.transpose(1, 2), size=x.size(1)).transpose(1, 2)
            
            outputs.append(out)
        
        # 融合多尺度特征
        merged = self.merge(torch.cat(outputs, dim=-1))
        return merged
```

**预期提升**: +0.5-1%  
**成本**: 参数量 +200%, 速度 -30%

---

## 💻 优先级 4: 工程优化（提速不提精度）

### 4.1 ⭐⭐ Gradient Accumulation（显存不足时）

```python
ACCUMULATION_STEPS = 4  # 累积4个batch

for i, (images, targets) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss = loss / ACCUMULATION_STEPS  # 缩放损失
    
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**效果**: 等效 batch_size × 4，显存需求不变

---

### 4.2 ⭐⭐ 编译优化 (torch.compile)

```python
# PyTorch 2.0+ 支持
model = torch.compile(model, mode='max-autotune')
```

**效果**: 速度提升 10-30%（取决于硬件）

---

### 4.3 ⭐ DataLoader 优化

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,  # 增加 workers
    pin_memory=True,
    persistent_workers=True,  # 新增：保持 workers 常驻
    prefetch_factor=2,  # 新增：预取2个batch
)
```

**效果**: 数据加载加速 20-40%

---

## 📋 优化实施路线图

### 阶段 1: 快速见效（1-2天）

```
1. ✅ 添加 Drop Path            → +0.5-1%
2. ✅ 添加 Random Erasing       → +0.3-0.5%
3. ✅ 添加 Gradient Clipping    → 稳定性提升
4. ✅ 添加 EMA                  → +0.3-0.7%

预期总提升: +1.1-2.2%
```

### 阶段 2: 中期优化（3-5天）

```
5. ✅ 混合精度训练 (AMP)        → 速度 +100-200%
6. ✅ AutoAugment 替换 RandAugment → +0.5-1%
7. ✅ Layer-wise LR Decay       → +0.3-0.5%

预期总提升: +0.8-1.5% + 2× 速度
```

### 阶段 3: 高级优化（1-2周）

```
8. ✅ Bidirectional SSM         → +1-2%
9. ✅ 改进 Patch Embedding      → +0.3-0.7%
10. ✅ TTA (测试时增强)          → +0.5-1%
11. ✅ Knowledge Distillation   → +0.5-1%

预期总提升: +2.3-4.7%
```

---

## 🎯 最终预期性能

| 数据集 | 当前 | 阶段1 | 阶段2 | 阶段3 | 目标 |
|--------|-----|-------|-------|-------|------|
| **CIFAR-10** | 94-95.5% | 95.5-97% | 96-97.5% | **97-98%** | 98%+ |
| **CIFAR-100** | 76-81% | 77-83% | 78-84% | **82-86%** | 85%+ |

---

## 📦 一键优化脚本（即将提供）

```bash
# 阶段1优化（推荐先做）
./run_mamba_baseline_v2.sh --stage 1

# 阶段2优化
./run_mamba_baseline_v2.sh --stage 2

# 阶段3优化（需要更多时间）
./run_mamba_baseline_v2.sh --stage 3
```

---

## 📚 参考论文

1. **DropPath**: Deep Networks with Stochastic Depth
2. **EMA**: Mean teachers are better role models
3. **Layer-wise LR**: ELECTRA: Pre-training Text Encoders
4. **Bidirectional SSM**: Vim: Vision Mamba
5. **TTA**: Test-Time Augmentation with Transformers

---

**最后更新**: 2026-01-19  
**下一步**: 实现阶段1优化
