# Mamba模型结构化剪枝方案（RSST + Refill）

**创建时间**: 2026-01-17  
**方法**: 完全结构化剪枝（通道级、神经元级）  
**不做非结构化剪枝！**

---

## 🎯 核心理念

### RSST vs Refill（都是结构化剪枝）

```
传统方法（不适用）：
└─ 非结构化剪枝 → 稀疏矩阵 → 无法加速 → ❌

本项目方法（采用）：
├─ Refill: 结构化剪枝 + 重填充 → 通道/神经元删除 → 真实加速 → ✅
└─ RSST:  结构化剪枝 + 正则化 → 通道/神经元删除 → 真实加速 → ✅
```

### 结构化剪枝 = 删除整个维度

- ✅ **通道级剪枝**: 删除整个输出通道（所有参数）
- ✅ **神经元级剪枝**: 删除整个神经元（输入+输出权重）
- ❌ **元素级剪枝**: 不做！（无法加速）

---

## 🏗️ Mamba结构化剪枝映射

### Mamba Block结构（可剪枝组件）

```python
MambaBlock:
    ├─ norm1 (LayerNorm)                    # 不剪枝
    ├─ ssm (SelectiveSSM):
    │   ├─ in_proj [d_model → 2*d_inner]   # ✅ 通道级剪枝（输出通道）
    │   ├─ conv1d [d_inner, d_conv, 1]     # ✅ 通道级剪枝（协同）
    │   ├─ x_proj [d_inner → ...]          # ✅ 通道级剪枝（输入通道）
    │   └─ out_proj [d_inner → d_model]    # ✅ 通道级剪枝（输入通道）★ 高优先级
    ├─ norm2 (LayerNorm)                    # 不剪枝
    └─ mlp (MLP):
        ├─ fc1 [d_model → mlp_dim]         # ✅ 神经元级剪枝 ★ 高优先级
        └─ fc2 [mlp_dim → d_model]         # ✅ 神经元级剪枝（协同）
```

### 剪枝粒度定义

| 组件 | 剪枝粒度 | 删除什么 | 实际效果 |
|------|---------|---------|---------|
| **SSM输出投影** | 输入通道 | 删除`out_proj`的输入维度 | 减少d_inner |
| **SSM输入投影** | 输出通道 | 删除`in_proj`的输出维度 | 减少d_inner（协同） |
| **MLP FC1** | 输出神经元 | 删除`fc1`的输出维度 + `fc2`的输入维度 | 减少mlp_dim |

---

## 📐 结构化剪枝的数学定义

### 1. 通道重要性评分

对于Linear层 `W: [out_features, in_features]`：

#### 输出通道重要性（剪枝整个输出通道）
```python
# 方法1: L1范数
importance_out[i] = ||W[i, :]||_1  # 第i个输出通道的L1范数

# 方法2: L2范数
importance_out[i] = ||W[i, :]||_2

# 方法3: Taylor展开（考虑梯度）
importance_out[i] = ||W[i, :] ⊙ ∇W[i, :]||_1
```

#### 输入通道重要性（剪枝整个输入通道）
```python
# 需要看下一层的权重
importance_in[j] = ||W[:, j]||_1  # 第j个输入通道的L1范数
```

### 2. MLP神经元重要性评分

对于MLP: `fc1[d_model, mlp_dim]` + `fc2[mlp_dim, d_model]`：

```python
# 神经元重要性 = fc1输出权重 + fc2输入权重
importance_neuron[i] = ||fc1[i, :]||_1 + ||fc2[:, i]||_1
```

---

## 🎯 Mamba的3种结构化剪枝策略

### 策略1: 仅剪SSM输出投影（最保守）✅ 推荐起点

**目标层**: `blocks.*.ssm.out_proj`

**剪枝方式**: 输入通道级

```python
# 伪代码
for each block:
    # 1. 计算通道重要性
    importance = calculate_channel_importance(block.ssm.out_proj)
    
    # 2. 选择保留的通道
    keep_ratio = 0.3  # 保留30%
    keep_channels = topk(importance, k=int(d_inner * keep_ratio))
    
    # 3. 剪枝（删除不重要的输入通道）
    out_proj_pruned = prune_input_channels(block.ssm.out_proj, keep_channels)
    
    # 4. 协同调整上游层
    # - in_proj的对应输出通道
    # - conv1d的对应通道
    # - x_proj的对应输入通道
```

**预期效果**:
- d_inner: 384 → 115 (70%剪枝)
- 参数减少: ~40%
- 精度下降: <2%

---

### 策略2: MLP神经元剪枝（与ViT相同）✅ 推荐

**目标层**: `blocks.*.mlp.fc1` + `blocks.*.mlp.fc2`

**剪枝方式**: 神经元级（完全复用ViT代码）

```python
# 伪代码（与ViT的MLP剪枝完全相同）
for each block:
    # 1. 计算神经元重要性
    importance = calculate_neuron_importance(
        block.mlp.fc1, 
        block.mlp.fc2
    )
    
    # 2. 选择保留的神经元
    keep_ratio = 0.3  # 保留30%
    keep_neurons = topk(importance, k=int(mlp_dim * keep_ratio))
    
    # 3. 协同剪枝
    fc1_pruned = prune_output_neurons(block.mlp.fc1, keep_neurons)
    fc2_pruned = prune_input_neurons(block.mlp.fc2, keep_neurons)
```

**预期效果**:
- mlp_dim: 768 → 230 (70%剪枝)
- 参数减少: ~50%（MLP占主要参数）
- 精度下降: <1%

---

### 策略3: 混合剪枝（SSM + MLP）✅ 最佳效果

**同时剪枝SSM和MLP**

```python
# 剪枝配置
pruning_config = {
    'ssm_out_proj': {
        'target': 'blocks.*.ssm.out_proj',
        'method': 'channel',
        'ratio': 0.7,  # 70%稀疏度
    },
    'mlp': {
        'target': 'blocks.*.mlp',
        'method': 'neuron',
        'ratio': 0.7,
    }
}
```

**预期效果**:
- 总参数减少: ~60%
- FLOPs减少: ~55%
- 推理加速: 1.8-2.2x
- 精度下降: 2-4%

---

## 🔧 RSST vs Refill的区别

### Refill方法（重填充）

**核心思想**: 剪枝后，在重要通道上"重填充"一些被剪掉的参数

```python
# Refill流程
1. 结构化剪枝（删除不重要的通道/神经元）
2. 评估剩余通道的重要性
3. 在最重要的通道上，恢复一些之前删除的参数位置
4. 微调训练

# 优点：可以恢复部分容量
# 缺点：仍是启发式方法
```

**实现**:
```python
def refill_pruning(model, prune_ratio=0.7, fillback_ratio=0.0):
    # 1. 结构化剪枝
    mask = prune_structured(model, prune_ratio)
    
    # 2. 重填充（在重要通道内恢复一些元素）
    if fillback_ratio > 0:
        mask = fillback_important_weights(mask, fillback_ratio)
    
    # 3. 应用mask并微调
    apply_mask(model, mask)
    finetune(model)
```

---

### RSST方法（正则化结构化稀疏训练）

**核心思想**: 训练时施加结构化稀疏正则化，自动学习稀疏结构

```python
# RSST流程
1. 在训练loss中添加通道级/神经元级的正则化项
2. 正则化强度逐渐增加（动态schedule）
3. 自然地将不重要的通道/神经元权重推向0
4. 最后删除这些接近0的结构

# 优点：端到端优化，效果更好
# 缺点：训练时间略长
```

**实现**:
```python
def rsst_training(model, reg_strength_schedule):
    for epoch in range(epochs):
        # 动态正则化强度
        reg_strength = reg_strength_schedule(epoch)
        
        for x, y in dataloader:
            logits = model(x)
            ce_loss = criterion(logits, y)
            
            # 结构化稀疏正则化
            struct_reg = compute_structured_regularization(model, reg_strength)
            
            total_loss = ce_loss + struct_reg
            total_loss.backward()
            optimizer.step()
    
    # 训练结束后，删除接近0的通道/神经元
    prune_near_zero_structures(model)
```

**正则化项定义**:
```python
def compute_structured_regularization(model, reg_strength):
    loss = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 通道级L1正则化（鼓励整个通道变0）
            if 'out_proj' in name or 'fc1' in name:
                channel_norms = module.weight.norm(p=1, dim=1)  # 每个输出通道的L1范数
                loss += reg_strength * channel_norms.sum()
    
    return loss
```

---

## 📊 对比表格

| 特性 | Refill | RSST |
|-----|--------|------|
| **剪枝方式** | 启发式（基于重要性评分） | 端到端优化（正则化） |
| **训练时间** | 正常 | 略长（+10-20%） |
| **精度** | 基线 | 更好（+0.5-1%） |
| **可控性** | 高（手动控制fillback） | 中（依赖schedule） |
| **实现复杂度** | 低 | 中 |
| **推荐场景** | 快速实验、已有预训练模型 | 追求最佳性能 |

---

## 🛠️ 实现计划

### 第1步: 创建Mamba模型 (1天)

**文件**: `models/mamba.py`

**要点**:
- 简化版Mamba实现（纯PyTorch，不依赖CUDA kernel）
- 明确的层命名（便于剪枝识别）
- 支持CIFAR-10/100和ImageNet

**关键组件**:
```python
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.in_proj = nn.Linear(d_model, expand * d_model * 2)  # 可剪枝
        self.conv1d = nn.Conv1d(...)                              # 可剪枝
        self.x_proj = nn.Linear(...)                              # 可剪枝
        self.out_proj = nn.Linear(expand * d_model, d_model)      # ★ 主要剪枝目标
        
class MambaBlock(nn.Module):
    def __init__(self, d_model, use_mlp=True):
        self.ssm = SelectiveSSM(d_model)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),      # ★ 主要剪枝目标
                nn.GELU(),
                nn.Linear(mlp_dim, d_model)       # 协同剪枝
            )
```

---

### 第2步: 结构化剪枝工具 (2天)

**文件**: `mamba_structured_pruning.py`

**核心函数**:

#### 2.1 模型识别
```python
def is_mamba_model(model):
    """判断是否是Mamba模型"""
    from models.mamba import MambaModel
    return isinstance(model, MambaModel)
```

#### 2.2 通道/神经元重要性评分
```python
def calculate_channel_importance(linear_layer, method='l1'):
    """
    计算通道重要性
    Args:
        linear_layer: nn.Linear
        method: 'l1', 'l2', 'taylor'
    Returns:
        importance: [out_features] tensor
    """
    if method == 'l1':
        importance = linear_layer.weight.abs().sum(dim=1)
    elif method == 'l2':
        importance = linear_layer.weight.pow(2).sum(dim=1).sqrt()
    elif method == 'taylor':
        if linear_layer.weight.grad is not None:
            importance = (linear_layer.weight * linear_layer.weight.grad).abs().sum(dim=1)
        else:
            importance = linear_layer.weight.abs().sum(dim=1)
    
    return importance

def calculate_neuron_importance(fc1, fc2):
    """
    计算MLP神经元重要性（完全复用ViT代码）
    """
    # fc1输出 + fc2输入
    importance = fc1.weight.abs().sum(dim=1) + fc2.weight.abs().sum(dim=0)
    return importance
```

#### 2.3 结构化剪枝函数（SSM）
```python
def prune_mamba_ssm_structured(model, prune_ratio=0.7, method='global'):
    """
    结构化剪枝Mamba的SSM输出投影层
    
    Args:
        model: MambaModel
        prune_ratio: 剪枝率（删除70%的通道）
        method: 'global' or 'layerwise'
    
    Returns:
        pruned_model: 剪枝后的模型（实际修改模型结构）
    """
    importance_dict = {}
    
    # 1. 收集所有SSM输出投影层的重要性
    for name, module in model.named_modules():
        if 'ssm.out_proj' in name:
            importance = calculate_channel_importance(module, method='l1')
            importance_dict[name] = importance
    
    # 2. 全局排序或逐层排序
    if method == 'global':
        # 全局排序：所有层合并排序
        all_importance = torch.cat(list(importance_dict.values()))
        threshold = torch.quantile(all_importance, prune_ratio)
        
        # 确定每层保留的通道
        keep_channels = {}
        for name, importance in importance_dict.items():
            keep_channels[name] = (importance >= threshold).nonzero().squeeze()
    
    elif method == 'layerwise':
        # 逐层剪枝：每层独立保留30%
        keep_channels = {}
        for name, importance in importance_dict.items():
            n_keep = int(len(importance) * (1 - prune_ratio))
            keep_channels[name] = torch.topk(importance, n_keep).indices
    
    # 3. 实际修改模型结构
    for block_idx, block in enumerate(model.blocks):
        name = f'blocks.{block_idx}.ssm.out_proj'
        if name in keep_channels:
            channels_to_keep = keep_channels[name]
            
            # 修改out_proj的输入通道
            prune_linear_input_channels(block.ssm.out_proj, channels_to_keep)
            
            # 协同修改上游层
            prune_ssm_upstream_layers(block.ssm, channels_to_keep)
    
    return model

def prune_linear_input_channels(linear_layer, keep_channels):
    """删除Linear层的输入通道"""
    in_features = len(keep_channels)
    out_features = linear_layer.out_features
    
    # 创建新的Linear层
    new_linear = nn.Linear(in_features, out_features, bias=linear_layer.bias is not None)
    
    # 复制保留的权重
    new_linear.weight.data = linear_layer.weight.data[:, keep_channels]
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
    
    # 替换原层
    return new_linear

def prune_ssm_upstream_layers(ssm_module, keep_channels):
    """协同调整SSM的上游层"""
    # in_proj: 输出通道对应剪枝
    # conv1d: 通道数对应剪枝
    # x_proj: 输入通道对应剪枝
    pass  # 详细实现
```

#### 2.4 结构化剪枝函数（MLP）
```python
def prune_mamba_mlp_structured(model, prune_ratio=0.7, method='global'):
    """
    结构化剪枝Mamba的MLP神经元
    与ViT的MLP剪枝完全相同！
    """
    importance_dict = {}
    
    # 1. 收集所有MLP的神经元重要性
    for block_idx, block in enumerate(model.blocks):
        if hasattr(block, 'mlp'):
            fc1 = block.mlp[0]  # Linear
            fc2 = block.mlp[2]  # Linear
            importance = calculate_neuron_importance(fc1, fc2)
            importance_dict[f'blocks.{block_idx}.mlp'] = importance
    
    # 2. 全局或逐层选择保留的神经元
    if method == 'global':
        all_importance = torch.cat(list(importance_dict.values()))
        threshold = torch.quantile(all_importance, prune_ratio)
        keep_neurons = {name: (imp >= threshold).nonzero().squeeze() 
                       for name, imp in importance_dict.items()}
    
    # 3. 实际修改MLP结构
    for block_idx, block in enumerate(model.blocks):
        name = f'blocks.{block_idx}.mlp'
        if name in keep_neurons:
            neurons_to_keep = keep_neurons[name]
            
            # 修改fc1的输出神经元 + fc2的输入神经元
            prune_mlp_neurons(block.mlp, neurons_to_keep)
    
    return model
```

#### 2.5 混合剪枝
```python
def prune_mamba_hybrid(model, ssm_ratio=0.7, mlp_ratio=0.7):
    """同时剪枝SSM和MLP"""
    model = prune_mamba_ssm_structured(model, ssm_ratio)
    model = prune_mamba_mlp_structured(model, mlp_ratio)
    return model
```

---

### 第3步: RSST正则化 (1天)

**文件**: `mamba_structured_pruning.py` (扩展)

```python
def compute_mamba_structured_regularization(model, reg_strength=1e-4, 
                                           reg_target='both'):
    """
    计算Mamba的结构化稀疏正则化
    
    Args:
        model: MambaModel
        reg_strength: 正则化强度
        reg_target: 'ssm', 'mlp', or 'both'
    
    Returns:
        reg_loss: 正则化损失
    """
    reg_loss = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # SSM输出投影的通道级正则化
            if reg_target in ['ssm', 'both'] and 'ssm.out_proj' in name:
                channel_norms = module.weight.norm(p=1, dim=1)
                reg_loss += reg_strength * channel_norms.sum()
            
            # MLP神经元级正则化
            if reg_target in ['mlp', 'both'] and 'mlp' in name:
                if 'mlp.0' in name:  # fc1
                    neuron_norms = module.weight.norm(p=1, dim=1)
                    reg_loss += reg_strength * neuron_norms.sum()
    
    return reg_loss

def rsst_schedule_exp(epoch, total_epochs, base_strength=1e-4, exponent=4):
    """
    RSST的指数增长schedule
    与ViT的RSST完全相同
    """
    progress = epoch / total_epochs
    strength = base_strength * (progress ** exponent)
    return strength
```

---

### 第4步: 集成到主训练脚本 (1天)

**修改**: `main_imp_fillback.py`

#### 4.1 添加命令行参数
```python
# Mamba相关参数
parser.add_argument('--mamba_pretrained', action='store_true',
                    help='use pretrained Mamba model')
parser.add_argument('--mamba_structured', action='store_true',
                    help='use structured pruning for Mamba')
parser.add_argument('--mamba_prune_target', type=str, default='both',
                    choices=['ssm', 'mlp', 'both'],
                    help='which component to prune')
parser.add_argument('--mamba_mlp_prune_ratio', type=float, default=0.7,
                    help='MLP pruning ratio for Mamba')
```

#### 4.2 添加模型判断逻辑
```python
import mamba_structured_pruning

# 在所有需要的地方添加Mamba分支
is_vit = vit_pruning_utils.is_vit_model(model)
is_mamba = mamba_structured_pruning.is_mamba_model(model)

if is_vit:
    # ViT逻辑
elif is_mamba:
    # Mamba逻辑
else:
    # CNN逻辑
```

#### 4.3 剪枝调用
```python
# 在剪枝点
if is_mamba:
    if args.mamba_structured:
        if args.mamba_prune_target == 'ssm':
            model = mamba_structured_pruning.prune_mamba_ssm_structured(
                model, args.rate, method='global'
            )
        elif args.mamba_prune_target == 'mlp':
            model = mamba_structured_pruning.prune_mamba_mlp_structured(
                model, args.mamba_mlp_prune_ratio, method='global'
            )
        else:  # both
            model = mamba_structured_pruning.prune_mamba_hybrid(
                model, args.rate, args.mamba_mlp_prune_ratio
            )
```

#### 4.4 RSST正则化
```python
# 在训练循环中
if is_mamba and args.RST_schedule:
    reg_strength = mamba_structured_pruning.rsst_schedule_exp(
        epoch, args.epochs, args.reg_granularity_prune, args.exponents
    )
    rsst_loss = mamba_structured_pruning.compute_mamba_structured_regularization(
        model, reg_strength, args.mamba_prune_target
    )
    loss += rsst_loss
```

---

### 第5步: 注册到utils.py (0.5天)

```python
# utils.py
from models.mamba import mamba_tiny, mamba_small, mamba_base

def build_model(args, classes):
    # ... 现有代码 ...
    
    elif args.arch == 'mamba_tiny':
        print('build model: mamba_tiny')
        img_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 64
        pretrained = args.mamba_pretrained if hasattr(args, 'mamba_pretrained') else False
        model = mamba_tiny(num_classes=classes, img_size=img_size, pretrained=pretrained)
    
    elif args.arch == 'mamba_small':
        print('build model: mamba_small')
        img_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 64
        pretrained = args.mamba_pretrained if hasattr(args, 'mamba_pretrained') else False
        model = mamba_small(num_classes=classes, img_size=img_size, pretrained=pretrained)
    
    elif args.arch == 'mamba_base':
        print('build model: mamba_base')
        img_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 64
        pretrained = args.mamba_pretrained if hasattr(args, 'mamba_pretrained') else False
        model = mamba_base(num_classes=classes, img_size=img_size, pretrained=pretrained)
```

---

## 🧪 测试与验证

### 测试脚本

**文件**: `test_mamba_structured_pruning.py`

```python
import torch
from models.mamba import mamba_small
import mamba_structured_pruning as msp

def test_basic_forward():
    """测试基本前向传播"""
    model = mamba_small(num_classes=10, img_size=32)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), "Output shape mismatch"
    print("✓ Basic forward pass")

def test_ssm_pruning():
    """测试SSM结构化剪枝"""
    model = mamba_small(num_classes=10, img_size=32)
    
    # 获取原始参数量
    params_before = sum(p.numel() for p in model.parameters())
    
    # 剪枝
    model = msp.prune_mamba_ssm_structured(model, prune_ratio=0.7)
    
    # 获取剪枝后参数量
    params_after = sum(p.numel() for p in model.parameters())
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    print(f"✓ SSM pruning: {params_before:,} → {params_after:,} ({(1-params_after/params_before)*100:.1f}% reduced)")
    assert y.shape == (2, 10), "Output shape mismatch after pruning"

def test_mlp_pruning():
    """测试MLP结构化剪枝"""
    model = mamba_small(num_classes=10, img_size=32)
    params_before = sum(p.numel() for p in model.parameters())
    
    model = msp.prune_mamba_mlp_structured(model, prune_ratio=0.7)
    params_after = sum(p.numel() for p in model.parameters())
    
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    print(f"✓ MLP pruning: {params_before:,} → {params_after:,} ({(1-params_after/params_before)*100:.1f}% reduced)")
    assert y.shape == (2, 10)

def test_hybrid_pruning():
    """测试混合剪枝"""
    model = mamba_small(num_classes=10, img_size=32)
    params_before = sum(p.numel() for p in model.parameters())
    
    model = msp.prune_mamba_hybrid(model, ssm_ratio=0.7, mlp_ratio=0.7)
    params_after = sum(p.numel() for p in model.parameters())
    
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    print(f"✓ Hybrid pruning: {params_before:,} → {params_after:,} ({(1-params_after/params_before)*100:.1f}% reduced)")

def test_rsst_regularization():
    """测试RSST正则化"""
    model = mamba_small(num_classes=10, img_size=32)
    
    reg_loss = msp.compute_mamba_structured_regularization(
        model, reg_strength=1e-4, reg_target='both'
    )
    
    assert reg_loss.item() > 0, "Regularization loss should be positive"
    print(f"✓ RSST regularization: {reg_loss.item():.6f}")

if __name__ == '__main__':
    print("=== Testing Mamba Structured Pruning ===\n")
    test_basic_forward()
    test_ssm_pruning()
    test_mlp_pruning()
    test_hybrid_pruning()
    test_rsst_regularization()
    print("\n✅ All tests passed!")
```

---

## 🚀 启动脚本示例

### Refill方法

```bash
#!/bin/bash
# run_mamba_small_refill.sh

ARCH="mamba_small"
DATASET="cifar10"
PRUNE_RATE=0.7
MLP_PRUNE_RATE=0.7

python main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data datasets/$DATASET \
    --mamba_structured \
    --mamba_prune_target both \
    --rate $PRUNE_RATE \
    --mamba_mlp_prune_ratio $MLP_PRUNE_RATE \
    --pruning_times 16 \
    --epochs 60 \
    --lr 0.01 \
    --batch_size 128 \
    --fillback_rate 0.0 \
    --exp_name mamba_small_refill_70p
```

### RSST方法

```bash
#!/bin/bash
# run_mamba_small_rsst.sh

ARCH="mamba_small"
DATASET="cifar10"
PRUNE_RATE=0.7
MLP_PRUNE_RATE=0.7

python main_imp_fillback.py \
    --arch $ARCH \
    --dataset $DATASET \
    --data datasets/$DATASET \
    --mamba_structured \
    --mamba_prune_target both \
    --rate $PRUNE_RATE \
    --mamba_mlp_prune_ratio $MLP_PRUNE_RATE \
    --pruning_times 16 \
    --epochs 60 \
    --lr 0.01 \
    --batch_size 128 \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --exp_name mamba_small_rsst_70p
```

---

## 📋 开发检查清单

### 阶段1: 模型定义
- [ ] 创建`models/mamba.py`
- [ ] 实现`SelectiveSSM`类
- [ ] 实现`MambaBlock`类
- [ ] 实现`MambaModel`类
- [ ] 实现factory函数（mamba_tiny/small/base）
- [ ] 测试基本前向传播
- [ ] 测试参数量统计

### 阶段2: 结构化剪枝工具
- [ ] 创建`mamba_structured_pruning.py`
- [ ] 实现`is_mamba_model()`
- [ ] 实现`calculate_channel_importance()`
- [ ] 实现`calculate_neuron_importance()`
- [ ] 实现`prune_mamba_ssm_structured()`
- [ ] 实现`prune_mamba_mlp_structured()`
- [ ] 实现`prune_mamba_hybrid()`
- [ ] 实现`compute_mamba_structured_regularization()`
- [ ] 实现`rsst_schedule_exp()`

### 阶段3: 集成
- [ ] 修改`utils.py`注册Mamba模型
- [ ] 修改`main_imp_fillback.py`添加参数
- [ ] 修改`main_imp_fillback.py`添加判断逻辑
- [ ] 修改`main_imp_fillback.py`添加剪枝调用
- [ ] 修改`main_imp_fillback.py`添加RSST正则化

### 阶段4: 测试
- [ ] 创建`test_mamba_structured_pruning.py`
- [ ] 测试基本功能
- [ ] 测试SSM剪枝
- [ ] 测试MLP剪枝
- [ ] 测试混合剪枝
- [ ] 测试RSST正则化

### 阶段5: 实验
- [ ] CIFAR-10 baseline（无剪枝）
- [ ] CIFAR-10 + Refill（70%）
- [ ] CIFAR-10 + RSST（70%）
- [ ] CIFAR-100实验
- [ ] 性能分析（精度、速度、内存）

---

## ✅ 验收标准

### 功能性
- [ ] Mamba模型可正常训练
- [ ] 结构化剪枝正常工作（实际减少参数）
- [ ] RSST正则化正常计算
- [ ] 不影响ViT/ResNet功能

### 性能
- [ ] CIFAR-10 baseline > 85%
- [ ] 70%剪枝后精度下降 < 5%
- [ ] RSST优于Refill（+0.5-1%）
- [ ] 实际推理加速 > 1.5x

### 代码质量
- [ ] 所有测试通过
- [ ] 代码清晰易读
- [ ] 文档完整

---

**准备好开始实施了吗？请确认方案！** 🚀
