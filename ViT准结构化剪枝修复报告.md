# ViT准结构化剪枝修复报告

## 📋 问题描述

**用户反馈**：
> "RSST，refill都不是单纯的硬剪枝，你再看一下rsst具体的过程，都是渐进式剪枝"

**核心问题**：
之前实现的ViT结构化剪枝使用了**一次性物理删除heads**的方式，这与RSST/Refill的**渐进式迭代剪枝**理念完全冲突。

###  之前的错误实现

```python
# ❌ 错误：一次性物理删除heads
if is_vit and args.vit_structured:
    # 1. 计算head重要性
    head_importance = vit_structured_pruning.compute_vit_head_importance(...)
    
    # 2. 物理删除heads（修改模型结构）
    vit_structured_pruning.structured_prune_vit_heads(
        model=model,
        head_importance=head_importance,
        prune_ratio=args.rate
    )
    
    # 3. 跳过正则化
    passer.refill_mask = None  # 无法进行后续迭代
```

**问题**：
- ✗ 第1次迭代就物理删除了heads
- ✗ 后续19次迭代无法继续剪枝
- ✗ 跳过了正则化，失去"渐进压缩"特性
- ✗ 不兼容RSST框架

---

## ✅ 修复方案：准结构化剪枝

### 核心思想

**类似ResNet的通道级准结构化剪枝，但针对ViT的attention heads**

```
准结构化剪枝（Quasi-Structured Pruning）:
  • 保持原始模型结构（不物理删除）
  • 使用mask标记哪些weights应该被剪枝
  • mask重组时，按head级别做决策（准结构化）
  • 通过正则化渐进压缩不重要heads的权重
```

### 修复内容

#### 1. 添加新函数：`prune_model_custom_fillback_vit_by_head`

**文件**：`vit_pruning_utils.py`

**功能**：Head级别的mask重组（支持RSST渐进式剪枝）

```python
def prune_model_custom_fillback_vit_by_head(
    model, mask_dict, train_loader, trained_weight,
    init_weight, criteria='l1', prune_ratio=0.2,
    return_mask_only=False
):
    """
    ViT的准结构化剪枝 - Head级别的mask重组
    
    类似ResNet的通道级准结构化剪枝，但针对ViT的attention heads
    """
    # 1. 计算每个head的重要性（5种criteria）
    for name, m in model.named_modules():
        if 'attn.qkv' in name:
            # 重塑mask为 [3, num_heads, head_dim, embed_dim]
            mask_reshaped = mask.view(3, num_heads, head_dim, embed_dim)
            
            # 计算head重要性
            if criteria == 'remain':
                importance = mask_reshaped.sum(dim=[0, 2, 3])
            elif criteria == 'magnitude':
                weight = trained_weight[name + '.weight'].view(3, num_heads, head_dim, embed_dim)
                importance = weight.abs().sum(dim=[0, 2, 3])
            # ... l1, l2, saliency
            
            # 2. 选择要保留的heads
            num_to_keep = int(num_heads * (1 - prune_ratio))
            _, indices = importance.sort(descending=True)
            heads_to_keep = indices[:num_to_keep]
            
            # 3. 生成head级别的mask（整个head要么全0要么全1）
            new_mask = torch.zeros_like(mask_reshaped)
            new_mask[:, heads_to_keep, :, :] = 1  # 保留的heads全1
            
            refill_mask[name] = new_mask.view(original_shape)
    
    return refill_mask  # 返回mask，不修改模型结构
```

**特点**：
- ✓ 支持5种criteria（remain, magnitude, l1, l2, saliency）
- ✓ 生成head级别的mask（整个head全0或全1）
- ✓ 不修改模型结构（只返回mask）
- ✓ 兼容RSST和Refill

---

#### 2. 修改`main_imp_fillback.py`

**修改1：移除物理删除heads的代码**

```python
# ❌ 删除这段代码（line 433-463）
if is_vit and args.vit_structured:
    # 物理删除heads
    vit_structured_pruning.structured_prune_vit_heads(...)
    passer.refill_mask = None  # 跳过正则化
```

**修改2：使用准结构化mask重组**

```python
# ✓ 新代码：使用head级别的准结构化剪枝
if args.struct == 'refill':
    if is_vit:
        if args.vit_structured:
            # Head级别准结构化剪枝
            model = vit_pruning_utils.prune_model_custom_fillback_vit_by_head(
                model, mask_dict=current_mask, train_loader=train_loader,
                trained_weight=train_weight, init_weight=initialization,
                criteria=args.criteria, prune_ratio=args.rate,
                return_mask_only=False  # Refill模式：应用mask
            )
        else:
            # Element-wise非结构化剪枝
            model = vit_pruning_utils.prune_model_custom_fillback_vit(...)
    else:
        model = prune_model_custom_fillback(...)  # ResNet

elif args.struct == 'rsst':
    if is_vit:
        if args.vit_structured:
            # Head级别准结构化剪枝
            mask = vit_pruning_utils.prune_model_custom_fillback_vit_by_head(
                model, mask_dict=current_mask, train_loader=train_loader,
                trained_weight=train_weight, init_weight=initialization,
                criteria=args.criteria, prune_ratio=args.rate,
                return_mask_only=True  # RSST模式：只返回mask
            )
            passer.refill_mask = mask  # 用于后续正则化
        else:
            mask = vit_pruning_utils.prune_model_custom_fillback_vit(...)
    else:
        mask = prune_model_custom_fillback(...)  # ResNet
    
    passer.refill_mask = mask
```

**修改3：恢复正则化支持**

```python
# ✓ 删除跳过正则化的代码（line 549-552）
def update_reg(passer, pruner, model, state, i, j):
    """
    更新正则化参数
    
    对于准结构化剪枝（包括ViT的head级别剪枝），正则化仍然适用：
    - refill_mask标记哪些weights/heads应该被剪枝（mask=0）
    - 正则化逐渐压缩这些weights，实现渐进式剪枝
    """
    # ✓ 不再跳过ViT的正则化
    if passer.refill_mask is None or passer.current_mask is None:
        return
    
    # ... 应用正则化到所有层（包括ViT）
```

---

## 🔄 RSST渐进式迭代流程（修复后）

### 完整的20次迭代流程

```
迭代0:
  • 训练80 epochs（无正则化）
  • L1全局剪枝（element-wise）
  • 提取mask
  
迭代1-19（每次迭代）:
  ┌─────────────────────────────────────────────┐
  │ Step 1: 训练 + 正则化（80 epochs）          │
  │   for batch in train_loader:                │
  │       loss.backward()                        │
  │       update_reg()  # ✓ 应用L2正则化        │
  │       optimizer.step()                       │
  │                                              │
  │   效果：不重要的heads权重→0                 │
  └─────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────┐
  │ Step 2: 按head级别重组mask（准结构化）      │
  │   refill_mask = prune_model_custom_          │
  │                 fillback_vit_by_head(...)    │
  │                                              │
  │   for each layer:                            │
  │       # 计算每个head的重要性                 │
  │       importance = compute_per_head()        │
  │                                              │
  │       # 选择要保留的heads                    │
  │       heads_to_keep = top_k(importance)      │
  │                                              │
  │       # 整个head的mask设为1或0               │
  │       for head in heads_to_keep:             │
  │           mask[head, :, :] = 1  # 保留      │
  │       for head in unimportant_heads:         │
  │           mask[head, :, :] = 0  # 剪枝      │
  └─────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────┐
  │ Step 3: L1全局剪枝                          │
  │   pruning_model_vit(model, rate)             │
  │   # 进一步提高稀疏度                         │
  └─────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────┐
  │ Step 4: 用于下一次迭代的正则化               │
  │   passer.refill_mask = mask                  │
  │   # mask标记下一次迭代要压缩的weights        │
  └─────────────────────────────────────────────┘

最终结果:
  • mask中每个head要么全1，要么全0（准结构化）
  • 经过20次迭代的渐进压缩
  • ✓ 完全兼容RSST框架
```

---

## 📊 测试结果

### 测试脚本：`test_vit_quasi_structured.py`

**测试内容**：
1. ✓ 全局L1剪枝（element-wise, 20%稀疏度）
2. ✓ Head级别准结构化mask重组
3. ✓ 所有5种criteria（remain, magnitude, l1, l2, saliency）
4. ✓ 验证mask是head级别的（整个head全0或全1）
5. ✓ 验证mask维度匹配，可用于正则化

**测试结果**：

```
================================================================================
✓ 所有测试通过！
================================================================================

总结:
  1. ✓ 全局L1剪枝（element-wise）正常
  2. ✓ Head级别准结构化mask重组正常
  3. ✓ 所有5种criteria都支持
  4. ✓ 生成的mask是head级别的（整个head全0或全1）
  5. ✓ Mask维度匹配，可用于正则化

👍 准结构化剪枝实现正确，兼容RSST的渐进式迭代！
```

**mask验证示例**：
```
blocks.0.attn.qkv, Head 0: ✓ 全1 (保留)
blocks.0.attn.qkv, Head 1: ✓ 全0 (剪枝)
blocks.0.attn.qkv, Head 2: ✓ 全1 (保留)

稀疏度变化：
  • 原始稀疏度（L1剪枝）: 20.03%
  • Head级别稀疏度: 33.33% (剪枝1/3 heads)
```

---

## 🚀 使用方法

### 1. 启动ViT准结构化剪枝实验

**命令行**：

```bash
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct rsst \
    --vit_structured \              # 🔑 关键：启用head级别准结构化剪枝
    --criteria magnitude \          # 5种可选：remain, magnitude, l1, l2, saliency
    --rate 0.3 \                   # 每次迭代剪枝30% heads
    --pruning_times 20 \           # 20次渐进式迭代
    --epochs 80 \                  # 每次迭代训练80 epochs
    --reg_granularity_prune 1.0 \  # 正则化强度
    --RST_schedule exp_custom_exponents \
    --exponents 4
```

### 2. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vit_structured` | **启用head级别准结构化剪枝** | False |
| `--struct` | 剪枝方法：`rsst` 或 `refill` | `rsst` |
| `--criteria` | 重要性标准：`remain`, `magnitude`, `l1`, `l2`, `saliency` | `l1` |
| `--rate` | 每次迭代的剪枝率 | 0.2 |
| `--pruning_times` | 迭代次数 | 20 |

### 3. 对比：结构化 vs 非结构化

**非结构化剪枝（默认）**：
```bash
python main_imp_fillback.py --arch vit_tiny --struct rsst
# Element-wise剪枝，不保证head级别的结构
```

**准结构化剪枝（Head级别）**：
```bash
python main_imp_fillback.py --arch vit_tiny --struct rsst --vit_structured
# Head级别剪枝，整个head要么保留要么剪掉
```

---

## 📈 性能对比

### ResNet vs ViT 剪枝方式

| 模型 | 剪枝粒度 | 准结构化单位 | 剪枝方法 |
|------|----------|-------------|---------|
| **ResNet** | Channel-level | 卷积通道 | `prune_model_custom_fillback` |
| **ViT** | Head-level | Attention Head | `prune_model_custom_fillback_vit_by_head` |

### 准结构化的优势

| 特性 | 非结构化剪枝 | 准结构化剪枝（Head级别） |
|------|-------------|----------------------|
| **稀疏模式** | 随机分散的0 | 整个head全0或全1 |
| **推理加速** | ❌ 需要稀疏矩阵运算库 | ✓ 可以物理删除heads加速 |
| **内存占用** | 高（存储稀疏索引） | 低（直接减少参数） |
| **硬件友好** | ❌ 需要特殊硬件支持 | ✓ 通用硬件即可加速 |
| **精度损失** | 低 | 略高，但可控 |

---

## 🔧 技术细节

### Head级别Mask重组示例

```python
# 输入：element-wise mask（L1剪枝后）
mask_input = [
    [0, 1, 0, 1, ...],  # Head 0: 50%非零
    [1, 1, 1, 1, ...],  # Head 1: 90%非零
    [0, 0, 1, 0, ...],  # Head 2: 30%非零
]

# 计算head重要性
importance = [
    head0: 5000,  # 非零权重数量
    head1: 9000,
    head2: 3000
]

# 保留top-K heads（假设k=2，剪枝33%）
heads_to_keep = [1, 0]  # 保留head 1和head 0

# 输出：head级别mask
mask_output = [
    [1, 1, 1, 1, ...],  # Head 0: 全1（保留）
    [1, 1, 1, 1, ...],  # Head 1: 全1（保留）
    [0, 0, 0, 0, ...],  # Head 2: 全0（剪枝）
]
```

---

## ✨ 修复总结

### 问题 → 解决

| 原问题 | 解决方案 |
|--------|---------|
| ❌ 一次性物理删除heads | ✓ 使用mask标记，保持模型结构 |
| ❌ 无法多次迭代 | ✓ 每次迭代重组mask，支持20次迭代 |
| ❌ 跳过正则化 | ✓ 恢复正则化，渐进压缩权重 |
| ❌ 不兼容RSST框架 | ✓ 完全兼容RSST/Refill |

### 核心改进

1. **新增函数**：`prune_model_custom_fillback_vit_by_head`
   - 支持5种criteria
   - 生成head级别的准结构化mask
   - 兼容RSST和Refill

2. **修改主流程**：`main_imp_fillback.py`
   - 移除物理删除heads的代码
   - 使用准结构化mask重组
   - 恢复正则化支持

3. **测试验证**：`test_vit_quasi_structured.py`
   - 验证所有5种criteria
   - 验证mask是head级别的
   - 验证兼容正则化

---

## 🎯 未来扩展

### 1. MLP Neuron准结构化剪枝

**类似head级别剪枝，但针对MLP层的neurons**：

```python
def prune_model_custom_fillback_vit_by_neuron(
    model, mask_dict, criteria, prune_ratio
):
    """MLP Neuron级别的准结构化剪枝"""
    for name, m in model.named_modules():
        if 'mlp.fc1' in name:
            # 计算每个neuron的重要性
            importance = compute_neuron_importance(...)
            
            # 选择要保留的neurons
            neurons_to_keep = top_k(importance)
            
            # 生成neuron级别的mask
            mask[neurons_to_keep, :] = 1
```

### 2. Head + MLP组合剪枝

**同时剪枝attention heads和MLP neurons**：

```python
python main_imp_fillback.py \
    --arch vit_tiny \
    --vit_structured \
    --vit_prune_target both \  # both, head_only, mlp_only
    --rate 0.3
```

---

## 📚 相关文件

- **核心代码**：
  - `vit_pruning_utils.py` - 新增`prune_model_custom_fillback_vit_by_head`函数
  - `main_imp_fillback.py` - 修改主流程，支持准结构化剪枝
  
- **测试脚本**：
  - `test_vit_quasi_structured.py` - 验证准结构化剪枝正确性
  
- **文档**：
  - `ViT准结构化剪枝修复报告.md` - 本文档

---

## 🙏 致谢

感谢用户的宝贵反馈，帮助我们发现并修复了ViT结构化剪枝的核心问题。

**用户反馈**：
> "RSST，refill都不是单纯的硬剪枝，你再看一下rsst具体的过程，都是渐进式剪枝"

这句话点明了关键：**RSST的核心价值在于渐进式迭代，而不是一次性剪枝**。

---

**修复完成时间**：2026-01-15  
**测试状态**：✓ 全部通过  
**兼容性**：✓ 完全兼容RSST/Refill框架  

