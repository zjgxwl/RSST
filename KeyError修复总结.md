
# KeyError: 'blocks.0.attn.qkv.weight' 修复总结

## 🐛 问题根源

**问题**: `KeyError: 'blocks.0.attn.qkv.weight'`

**根本原因**: PyTorch的`prune.CustomFromMask.apply()`会修改`state_dict()`的键名结构！

### 详细说明

当对模型应用`prune.CustomFromMask.apply(m, 'weight', mask)`后：

```python
# 应用前
state_dict = {'layer.weight': tensor(...), 'layer.bias': tensor(...)}

# 应用后  
state_dict = {
    'layer.weight_orig': tensor(...),  # 原始权重
    'layer.weight_mask': tensor(...),  # mask
    'layer.bias': tensor(...)
}
# 注意：'layer.weight' 键消失了！
```

### 问题发生时机

```python
for state in range(pruning_times):
    # State 0: 训练 → 应用Refill剪枝 → model带着prune hooks
    # State 1: 训练（model已有prune） → 保存train_weight
    
    train_weight = model.state_dict()  # ❌ 包含weight_orig而不是weight
    
    # 调用准结构化剪枝
    prune_model_custom_fillback_vit_head_and_mlp(
        ...,
        trained_weight=train_weight,  # ❌ 尝试访问train_weight['xxx.weight']
        ...
    )
    # KeyError: 'xxx.weight' 不存在！
```

---

## ✅ 解决方案

### 方案1: 修改权重访问逻辑（采用）

在`vit_pruning_utils_head_mlp.py`中，优先使用`weight_orig`，如果不存在则使用`weight`：

```python
# 修改前
weight = trained_weight[name + '.weight']

# 修改后
weight_key = name + '.weight_orig' if (name + '.weight_orig') in trained_weight else name + '.weight'
weight = trained_weight[weight_key]
```

**优点**: 
- ✅ 兼容两种情况（有prune和无prune）
- ✅ 不需要修改主训练流程
- ✅ 简单直接

---

## 📝 修改的文件

### 1. `vit_pruning_utils_head_mlp.py`

修改了所有访问`trained_weight[name + '.weight']`的地方（共15处）：

- **Head剪枝部分** (line 133-150):
  - Global排序的重要性计算
  - Layer-wise排序的重要性计算
  
- **MLP剪枝部分** (line 327-430):
  - Global排序的重要性计算  
  - Layer-wise排序的重要性计算

**修改模式**:
```python
# 所有的magnitude/l1/l2/saliency criteria都需要修改
weight_key = name + '.weight_orig' if (name + '.weight_orig') in trained_weight else name + '.weight'
weight = trained_weight[weight_key]
```

### 2. `main_imp_fillback.py`

之前的修改（移动`train_weight`位置）保持不变：
```python
# line 455: 在非结构化剪枝之前保存
train_weight = model.state_dict()
```

---

## 🧪 测试验证

### 测试配置
- 数据集: CIFAR-10
- 迭代次数: 3次（快速测试）
- 每次迭代: 2个epoch  
- 排序模式: global (混合排序)
- 剪枝率: 30%

### 关键验证点
- ✅ State 0不会因KeyError崩溃
- ✅ State 1+能够正确访问训练后的权重
- ✅ Global sorting逻辑正常工作
- ✅ RSST正则化流程正常

---

## 🎓 经验教训

1. **PyTorch prune机制**: `prune.CustomFromMask.apply()`会修改`state_dict`结构
2. **权重访问**: 访问带prune的模型权重时，需要使用`weight_orig`
3. **兼容性**: 代码需要兼容两种情况（有prune和无prune）
4. **调试技巧**: 使用简单的测试脚本验证`state_dict`的键名

---

## ✨ 最终状态

**Refill方法现在可以正常工作**:
1. ✅ Head级别结构化剪枝
2. ✅ MLP Neuron级别结构化剪枝
3. ✅ Layer-wise和Global两种排序模式
4. ✅ 支持迭代式训练和剪枝
5. ✅ 兼容PyTorch prune机制

