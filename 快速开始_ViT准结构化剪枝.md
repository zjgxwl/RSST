# 快速开始：ViT准结构化剪枝

## 🎯 核心改进

✓ **修复前（错误）**：一次性物理删除heads，无法迭代  
✓ **修复后（正确）**：Head级别准结构化mask重组，支持20次渐进式迭代

---

## 🚀 快速启动

### 1. 基础测试（验证功能）

```bash
# 运行测试脚本，验证准结构化剪枝是否正常
python test_vit_quasi_structured.py
```

**预期输出**：
```
✓ 所有测试通过！
  1. ✓ 全局L1剪枝（element-wise）正常
  2. ✓ Head级别准结构化mask重组正常
  3. ✓ 所有5种criteria都支持
  4. ✓ 生成的mask是head级别的（整个head全0或全1）
  5. ✓ Mask维度匹配，可用于正则化
```

---

### 2. RSST + ViT准结构化剪枝（推荐）

```bash
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct rsst \
    --vit_structured \
    --criteria magnitude \
    --rate 0.3 \
    --pruning_times 20 \
    --epochs 80 \
    --batch_size 128 \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --exp_name vit_rsst_head30_test
```

---

### 3. Refill + ViT准结构化剪枝

```bash
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct refill \
    --vit_structured \
    --criteria magnitude \
    --fillback_rate 0.1 \
    --rate 0.3 \
    --pruning_times 20 \
    --epochs 80 \
    --exp_name vit_refill_head30_test
```

---

## 📊 对比实验

### 非结构化 vs 准结构化

```bash
# 1. 非结构化剪枝（baseline）
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct rsst \
    --rate 0.3 \
    --exp_name vit_unstructured_30

# 2. 准结构化剪枝（head级别）
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct rsst \
    --vit_structured \
    --rate 0.3 \
    --exp_name vit_structured_head30
```

---

## 🎛️ 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--vit_structured` | **启用head级别准结构化剪枝** | 必需 |
| `--criteria` | `remain`/`magnitude`/`l1`/`l2`/`saliency` | `magnitude` |
| `--rate` | 每次迭代剪枝率 | 0.2-0.3 |
| `--pruning_times` | 迭代次数 | 20 |
| `--reg_granularity_prune` | 正则化强度（RSST） | 1.0 |
| `--fillback_rate` | 恢复比例（Refill） | 0.1 |

---

## 🔍 查看结果

### WandB

所有实验会自动上传到WandB：

- 项目名：`RSST`
- 实验名：自动生成（包含方法、模型、数据集、参数等）

### 本地日志

```bash
# 查看保存目录
ls -la cifar100_rsst_output_*/

# 查看checkpoint
ls -la cifar100_rsst_output_*/[0-9]*checkpoint.pth.tar
```

---

## ✅ 验证准结构化剪枝是否生效

### 方法1：查看WandB日志

在实验名中查找：
- `struct_head`：表示启用了准结构化剪枝

### 方法2：查看终端输出

```
================================================================================
[ViT Head-level Quasi-Structured Pruning]
  Criteria: magnitude
  Prune Ratio: 0.3
  Mode: RSST (mask only)
================================================================================

Layer: blocks.0.attn.qkv
  Shape: torch.Size([576, 192])
  Num heads: 3, Head dim: 64, Embed dim: 192
  Head importance: [570.12 572.15 567.86]
  Keeping 2/3 heads: [1 0]
  Original sparsity: 19.89%
  New sparsity: 33.33% (head-level)  ← 注意这里！
```

### 方法3：检查稀疏度

准结构化剪枝的稀疏度应该是head数量的整数倍：
- 3 heads，剪枝1个 → 33.33%
- 3 heads，剪枝2个 → 66.67%

---

## 🛠️ 故障排除

### 问题1：提示"不支持vit_structured"

**原因**：使用了旧版本代码

**解决**：
```bash
git pull  # 更新代码
python test_vit_quasi_structured.py  # 验证功能
```

---

### 问题2：稀疏度为0%（RSST）

**原因**：这是正常现象！

**解释**：
- RSST使用正则化渐进压缩权重，不会立即设为0
- 权重会逐渐接近0，但不会显示为稀疏
- Refill方法会显示稀疏度

**验证RSST是否工作**：
```bash
# 查看正则化lambda增长
wandb log | grep reg_lambda

# 查看权重分布变化
# 随着迭代进行，权重应该逐渐接近0
```

---

### 问题3：内存不足

**解决**：
```bash
# 减小batch size
python main_imp_fillback.py ... --batch_size 64

# 或使用更小的模型
python main_imp_fillback.py --arch vit_tiny ...  # 推荐
```

---

## 📈 预期结果

### CIFAR-100, ViT-Tiny

| 方法 | 剪枝率 | 准确率（预期） | 压缩率 |
|------|--------|---------------|--------|
| Baseline（无剪枝） | 0% | ~67% | 1x |
| RSST（非结构化） | 30% | ~65-66% | 1.4x |
| RSST（准结构化） | 30% | ~64-65% | 1.5x |
| RSST（准结构化） | 50% | ~62-64% | 2x |

*注：具体结果取决于训练超参数和随机种子*

---

## 🎓 进一步探索

### 1. 不同criteria对比

```bash
for criteria in remain magnitude l1 l2 saliency; do
    python main_imp_fillback.py \
        --arch vit_tiny \
        --struct rsst \
        --vit_structured \
        --criteria $criteria \
        --rate 0.3 \
        --exp_name vit_head30_${criteria}
done
```

### 2. 不同剪枝率对比

```bash
for rate in 0.2 0.3 0.4 0.5; do
    python main_imp_fillback.py \
        --arch vit_tiny \
        --struct rsst \
        --vit_structured \
        --rate $rate \
        --exp_name vit_head_rate${rate}
done
```

### 3. 与ResNet对比

```bash
# ResNet-20 准结构化剪枝
python main_imp_fillback.py \
    --arch res20s \
    --dataset cifar100 \
    --struct rsst \
    --rate 0.3

# ViT-Tiny 准结构化剪枝  
python main_imp_fillback.py \
    --arch vit_tiny \
    --dataset cifar100 \
    --struct rsst \
    --vit_structured \
    --rate 0.3
```

---

## 📝 注意事项

1. **必须添加`--vit_structured`标志**才能启用准结构化剪枝
2. **RSST不会显示稀疏度**（使用正则化渐进压缩，不会立即设为0）
3. **准结构化剪枝的稀疏度是head数量的整数倍**
4. **第一次迭代（迭代0）不使用正则化**，从迭代1开始

---

## 🔗 相关文档

- **详细报告**：`ViT准结构化剪枝修复报告.md`
- **测试脚本**：`test_vit_quasi_structured.py`
- **核心代码**：`vit_pruning_utils.py`, `main_imp_fillback.py`

---

**祝实验顺利！** 🚀

如有问题，请查看详细报告或运行测试脚本验证功能。
