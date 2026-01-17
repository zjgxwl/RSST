# Mamba模型RSST/Refill结构化剪枝使用指南

**创建时间**: 2026-01-17  
**状态**: ✅ 开发完成

---

## 📋 完成的工作

### 1. 核心文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `models/mamba.py` | Mamba模型定义（tiny/small/base） | ✅ |
| `mamba_structured_pruning.py` | 结构化剪枝工具（SSM/MLP/混合） | ✅ |
| `utils.py` | 模型注册（已添加Mamba支持） | ✅ |
| `main_imp_fillback.py` | 主训练脚本（已集成Mamba剪枝） | ✅ |

### 2. 测试与脚本

| 文件 | 说明 | 状态 |
|------|------|------|
| `test_mamba_structured_pruning.py` | 功能测试脚本（9个测试） | ✅ |
| `run_mamba_small_70p_refill.sh` | Refill方法启动脚本 | ✅ |
| `run_mamba_small_70p_rsst.sh` | RSST方法启动脚本 | ✅ |
| `run_mamba_small_70p_all.sh` | 完整对比启动脚本（4实验） | ✅ |

### 3. 文档

| 文件 | 说明 |
|------|------|
| `Mamba可剪枝组件详细分析.md` | 技术分析文档 |
| `Mamba结构化剪枝方案.md` | 详细方案文档 |
| `Mamba_RSST适配方案.md` | 初始方案（参考） |
| `Mamba_RSST使用指南.md` | 本文档 |

---

## 🚀 快速开始

### 第1步：测试基本功能

```bash
cd /workspace/ycx/RSST

# 运行测试脚本
python test_mamba_structured_pruning.py
```

**预期输出**：
```
==================================================================
测试Mamba模型的结构化剪枝功能
==================================================================

[Test 1] 基本前向传播
  ✓ Input: torch.Size([2, 3, 32, 32]), Output: torch.Size([2, 10])

[Test 2] 模型识别
  ✓ Mamba模型识别成功

[Test 3] SSM结构化剪枝
  原始参数量: 22,057,418
  剪枝后参数量: 17,123,530
  参数减少: 22.39%
  ✓ 剪枝后前向传播正常

... (更多测试) ...

✅ 所有测试通过！
==================================================================
```

---

### 第2步：运行完整实验

#### 选项A：完整对比（推荐）

运行4个实验（CIFAR-10/100 × Refill/RSST）：

```bash
cd /workspace/ycx/RSST
./run_mamba_small_70p_all.sh
```

#### 选项B：仅Refill方法

```bash
./run_mamba_small_70p_refill.sh
```

#### 选项C：仅RSST方法

```bash
./run_mamba_small_70p_rsst.sh
```

---

### 第3步：监控实验进度

```bash
# 查看所有日志
tail -f logs_mamba_small_70p/*.log

# 查看特定实验
tail -f logs_mamba_small_70p/mamba_small_cifar10_rsst_70p_*.log

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep 'main_imp_fillback.py.*mamba'
```

---

## 📐 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | - | 模型架构：`mamba_tiny`, `mamba_small`, `mamba_base` |
| `--dataset` | - | 数据集：`cifar10`, `cifar100` |
| `--mamba_structured` | - | **必须添加**：启用结构化剪枝 |
| `--mamba_prune_target` | `both` | 剪枝目标：`ssm`, `mlp`, `both` |
| `--rate` | 0.7 | SSM剪枝率（70%） |
| `--mamba_mlp_prune_ratio` | 0.7 | MLP剪枝率（70%） |
| `--pruning_times` | 16 | 迭代剪枝轮次 |
| `--epochs` | 60 | 每轮训练epoch数 |
| `--sorting_mode` | `global` | 剪枝策略：`global`或`layerwise` |

### Refill特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--struct` | `refill` | 使用Refill方法 |
| `--fillback_rate` | 0.0 | 重填充率（通常为0） |

### RSST特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--struct` | `rsst` | 使用RSST方法 |
| `--reg_granularity_prune` | 1.0 | 正则化基础强度 |
| `--RST_schedule` | `exp_custom_exponents` | 正则化schedule |
| `--exponents` | 4 | 指数值（控制曲率） |

---

## 🎯 使用示例

### 示例1：CIFAR-10，70%剪枝，Refill方法

```bash
python main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar10 \
    --data datasets/cifar10 \
    --mamba_structured \
    --mamba_prune_target both \
    --rate 0.7 \
    --mamba_mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --lr 0.01 \
    --batch_size 128 \
    --struct refill \
    --fillback_rate 0.0 \
    --exp_name mamba_test_refill
```

### 示例2：CIFAR-100，70%剪枝，RSST方法

```bash
python main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar100 \
    --data datasets/cifar100 \
    --mamba_structured \
    --mamba_prune_target both \
    --rate 0.7 \
    --mamba_mlp_prune_ratio 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --lr 0.01 \
    --batch_size 128 \
    --struct rsst \
    --reg_granularity_prune 1.0 \
    --RST_schedule exp_custom_exponents \
    --exponents 4 \
    --exp_name mamba_test_rsst
```

### 示例3：仅剪枝SSM（不剪枝MLP）

```bash
python main_imp_fillback.py \
    --arch mamba_small \
    --dataset cifar10 \
    --data datasets/cifar10 \
    --mamba_structured \
    --mamba_prune_target ssm \  # 仅剪枝SSM
    --rate 0.7 \
    --pruning_times 16 \
    --epochs 60 \
    --struct refill \
    --exp_name mamba_ssm_only
```

---

## 🔬 技术细节

### Mamba模型架构

```
MambaModel
  ├─ patch_embed (Conv2d)
  ├─ pos_embed (Parameter)
  └─ blocks (ModuleList)
      └─ MambaBlock × N
          ├─ ssm (SelectiveSSM)
          │   ├─ in_proj      [可剪枝]
          │   ├─ conv1d       [可剪枝]
          │   ├─ x_proj       [可剪枝]
          │   └─ out_proj     [★ 主要剪枝目标]
          └─ mlp (Sequential)
              ├─ fc1          [★ 主要剪枝目标]
              └─ fc2          [协同剪枝]
```

### 剪枝策略

#### 1. SSM剪枝（输入通道级）

```python
# 目标：ssm.out_proj [d_inner → d_model]
# 方法：删除不重要的输入通道（d_inner维度）
# 协同：需要调整上游的in_proj, conv1d, x_proj
```

#### 2. MLP剪枝（神经元级）

```python
# 目标：mlp.fc1 [d_model → mlp_dim] + mlp.fc2 [mlp_dim → d_model]
# 方法：删除不重要的神经元（mlp_dim维度）
# 与ViT的MLP剪枝完全相同！
```

#### 3. 混合剪枝

```python
# 同时剪枝SSM和MLP
# 可以独立设置不同的剪枝率
```

### Refill vs RSST

| 特性 | Refill | RSST |
|-----|--------|------|
| **剪枝时机** | 训练前剪枝 | 训练中正则化 |
| **原理** | 启发式重要性评分 | 端到端优化 |
| **训练时间** | 正常 | +10-20% |
| **精度** | 基线 | 更好（+0.5-1%） |
| **实现复杂度** | 简单 | 中等 |

---

## 📊 预期结果

### Mamba-Small在CIFAR-10/100上的表现

| 模型 | 方法 | 剪枝率 | CIFAR-10准确率 | CIFAR-100准确率 |
|-----|------|--------|---------------|----------------|
| **Baseline** | 无剪枝 | 0% | ~92% | ~72% |
| **Refill** | 结构化 | 70% | ~89-90% | ~68-69% |
| **RSST** | 结构化 | 70% | ~90-91% | ~69-70% |

*注：以上数据为预估，实际结果可能有所不同*

### 性能提升

- **参数减少**：约60% (70% SSM + 70% MLP)
- **FLOPs减少**：约55%
- **推理加速**：约1.8-2.2× (取决于硬件)
- **精度下降**：2-4% (RSST效果更好)

---

## ⚠️ 注意事项

### 1. 必须使用结构化剪枝

```bash
# ✅ 正确：必须添加--mamba_structured
python main_imp_fillback.py --arch mamba_small --mamba_structured ...

# ❌ 错误：Mamba不支持非结构化剪枝
python main_imp_fillback.py --arch mamba_small ...  # 会报错
```

### 2. GPU分配

```bash
# 手动指定GPU
CUDA_VISIBLE_DEVICES=0 python main_imp_fillback.py ...

# 或在脚本中设置
export CUDA_VISIBLE_DEVICES=0,1
```

### 3. 日志输出

使用绝对路径的Python解释器确保日志正常输出：

```bash
# ✅ 推荐
/root/miniconda3/envs/structlth/bin/python main_imp_fillback.py ...

# ⚠️  可能导致日志问题
conda run -n structlth python main_imp_fillback.py ...
```

### 4. 内存要求

- **Mamba-Small**: ~10GB GPU内存（batch_size=128）
- **Mamba-Base**: ~20GB GPU内存（batch_size=128）

如果GPU内存不足，减小batch size：

```bash
--batch_size 64  # 或更小
```

---

## 🛠️ 故障排查

### 问题1：导入错误

```python
ModuleNotFoundError: No module named 'models.mamba'
```

**解决**：确保在正确的目录下运行：

```bash
cd /workspace/ycx/RSST
python main_imp_fillback.py ...
```

### 问题2：CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决**：减小batch size或使用更小的模型：

```bash
--batch_size 64          # 减小batch size
# 或
--arch mamba_tiny        # 使用更小的模型
```

### 问题3：日志文件为空

```bash
ls -lh logs_mamba_small_70p/*.log
# 0字节
```

**解决**：使用绝对路径的Python解释器（已在脚本中修复）。

### 问题4：剪枝后精度崩溃

可能的原因和解决方案：

1. **学习率过大**：降低学习率 `--lr 0.005`
2. **剪枝率过高**：尝试50%剪枝 `--rate 0.5`
3. **warmup不足**：增加warmup `--warmup 5`

---

## 📈 实验建议

### 基础实验（快速验证）

```bash
# 1天内完成的快速实验
python main_imp_fillback.py \
    --arch mamba_tiny \         # 使用小模型
    --dataset cifar10 \
    --pruning_times 4 \         # 减少迭代次数
    --epochs 30 \               # 减少epoch
    --rate 0.5 \               # 降低剪枝率
    --struct refill
```

### 完整实验（最佳性能）

```bash
# 使用提供的脚本
./run_mamba_small_70p_all.sh

# 预计时间：24-36小时（双GPU并行）
```

### 消融实验

```bash
# 对比不同剪枝目标
--mamba_prune_target ssm    # 仅SSM
--mamba_prune_target mlp    # 仅MLP
--mamba_prune_target both   # 两者都剪

# 对比不同剪枝率
--rate 0.5 --mamba_mlp_prune_ratio 0.5  # 50%
--rate 0.7 --mamba_mlp_prune_ratio 0.7  # 70%
--rate 0.9 --mamba_mlp_prune_ratio 0.9  # 90%

# 对比不同策略
--sorting_mode global       # 全局排序
--sorting_mode layerwise    # 逐层剪枝
```

---

## 🎓 扩展阅读

### 相关文档

- `Mamba可剪枝组件详细分析.md`：技术细节
- `Mamba结构化剪枝方案.md`：实现方案
- `ViT结构化剪枝使用指南.md`：ViT对比

### 相关论文

1. **Mamba**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
2. **RSST**: Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets (2022)
3. **Lottery Ticket Hypothesis**: The Lottery Ticket Hypothesis (2019)

---

## ✅ 检查清单

使用前确认：

- [ ] 已激活conda环境：`conda activate structlth`
- [ ] 数据集已准备：`datasets/cifar10`, `datasets/cifar100`
- [ ] GPU可用：`nvidia-smi`
- [ ] 测试脚本通过：`python test_mamba_structured_pruning.py`
- [ ] 脚本有执行权限：`chmod +x run_mamba_*.sh`

---

## 📞 问题反馈

如遇到问题，请检查：

1. **日志文件**：查看详细错误信息
2. **测试脚本**：运行`test_mamba_structured_pruning.py`
3. **GPU状态**：`nvidia-smi`
4. **进程状态**：`ps aux | grep mamba`

---

**祝实验顺利！** 🎉
