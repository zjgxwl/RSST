# Mamba模型适配RSST算法 - 详细方案

**创建时间**: 2026-01-17  
**分支**: `mamba-rsst`  
**状态**: 待审核

---

## 📋 目录
1. [背景与目标](#背景与目标)
2. [当前代码库分析](#当前代码库分析)
3. [Mamba模型架构分析](#mamba模型架构分析)
4. [适配策略](#适配策略)
5. [实施计划（6个阶段）](#实施计划)
6. [技术细节与挑战](#技术细节与挑战)
7. [风险与应对](#风险与应对)
8. [验收标准](#验收标准)

---

## 🎯 背景与目标

### 项目背景
- **RSST算法**: Regularized Structured Sparsity Training（正则化结构化稀疏训练）
- **当前支持模型**: ResNet系列、VGG系列、MobileNet系列、ViT系列
- **目标**: 将RSST算法适配到Mamba（State Space Model）架构上

### 核心目标
1. **功能性**: 实现Mamba模型的结构化剪枝（头级别、MLP神经元级别）
2. **一致性**: 保持与现有ViT剪枝流程的一致性
3. **兼容性**: 不破坏现有代码，支持无缝切换
4. **性能**: 验证Mamba+RSST的剪枝效果

---

## 📊 当前代码库分析

### 1. 核心文件结构
```
RSST/
├── models/                      # 模型定义
│   ├── vit.py                  # ViT模型（参考对象）
│   ├── resnet.py               # ResNet模型
│   └── [待添加] mamba.py       # Mamba模型
├── utils.py                     # 模型构建入口（build_model函数）
├── main_imp_fillback.py        # 主训练脚本（Refill方法）
├── vit_pruning_utils.py        # ViT专用剪枝工具
├── pruning_utils.py            # 通用剪枝工具（CNN）
└── [待添加] mamba_pruning_utils.py  # Mamba专用剪枝工具
```

### 2. 模型注册机制
**位置**: `utils.py` 的 `build_model()` 函数

**现有模式**（以ViT为例）:
```python
elif args.arch == 'vit_small':
    print('build model: vit_small')
    img_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 64
    pretrained = args.vit_pretrained if hasattr(args, 'vit_pretrained') else False
    model = vit_small(num_classes=classes, img_size=img_size, pretrained=pretrained)
```

**需要添加**:
```python
elif args.arch == 'mamba_small':
    print('build model: mamba_small')
    pretrained = args.mamba_pretrained if hasattr(args, 'mamba_pretrained') else False
    model = mamba_small(num_classes=classes, pretrained=pretrained)
```

### 3. 剪枝工具架构

#### 3.1 模型识别函数
```python
# vit_pruning_utils.py
def is_vit_model(model):
    from models.vit import VisionTransformer
    return isinstance(model, VisionTransformer)
```

**需要添加**:
```python
# mamba_pruning_utils.py
def is_mamba_model(model):
    from models.mamba import MambaModel
    return isinstance(model, MambaModel)
```

#### 3.2 剪枝函数
**ViT的剪枝方法**（参考）:
- `pruning_model_vit(model, px)`: 非结构化L1剪枝
- `prune_model_custom_vit(model, mask_dict)`: 自定义mask剪枝
- `extract_mask_vit(model)`: 提取剪枝mask
- `remove_prune_vit(model)`: 移除剪枝钩子

**Mamba需要实现**:
- `pruning_model_mamba(model, px)`: 非结构化剪枝
- `prune_model_custom_mamba(model, mask_dict)`: 自定义mask剪枝
- `extract_mask_mamba(model)`: 提取mask
- `remove_prune_mamba(model)`: 移除钩子

### 4. 主训练脚本集成点

**关键位置**（`main_imp_fillback.py`）:

1. **命令行参数** (行40-60):
```python
parser.add_argument('--mamba_pretrained', action='store_true', 
                    help='use pretrained model (for Mamba)')
parser.add_argument('--mamba_structured', action='store_true',
                    help='use structured pruning for Mamba')
```

2. **模型识别** (行308, 359, 448, 497, 618, 648):
```python
if vit_pruning_utils.is_vit_model(model):
    # ViT特定逻辑
elif mamba_pruning_utils.is_mamba_model(model):
    # Mamba特定逻辑
else:
    # CNN默认逻辑
```

3. **剪枝执行** (行359):
```python
if is_mamba:
    mamba_pruning_utils.pruning_model_mamba(model, rate)
```

4. **Mask提取** (行448):
```python
if is_mamba:
    current_mask = mamba_pruning_utils.extract_mask_mamba(model)
```

---

## 🧬 Mamba模型架构分析

### 1. Mamba核心组件

**标准Mamba Block结构**:
```
Input
  ↓
LayerNorm
  ↓
SSM (Selective State Space Module)
  ├── Linear Projection (x → B, C, Δ)
  ├── Selective Scan (状态空间计算)
  └── Output Projection
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
MLP (Feed-Forward)
  ├── Linear1 (d_model → mlp_ratio * d_model)
  ├── GELU
  └── Linear2 (mlp_ratio * d_model → d_model)
  ↓
Residual Connection
```

### 2. 可剪枝组件对比

| 组件类型 | ViT | Mamba | 剪枝策略 |
|---------|-----|-------|---------|
| **Attention/SSM** | QKV Linear (头级别) | B/C/Δ Linear | **SSM通道级别** |
| **Attention Proj** | Linear (embed_dim → embed_dim) | Output Linear | **通道级别** |
| **MLP FC1** | Linear (dim → mlp_dim) | Linear (d_model → mlp_dim) | **神经元级别** |
| **MLP FC2** | Linear (mlp_dim → dim) | Linear (mlp_dim → d_model) | **神经元级别** |
| **Norm层** | LayerNorm | LayerNorm/RMSNorm | **不剪枝** |

### 3. Mamba特有考虑

#### 3.1 SSM模块的特殊性
- **状态矩阵**: A, B, C, Δ参数需要协同剪枝
- **卷积路径**: 某些实现有Conv1D分支，需单独处理
- **选择性机制**: 剪枝可能影响选择性门控

#### 3.2 与ViT的相似性
✅ **可复用的部分**:
- MLP模块结构几乎相同
- 残差连接处理方式类似
- LayerNorm不需要剪枝

⚠️ **需要特殊处理**:
- SSM替代了Attention（线性复杂度 vs 二次复杂度）
- 状态空间参数的依赖关系
- 可能的混合架构（Mamba + Attention）

---

## 🔧 适配策略

### 策略选择

**方案A**: 完全模仿ViT剪枝流程（推荐✅）
- **优点**: 代码复用度高，风险低，易于维护
- **缺点**: 可能未充分利用Mamba的特性
- **适用场景**: 快速验证、建立baseline

**方案B**: 定制化Mamba剪枝策略
- **优点**: 可能获得更好性能
- **缺点**: 开发周期长，风险高
- **适用场景**: 后续优化阶段

**当前采用**: 方案A（后续可迭代到方案B）

### 剪枝粒度

#### 1. 非结构化剪枝（Unstructured）
- **目标**: SSM的Linear层、MLP的Linear层
- **方法**: 全局L1剪枝
- **优先级**: ⭐⭐⭐ (必须实现)

#### 2. 结构化剪枝（Structured）
**SSM级别**:
- 剪枝SSM的输出通道（类比ViT的Head）
- 需要同时调整B、C、Δ的维度

**MLP神经元级别**:
- 剪枝MLP FC1的输出神经元
- 同步调整FC2的输入维度

**优先级**: ⭐⭐ (实验性功能)

---

## 📅 实施计划

### 阶段0: 准备工作（当前阶段）
**时间**: 1天  
**任务**:
- [x] 分析现有代码库
- [x] 制定适配方案
- [ ] **用户审核方案** ⬅️ 当前位置
- [ ] 确定Mamba模型来源（自己实现 vs 使用开源库）

**输出**: 本方案文档

---

### 阶段1: Mamba模型集成
**时间**: 1-2天  
**任务**:
1. **获取/实现Mamba模型**
   - 选项A: 使用 `mamba-ssm` 官方库
   - 选项B: 参考论文自己实现
   - 选项C: 使用 `transformers` 库的Mamba实现

2. **创建 `models/mamba.py`**
   ```python
   class MambaModel(nn.Module):
       def __init__(self, d_model, n_layers, num_classes, ...):
           ...
   
   def mamba_small(num_classes=100, pretrained=False):
       return MambaModel(d_model=192, n_layers=24, num_classes=num_classes)
   
   def mamba_base(num_classes=100, pretrained=False):
       return MambaModel(d_model=384, n_layers=24, num_classes=num_classes)
   ```

3. **在 `utils.py` 中注册**
   - 添加 `mamba_small`, `mamba_base` 等选项
   - 支持 `--arch mamba_small` 参数

4. **基础测试**
   - 测试前向传播
   - 测试参数数量
   - 测试CIFAR-10/100训练（无剪枝）

**输出**: 
- `models/mamba.py`
- 测试脚本 `test_mamba_model.py`
- 基线性能报告

**验收标准**:
- ✅ Mamba模型可以正常训练
- ✅ 在CIFAR-10达到合理精度（> 85%）
- ✅ 无内存泄漏或CUDA错误

---

### 阶段2: 剪枝工具开发
**时间**: 2-3天  
**任务**:

1. **创建 `mamba_pruning_utils.py`**
   
   **核心函数**:
   ```python
   def is_mamba_model(model):
       """判断是否是Mamba模型"""
       
   def pruning_model_mamba(model, px, prune_ssm=True):
       """非结构化L1剪枝"""
       # 收集可剪枝层
       # - SSM的Linear层
       # - MLP的FC层
       
   def prune_model_custom_mamba(model, mask_dict):
       """应用自定义mask"""
       
   def extract_mask_mamba(model):
       """提取当前mask"""
       
   def remove_prune_mamba(model):
       """移除剪枝钩子"""
       
   def check_sparsity_mamba(model):
       """检查稀疏度"""
   ```

2. **识别Mamba的可剪枝层**
   - 遍历模型，找到所有Linear层
   - 排除分类头（head/fc）
   - 排除位置编码等特殊层

3. **实现mask管理**
   - 与ViT保持一致的mask格式
   - 支持checkpoint保存/加载

4. **单元测试**
   ```python
   # test_mamba_pruning.py
   def test_pruning_functionality():
       model = mamba_small(num_classes=10)
       pruning_model_mamba(model, 0.5)
       sparsity = check_sparsity_mamba(model)
       assert abs(sparsity - 0.5) < 0.01
   ```

**输出**:
- `mamba_pruning_utils.py`
- `test_mamba_pruning.py`
- 单元测试报告

**验收标准**:
- ✅ 所有单元测试通过
- ✅ 剪枝后稀疏度符合预期
- ✅ mask提取/加载正常

---

### 阶段3: 主训练脚本集成
**时间**: 1-2天  
**任务**:

1. **修改 `main_imp_fillback.py`**
   
   **添加命令行参数**:
   ```python
   parser.add_argument('--mamba_pretrained', action='store_true')
   parser.add_argument('--mamba_structured', action='store_true')
   parser.add_argument('--mamba_ssm_prune_ratio', type=float, default=0.0)
   ```

2. **添加模型判断逻辑**
   
   在所有关键位置（6处）添加Mamba分支:
   ```python
   is_vit = vit_pruning_utils.is_vit_model(model)
   is_mamba = mamba_pruning_utils.is_mamba_model(model)
   
   if is_vit:
       # ViT逻辑
   elif is_mamba:
       # Mamba逻辑
   else:
       # CNN逻辑
   ```

3. **集成剪枝调用**
   ```python
   if is_mamba:
       if args.mamba_structured:
           mamba_pruning_utils.pruning_model_mamba_structured(
               model, rate, args.mamba_ssm_prune_ratio
           )
       else:
           mamba_pruning_utils.pruning_model_mamba(model, rate)
   ```

4. **处理checkpoint兼容性**
   - 确保mask保存/加载兼容
   - 预训练权重初始化验证

**输出**:
- 修改后的 `main_imp_fillback.py`
- 集成测试脚本

**验收标准**:
- ✅ 可以启动Mamba训练
- ✅ 剪枝流程正常执行
- ✅ checkpoint正常保存/加载
- ✅ 不影响ViT和CNN的训练

---

### 阶段4: 实验验证
**时间**: 2-3天  
**任务**:

1. **CIFAR-10基础实验**
   ```bash
   # 无剪枝baseline
   python main_imp_fillback.py --arch mamba_small --dataset cifar10 \
       --pruning_times 0 --epochs 160
   
   # 70%非结构化剪枝 + Refill
   python main_imp_fillback.py --arch mamba_small --dataset cifar10 \
       --rate 0.7 --pruning_times 16 --epochs 60 --fillback_rate 0.0
   
   # 70%非结构化剪枝 + RSST
   python main_imp_fillback.py --arch mamba_small --dataset cifar10 \
       --rate 0.7 --pruning_times 16 --epochs 60 \
       --reg_granularity_prune 1.0 --RST_schedule exp_custom_exponents \
       --exponents 4
   ```

2. **CIFAR-100验证实验**
   - 重复CIFAR-10的实验设置

3. **对比分析**
   - Mamba vs ViT (相同剪枝率)
   - RSST vs Refill (Mamba上的效果)
   - 不同剪枝率的性能曲线

4. **性能监控**
   - 训练时间
   - GPU显存占用
   - 推理速度（剪枝前后）

**输出**:
- 实验结果表格
- 性能曲线图
- 对比分析报告

**验收标准**:
- ✅ Mamba+RSST精度 > Mamba+Refill
- ✅ 70%稀疏度下精度下降 < 5%
- ✅ 训练过程稳定，无异常

---

### 阶段5: 结构化剪枝（可选）
**时间**: 3-4天  
**任务**:

1. **SSM通道级剪枝**
   - 实现SSM输出通道的mask
   - 动态调整B、C、Δ维度
   - 验证状态空间计算正确性

2. **MLP神经元级剪枝**
   - 类比ViT的MLP剪枝
   - FC1输出 → FC2输入的维度同步

3. **混合剪枝策略**
   - `--mamba_prune_target ssm`: 仅剪枝SSM
   - `--mamba_prune_target mlp`: 仅剪枝MLP
   - `--mamba_prune_target both`: 两者都剪枝

4. **实验验证**
   - 对比结构化 vs 非结构化
   - 测量实际加速比

**输出**:
- 结构化剪枝实现
- 加速测试报告
- 最佳实践文档

**验收标准**:
- ✅ 结构化剪枝后模型可正常运行
- ✅ 获得实际推理加速（> 1.5x）
- ✅ 精度损失可控

---

### 阶段6: 文档与清理
**时间**: 1天  
**任务**:

1. **代码清理**
   - 移除debug代码
   - 统一命名规范
   - 添加详细注释

2. **文档编写**
   ```markdown
   - Mamba_RSST使用指南.md
   - Mamba模型说明.md
   - Mamba剪枝API文档.md
   ```

3. **示例脚本**
   ```bash
   run_mamba_rsst.sh
   run_mamba_experiments.sh
   ```

4. **更新主README**
   - 添加Mamba支持说明
   - 更新模型列表
   - 添加citation

**输出**:
- 完整文档
- 示例脚本
- 更新的README

**验收标准**:
- ✅ 文档清晰易懂
- ✅ 新用户可以快速上手
- ✅ 所有示例可运行

---

## 🔬 技术细节与挑战

### 挑战1: Mamba模型来源

**选项分析**:

| 选项 | 优点 | 缺点 | 推荐度 |
|-----|------|------|--------|
| **mamba-ssm官方库** | 高质量实现、性能优化好 | 依赖CUDA kernels、可能不易修改 | ⭐⭐⭐⭐ |
| **transformers库** | 易集成、文档完善 | 可能缺少某些功能 | ⭐⭐⭐⭐ |
| **自己实现** | 完全可控、易于剪枝 | 开发成本高、可能有bug | ⭐⭐ |

**建议**: 优先使用 `mamba-ssm` 或 `transformers`，封装一层wrapper便于剪枝。

### 挑战2: SSM模块的剪枝

**问题**: SSM的B、C、Δ参数相互依赖

**解决方案**:
1. **非结构化剪枝**: 直接对Linear层权重剪枝（与ViT一致）
2. **结构化剪枝**: 需要协同调整多个参数矩阵的维度

```python
# 伪代码
def prune_ssm_channel(ssm_module, channel_mask):
    # channel_mask: [d_state] bool tensor
    ssm_module.B = ssm_module.B[:, channel_mask]
    ssm_module.C = ssm_module.C[channel_mask, :]
    ssm_module.delta_proj.weight = ssm_module.delta_proj.weight[channel_mask, :]
```

### 挑战3: 不同Mamba变体

**Mamba-1 vs Mamba-2**:
- Mamba-2引入了更多优化（SSD、分组等）
- 需要确保剪枝逻辑兼容不同版本

**应对**: 
- 从Mamba-1开始
- 预留扩展接口

### 挑战4: 预训练模型

**问题**: Mamba在CIFAR-10/100上没有官方预训练模型

**应对**:
1. 从随机初始化开始（与ViT无预训练模式一致）
2. 后续可自己在ImageNet上预训练
3. 或使用transfer learning

---

## ⚠️ 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|-----|------|------|----------|
| **Mamba库依赖冲突** | 中 | 高 | 创建独立conda环境，固定版本 |
| **CUDA kernel不兼容** | 低 | 高 | 使用纯PyTorch实现的Mamba |
| **剪枝后精度崩溃** | 中 | 中 | 从低剪枝率开始，逐步增加 |
| **内存溢出** | 低 | 中 | 减小batch size，使用gradient checkpointing |
| **训练不稳定** | 中 | 中 | 调整学习率、warmup、正则化 |
| **与现有代码冲突** | 低 | 高 | 充分测试，使用分支隔离 |

---

## ✅ 验收标准

### 功能性验收
- [ ] Mamba模型可以独立训练（无剪枝）
- [ ] 非结构化剪枝功能正常
- [ ] RSST正则化正常工作
- [ ] checkpoint保存/加载正常
- [ ] 不影响现有ViT/ResNet功能

### 性能验收
- [ ] CIFAR-10 baseline精度 > 85%
- [ ] 70%稀疏度精度下降 < 5%
- [ ] RSST优于Refill（至少+1%）
- [ ] 训练时间增加 < 20%

### 代码质量
- [ ] 所有单元测试通过
- [ ] 无linter错误
- [ ] 代码覆盖率 > 80%
- [ ] 文档完整清晰

### 可扩展性
- [ ] 易于添加新的Mamba变体
- [ ] 易于调整剪枝策略
- [ ] 易于集成到其他项目

---

## 📝 开发检查清单

### 阶段0: 准备
- [x] 分析现有代码库
- [x] 撰写方案文档
- [ ] 用户审核通过
- [ ] 确定Mamba来源
- [ ] 创建开发分支

### 阶段1: 模型集成
- [ ] 实现/集成Mamba模型
- [ ] 在utils.py注册
- [ ] 基础训练测试
- [ ] 性能baseline测试

### 阶段2: 剪枝工具
- [ ] 创建mamba_pruning_utils.py
- [ ] 实现is_mamba_model
- [ ] 实现pruning_model_mamba
- [ ] 实现extract_mask_mamba
- [ ] 实现remove_prune_mamba
- [ ] 单元测试

### 阶段3: 主脚本集成
- [ ] 添加命令行参数
- [ ] 集成模型判断逻辑
- [ ] 集成剪枝调用
- [ ] checkpoint兼容性测试

### 阶段4: 实验验证
- [ ] CIFAR-10 baseline
- [ ] CIFAR-10 + Refill
- [ ] CIFAR-10 + RSST
- [ ] CIFAR-100实验
- [ ] 性能对比分析

### 阶段5: 结构化剪枝（可选）
- [ ] SSM通道级剪枝
- [ ] MLP神经元级剪枝
- [ ] 混合剪枝策略
- [ ] 加速验证

### 阶段6: 收尾
- [ ] 代码清理
- [ ] 文档编写
- [ ] 示例脚本
- [ ] README更新
- [ ] 合并到主分支

---

## 🎓 参考资料

### 论文
1. **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. **RSST原论文**: Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets
3. **LTH**: The Lottery Ticket Hypothesis

### 代码参考
1. **mamba-ssm**: https://github.com/state-spaces/mamba
2. **transformers Mamba**: https://huggingface.co/docs/transformers/model_doc/mamba
3. **当前项目ViT实现**: `models/vit.py`, `vit_pruning_utils.py`

---

## 💬 待讨论问题

请审核以下问题并给出反馈：

1. **Mamba模型来源**: 您倾向于使用哪个Mamba实现？
   - [ ] mamba-ssm官方库
   - [ ] transformers库
   - [ ] 自己实现简化版
   - [ ] 其他: __________

2. **数据集选择**: 除了CIFAR-10/100，是否需要ImageNet实验？
   - [ ] 仅CIFAR即可
   - [ ] 需要ImageNet
   - [ ] 先CIFAR，后续考虑ImageNet

3. **优先级**: 非结构化 vs 结构化剪枝？
   - [ ] 优先非结构化（快速验证）
   - [ ] 优先结构化（实际加速）
   - [ ] 两者并行

4. **时间预期**: 整体开发周期？
   - [ ] 1周（最小可行版本）
   - [ ] 2周（完整功能）
   - [ ] 1个月（包括充分实验）

5. **其他需求**: 还有什么特殊要求或关注点？
   - ______________________

---

**请审核此方案，确认后我们开始实施！** 🚀
