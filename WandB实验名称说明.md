# WandB实验名称配置说明

## 📝 概述

现在支持**灵活配置WandB实验名称**，既可以自定义，也可以自动生成有意义的名称。

---

## 🎯 使用方法

### 方法1：自动生成（默认）⭐推荐

**不指定`--exp_name`**，系统自动生成包含关键信息的名称：

```bash
python main_imp_fillback.py \
    --dataset cifar100 \
    --arch vit_small \
    --struct rsst \
    --pretrained
```

**生成的名称示例：**
```
rsst_vit_small_cifar100_sched_exp_custom_exponents_reg_0.5_exp3_crit_l1_rate_0.15_pretrained_0108_1430
```

**包含的信息：**
- `rsst`: 剪枝方法
- `vit_small`: 模型架构
- `cifar100`: 数据集
- `sched_exp_custom_exponents`: 正则化schedule
- `reg_0.5`: 正则化粒度
- `exp3`: 指数参数
- `crit_l1`: 重要性评估标准
- `rate_0.15`: 剪枝率
- `pretrained`: 使用预训练
- `0108_1430`: 时间戳（月日_时分）

---

### 方法2：自定义名称

**指定`--exp_name`**，使用完全自定义的名称：

```bash
python main_imp_fillback.py \
    --dataset cifar100 \
    --arch vit_small \
    --exp_name "my_vit_experiment_v1"
```

**WandB名称：**
```
my_vit_experiment_v1
```

---

### 方法3：半自定义（推荐用于系列实验）

使用有意义的前缀 + 自动时间戳：

```bash
# 实验1
python main_imp_fillback.py \
    --exp_name "vit_ablation_lr0.001" \
    --lr 0.001

# 实验2
python main_imp_fillback.py \
    --exp_name "vit_ablation_lr0.0005" \
    --lr 0.0005
```

---

## 📊 自动生成规则

### RSST算法

```
格式: rsst_{arch}_{dataset}_sched_{schedule}_reg_{reg_value}_exp{exponents}_crit_{criteria}_rate_{rate}_{pretrained}_{timestamp}

示例: rsst_vit_tiny_cifar10_sched_exp_custom_exponents_reg_0.5_exp3_crit_l1_rate_0.15_pretrained_0108_1430
```

### Refill算法

```
格式: refill_{arch}_{dataset}_fill_{fillback_rate}_crit_{criteria}_rate_{rate}_{pretrained}_{timestamp}

示例: refill_vit_small_cifar100_fill_0.1_crit_magnitude_rate_0.2_pretrained_0108_1520
```

### CNN模型（不使用预训练）

```
格式: rsst_{arch}_{dataset}_sched_{schedule}_reg_{reg_value}_crit_{criteria}_rate_{rate}_{timestamp}

示例: rsst_res20s_cifar100_sched_exp_custom_exponents_reg_1.0_crit_l1_rate_0.2_0108_1600
```

---

## 💡 命名建议

### 场景1：对比实验

```bash
# 对比不同模型
--exp_name "vit_tiny_rsst_baseline"
--exp_name "vit_small_rsst_baseline"
--exp_name "vit_base_rsst_baseline"

# 对比不同剪枝率
--exp_name "rsst_rate0.15"
--exp_name "rsst_rate0.20"
--exp_name "rsst_rate0.25"
```

### 场景2：消融实验

```bash
# 测试不同schedule
--exp_name "ablation_schedule_linear"    --RST_schedule x
--exp_name "ablation_schedule_exp2"      --RST_schedule exp_custom_exponents --exponents 2
--exp_name "ablation_schedule_exp4"      --RST_schedule exp_custom_exponents --exponents 4

# 测试不同criteria
--exp_name "ablation_criteria_magnitude" --criteria magnitude
--exp_name "ablation_criteria_l1"        --criteria l1
--exp_name "ablation_criteria_saliency"  --criteria saliency
```

### 场景3：复现实验

```bash
# 方便后续查找和复现
--exp_name "paper_fig3_vit_cifar10"
--exp_name "paper_table2_rsst_vs_refill"
--exp_name "reproduce_baseline_v2"
```

---

## 🔍 查看实验名称

### 在命令行查看

运行时会打印：

```
WandB实验名称: rsst_vit_small_cifar100_sched_exp_custom_exponents_reg_0.5_exp3_crit_l1_rate_0.15_pretrained_0108_1430

Run data is saved locally in /path/to/wandb/run-xxx
View run at: https://wandb.ai/ycx/RSST/runs/xxx
```

### 在WandB网页查看

访问: https://wandb.ai/ycx/RSST

可以看到所有实验，按名称排序和筛选。

---

## 📝 完整示例

### 示例1：使用自动生成（推荐日常使用）

```bash
python main_imp_fillback.py \
    --dataset cifar100 \
    --arch vit_small \
    --pretrained \
    --struct rsst \
    --criteria l1 \
    --epochs 80 \
    --pruning_times 15 \
    --rate 0.15 \
    --RST_schedule exp_custom_exponents \
    --reg_granularity_prune 0.5 \
    --exponents 3
    
# 自动生成名称:
# rsst_vit_small_cifar100_sched_exp_custom_exponents_reg_0.5_exp3_crit_l1_rate_0.15_pretrained_0108_1430
```

### 示例2：使用自定义名称（推荐重要实验）

```bash
python main_imp_fillback.py \
    --dataset cifar100 \
    --arch vit_small \
    --pretrained \
    --struct rsst \
    --exp_name "final_vit_small_cifar100_best_config" \
    --epochs 120 \
    --pruning_times 15
    
# 使用名称:
# final_vit_small_cifar100_best_config
```

### 示例3：系列对比实验

```bash
# 实验组1: 不同剪枝率
for rate in 0.10 0.15 0.20; do
    python main_imp_fillback.py \
        --dataset cifar10 \
        --arch vit_tiny \
        --pretrained \
        --struct rsst \
        --rate $rate \
        --exp_name "series1_rate_${rate}"
done

# WandB中显示为:
# series1_rate_0.10
# series1_rate_0.15
# series1_rate_0.20
```

---

## 🎨 自定义生成逻辑

如果想修改自动生成的格式，编辑 `main_imp_fillback.py` 第94-128行：

```python
# 基础信息
name_parts = [args.struct, args.arch, args.dataset]

# 添加你想要的信息
name_parts.append(f"bs_{args.batch_size}")      # 添加batch size
name_parts.append(f"lr_{args.lr}")              # 添加学习率
name_parts.append(f"seed_{args.seed}")          # 添加随机种子

# 修改时间戳格式
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 更详细

wdb_name = '_'.join(name_parts)
```

---

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|-----|--------|------|
| `--exp_name` | str | None | 自定义实验名称。如果不指定，自动生成 |

### 使用技巧

```bash
# ✅ 推荐：让系统自动生成（包含所有关键信息）
python main_imp_fillback.py --arch vit_small

# ✅ 推荐：重要实验用有意义的名称
python main_imp_fillback.py --exp_name "paper_final_results"

# ✅ 推荐：系列实验用统一前缀
python main_imp_fillback.py --exp_name "ablation_exp2" --exponents 2

# ❌ 不推荐：名称太简单，难以区分
python main_imp_fillback.py --exp_name "test1"

# ❌ 不推荐：名称太长，难以阅读
python main_imp_fillback.py --exp_name "vit_small_cifar100_rsst_with_pretrained_imagenet_weights_exp3"
```

---

## 🔄 迁移旧实验

如果之前运行的实验名称格式是：

```
old_format: rsst_exp_custom_exponents_l1_vit_small_cifar100
```

新格式会是：

```
new_format: rsst_vit_small_cifar100_sched_exp_custom_exponents_reg_0.5_exp3_crit_l1_rate_0.15_pretrained_0108_1430
```

**建议：**
- 新实验使用新格式（更详细）
- 旧实验保持不变（向后兼容）
- 重要实验可以用`--exp_name`指定统一命名

---

## 📚 相关文档

- **WandB官方文档**: https://docs.wandb.ai/
- **实验追踪最佳实践**: https://wandb.ai/site/experiment-tracking

---

**文档版本：** v1.0  
**更新日期：** 2026-01-08  
**作者：** AI Assistant

🎉 现在您可以更灵活地管理实验名称了！

