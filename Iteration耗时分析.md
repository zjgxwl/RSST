# 一个Iteration (State) 耗时分析

## 总体结构

```python
for state in range(start_state, args.pruning_times):  # 16次
    state_start_time = time.time()  # 开始计时
    
    ① 初始化准备          (~1秒)
    ② 训练循环 (60 epochs) (~70分钟)  ← 最耗时！98%
    ③ 剪枝操作            (~30秒-2分钟)
    
    state_end_time = time.time()  # 结束计时
```

**预计总耗时**: ~72分钟/State

---

## ① 初始化准备 (~1秒, <1%)

### 代码位置: 第325-338行

```python
# 检查稀疏度
if vit_pruning_utils.is_vit_model(model):
    remain_weight = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)

# RSST特有：初始化正则化剪枝器
if state > 0 and passer.args.struct == 'rsst':
    passer.reg_plot_init = 0
    passer.reg_plot = []
    pruner = reg_pruner.Pruner(model, args, passer)  # 初始化剪枝构造函数
```

**耗时**: 极少，可忽略

---

## ② 训练循环 (~70分钟, 98%)

### 代码位置: 第339-382行

```python
for epoch in range(start_epoch, args.epochs):  # 60个epoch
    
    # 2.1 训练一个epoch (~68秒, 95%)
    if passer.args.struct == 'rsst':
        acc = train(state, train_loader, model, criterion, optimizer, epoch, passer, pruner)
    else:
        acc = train(state, train_loader, model, criterion, optimizer, epoch, passer=None, pruner=None)
    
    # 2.2 验证集评估 (~2秒, 3%)
    tacc = validate(val_loader, model, criterion)
    
    # 2.3 测试集评估 (~2秒, 3%)
    test_tacc = validate(test_loader, model, criterion)
    
    # 2.4 保存checkpoint (~0.5秒, <1%)
    save_checkpoint({...}, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
```

**总耗时**: 60 epochs × 72秒 = ~72分钟

### 2.1 训练函数详解 (最耗时)

#### 代码位置: 第680-740行

```python
def train(state, train_loader, model, criterion, optimizer, epoch, passer, pruner):
    model.train()
    start = time.time()
    
    for i, (image, target) in enumerate(train_loader):  # 352个batch
        image = image.cuda()
        target = target.cuda()
        
        # A. 前向传播 (~0.15秒/batch, 60%)
        output_clean = model(image)
        loss = criterion(output_clean, target)
        
        # B. 反向传播 (~0.08秒/batch, 30%)
        optimizer.zero_grad()
        loss.backward()
        
        # C. RSST正则化 (仅RSST, ~0.02秒/batch, 10%)
        if state > 0 and args.struct == 'rsst':
            # 更新正则化参数
            if passer.args.reg_granularity_prune * i < 1:
                update_reg(passer, pruner, model, state, i, j)
            # 修改梯度，加入正则化项
            model = apply_reg(pruner, model, passer)
        
        # D. 更新权重 (~0.01秒/batch, <5%)
        optimizer.step()
```

**单个batch耗时**: ~0.19秒  
**352个batch总耗时**: 352 × 0.19 = ~67秒

**关键耗时点**:
- **前向传播** (60%): ViT-Small在CIFAR-10上的计算
- **反向传播** (30%): 梯度计算
- **RSST正则化** (10%, 仅RSST): `update_reg` + `apply_reg`

### 2.2 & 2.3 验证/测试函数

#### 代码位置: 第747-784行

```python
def validate(val_loader, model, criterion):
    model.eval()
    
    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()
            
            # 前向传播（无反向传播，更快）
            output = model(image)
            loss = criterion(output, target)
```

**耗时**: 
- 验证集: ~2秒 (79个batch × 0.025秒)
- 测试集: ~2秒 (79个batch × 0.025秒)

### 2.4 保存Checkpoint

```python
save_checkpoint({
    'state': state,
    'result': all_result,
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_sa': best_sa,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'init_weight': initialization
}, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
```

**耗时**: ~0.5秒 (磁盘I/O)

---

## ③ 剪枝操作 (~30秒-2分钟, 1-2%)

### 代码位置: 第463-590行

```python
# 3.1 非结构化剪枝 (~5秒)
vit_pruning_utils.pruning_model_vit(model, args.rate, prune_patch_embed=False)
remain_weight_after = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
current_mask = vit_pruning_utils.extract_mask_vit(model.state_dict())
vit_pruning_utils.remove_prune_vit(model, prune_patch_embed=False)

# 3.2 保存训练后权重
train_weight = model.state_dict()  # ~0.1秒

# 3.3 加载初始权重
model.load_state_dict(initialization)  # ~0.1秒

# 3.4 准结构化剪枝（Refill或RSST）
if args.struct == 'refill':
    # Refill: 直接剪枝并填充
    model = vit_pruning_utils_head_mlp.prune_model_custom_fillback_vit_head_and_mlp(
        model, mask_dict=current_mask, train_loader=train_loader,
        trained_weight=train_weight, init_weight=initialization,
        criteria=args.criteria, head_prune_ratio=args.rate,
        mlp_prune_ratio=mlp_ratio, return_mask_only=False,
        sorting_mode=args.sorting_mode)
    # 耗时: ~30秒-1分钟（需要计算importance scores）

elif args.struct == 'rsst':
    # RSST: 只计算mask，不剪枝
    mask = vit_pruning_utils_head_mlp.prune_model_custom_fillback_vit_head_and_mlp(
        model, mask_dict=current_mask, train_loader=train_loader,
        trained_weight=train_weight, init_weight=initialization,
        criteria=args.criteria, head_prune_ratio=args.rate,
        mlp_prune_ratio=mlp_ratio, return_mask_only=True,
        sorting_mode=args.sorting_mode)
    passer.refill_mask = mask
    # 耗时: ~30秒-1分钟（需要计算importance scores）

# 3.5 RSST特有：正则化剪枝
if state > 0 and passer.args.struct == 'rsst':
    # 遍历所有层，应用mask
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and is_vit:
            if 'attn' in name or 'mlp' in name:
                mask = passer.current_mask[name + '.weight_mask']
                m.weight.data = initialization[name + ".weight"]
                prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
    # 耗时: ~5-10秒
```

**总耗时**: 
- Refill: ~40秒-1.5分钟
- RSST: ~40秒-1.5分钟

**关键耗时点**:
- `prune_model_custom_fillback_vit_head_and_mlp`: 计算importance scores (30-60秒)
  - 需要对每个head和MLP neuron计算重要性
  - Global sorting需要收集和排序所有层的scores

---

## 总结：耗时占比

```
训练循环 (60 epochs):  70.5分钟 (98.0%)
  ├─ 训练 (352 batch/epoch):  67秒/epoch  (93%)
  │   ├─ 前向传播:            40秒 (60%)
  │   ├─ 反向传播:            20秒 (30%)
  │   └─ RSST正则化:          7秒  (10%)
  ├─ 验证集:                  2秒/epoch   (3%)
  ├─ 测试集:                  2秒/epoch   (3%)
  └─ Checkpoint:              0.5秒/epoch (<1%)

剪枝操作:                    1.5分钟    (2.0%)
  ├─ 非结构化剪枝:            5秒        (0.1%)
  ├─ 准结构化剪枝:            60秒       (1.4%)
  └─ RSST正则化剪枝:          10秒       (0.2%)

初始化:                      1秒        (<0.1%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计:                        72分钟/State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 优化建议

### 已完成的优化 ✅
1. ✅ 注释掉 `wandb.log` (节省网络请求时间)
2. ✅ 注释掉 `plt.savefig` (节省绘图和I/O时间)
3. ✅ 使用 `python -u` (减少输出缓冲)

### 可能的进一步优化

1. **减少验证频率** (可节省2-3%)
   - 当前每个epoch都验证+测试
   - 可改为每5-10个epoch验证一次
   - 节省时间: ~1.4分钟/State

2. **减少Checkpoint保存频率** (可节省<1%)
   - 当前每个epoch都保存
   - 可改为只保存best checkpoint
   - 节省时间: ~30秒/State

3. **混合精度训练** (可节省20-30%)
   - 使用 `torch.cuda.amp`
   - 可加速前向+反向传播
   - 节省时间: ~15分钟/State
   - ⚠️ 需要修改代码，可能影响准确率

4. **数据加载优化** (可节省5-10%)
   - 增加 `num_workers`
   - 使用 `pin_memory=True`
   - 节省时间: ~4-7分钟/State

5. **减少剪枝计算** (可节省1-2%)
   - 使用更快的importance计算方法
   - 缓存部分计算结果
   - 节省时间: ~30秒-1分钟/State

---

## 关键代码段总结

**最耗时的3个函数**:

1. **`train()` 函数** - 67秒/epoch × 60 = 67分钟 (93%)
2. **`prune_model_custom_fillback_vit_head_and_mlp()` - 60秒 (1.4%)
3. **`validate()` 函数** - 4秒/epoch × 60 = 4分钟 (5.6%)

**结论**: 训练循环占据了98%的时间，优化重点应该在训练效率上！
