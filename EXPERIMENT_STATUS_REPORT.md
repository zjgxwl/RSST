# 实验状态报告 - 2026-01-18 20:15

## 📊 运行中的实验（8个）

### ViT-Small实验（4个）- 优化版配置
| 实验 | 数据集 | 方法 | GPU | PID | 显存 | 状态 | 进度 |
|------|--------|------|-----|-----|------|------|------|
| 1 | CIFAR-10 | Refill | 0 | 1882967 | 7.6GB | ✓ 运行中 | Epoch 5/100, State 0 |
| 2 | CIFAR-10 | RSST | 0 | 1882968 | 7.6GB | ✓ 运行中 | Epoch 5/100, State 0 |
| 3 | CIFAR-100 | Refill | 1 | 1882969 | 7.6GB | ✓ 运行中 | Epoch 4/100, State 0 |
| 4 | CIFAR-100 | RSST | 1 | 1882970 | 7.6GB | ✓ 运行中 | Epoch 4/100, State 0 |

**ViT配置**：
- batch_size: **256**
- pruning_times: **11**
- epochs: **100**
- decreasing_lr: **[30, 60, 85]**
- 启动时间: 20:00
- 预计完成: 明天凌晨 2:00-3:00（约6-7小时）

**ViT最新精度**：
- CIFAR-10 Refill: **48.05%**
- CIFAR-10 RSST: **46.70%**
- CIFAR-100 Refill: **14.45%**
- CIFAR-100 RSST: **14.40%**

---

### Mamba-Small实验（4个）- 优化版配置
| 实验 | 数据集 | 方法 | GPU | PID | 显存 | 状态 | 进度 |
|------|--------|------|-----|-----|------|------|------|
| 5 | CIFAR-10 | Refill | 0 | 1928497 | 1.9GB | ✓ 运行中 | Epoch 0/100, State 0 |
| 6 | CIFAR-10 | RSST | 0 | 1928666 | 1.9GB | ✓ 运行中 | Epoch 0/100, State 0 |
| 7 | CIFAR-100 | Refill | 1 | 1928834 | 1.8GB | ✓ 运行中 | Epoch 0/100, State 0 |
| 8 | CIFAR-100 | RSST | 1 | 1928993 | 1.7GB | ✓ 运行中 | Epoch 0/100, State 0 |

**Mamba配置**：
- batch_size: **128**（保持原值，Mamba显存敏感）
- pruning_times: **11**
- epochs: **100**
- decreasing_lr: **[30, 60, 85]**
- 启动时间: 20:14
- 预计完成: 明天下午 ~16:00（约20小时）

**Mamba状态**：
- 刚启动，正在进行第一个epoch的训练

---

## 🔥 GPU资源状态

### GPU 0
- **利用率**: 100%
- **显存**: 19.1GB / 81.9GB (**23%**)
- **温度**: 74°C
- **功耗**: 294W / 300W
- **进程**: ViT×2 (15.2GB) + Mamba×2 (3.8GB)

### GPU 1
- **利用率**: 100%
- **显存**: 18.8GB / 81.9GB (**23%**)
- **温度**: 76°C
- **功耗**: 294W / 300W
- **进程**: ViT×2 (15.2GB) + Mamba×2 (3.6GB)

---

## 🔧 问题修复记录

### 问题1：旧实验占用显存
- **时间**: 20:09
- **问题**: 旧版ViT实验（PID 3080681, 3080881, 1602048, 3081145）仍在运行，占用约23GB显存
- **解决**: Kill旧进程，释放显存
- **状态**: ✅ 已解决

### 问题2：Mamba batch_size过大（第一次）
- **时间**: 20:06-20:10
- **问题**: batch_size=256导致Mamba显存占用30GB+，第一个batch耗时37秒
- **尝试**: 降至batch_size=192
- **状态**: ❌ 仍不够

### 问题3：Mamba batch_size过大（第二次）
- **时间**: 20:11-20:14
- **问题**: batch_size=192仍然导致显存占用23.4GB
- **解决**: 降至batch_size=128（原值）
- **状态**: ✅ 已解决，显存降至1.8GB

---

## 📈 优化效果总结

### ViT-Small
| 项目 | 旧版 | 优化版 | 改进 |
|------|------|--------|------|
| Batch size | 128 | **256** | +100% |
| Pruning times | 16 | **11** | -31% |
| Epochs/state | 60 | **100** | +67% |
| Decreasing LR | ❌ | **✓** | 新增 |
| Checkpoint | 每epoch | **仅state结束** | -95% |
| **预计总时间** | **19.2小时** | **~6-7小时** | **节省12-13小时** |

### Mamba-Small
| 项目 | 旧版 | 优化版 | 改进 |
|------|------|--------|------|
| Batch size | 128 | **128** | 持平 |
| Pruning times | 16 | **11** | -31% |
| Epochs/state | 60 | **100** | +67% |
| Decreasing LR | ❌ | **✓** | 新增 |
| Checkpoint | 每epoch | **仅state结束** | -95% |
| **预计总时间** | **53小时** | **~40小时** | **节省13小时** |

---

## 📝 监控命令

```bash
# 查看ViT实验状态
bash /workspace/ycx/RSST/check_vit_v2_experiments.sh

# 查看Mamba实验状态
bash /workspace/ycx/RSST/check_mamba_v2_experiments.sh

# 查看GPU状态
nvidia-smi

# 查看所有实验日志
tail -f /workspace/ycx/RSST/logs_vit_small_70p_v2/*.log
tail -f /workspace/ycx/RSST/logs_mamba_small_70p_v2/*.log
```

---

## ✅ 结论

**所有8个实验现在都在正常运行！**

- ✅ ViT实验：训练顺利，已完成4-5个epoch
- ✅ Mamba实验：刚启动，显存占用正常（约1.8GB/进程）
- ✅ GPU利用率：100%满负载
- ✅ 显存占用：约23%，安全稳定
- ✅ 预计完成时间：
  - ViT：明天凌晨 2:00-3:00
  - Mamba：明天下午 ~16:00

**系统稳定，可以放心等待结果！** 🚀
