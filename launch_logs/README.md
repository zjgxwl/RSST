# 实验启动日志目录

本目录用于存放实验启动时的日志文件，记录实验的启动参数、配置和初始输出。

## 目录说明

这里保存的是通过 `nohup` 或后台方式启动实验时产生的简短启动日志，主要包含：
- 实验配置确认信息
- 启动成功确认
- 进程ID信息
- 实验目录位置

## 与experiments目录的区别

| 目录 | 用途 | 内容 |
|------|------|------|
| `launch_logs/` | 启动日志 | 记录实验启动过程，文件较小（~1-2KB） |
| `experiments/` | 完整实验结果 | 包含训练日志、模型检查点、配置等（数GB） |

## 日志文件命名规范

建议使用以下命名格式：
```
<dataset>_<algorithm>_<timestamp>.log
```

例如：
- `cifar10_rsst_20260109.log`
- `cifar100_refill_20260109.log`

## 查看日志

```bash
# 查看所有启动日志
ls -lh launch_logs/

# 查看特定日志内容
cat launch_logs/cifar10_refill.log

# 实时监控最新日志
tail -f launch_logs/*.log
```

## 清理旧日志

```bash
# 清理30天前的启动日志
find launch_logs/ -name "*.log" -mtime +30 -delete
```

## 注意事项

- 这些日志文件记录的是启动过程，不是完整的训练日志
- 完整的训练日志位于 `experiments/<实验名称>/logs/` 目录中
- 定期清理旧日志以节省空间
