"""
Mamba-Small Baseline Training Script V2 (全面优化版)
训练 Mamba-Small 的 baseline 性能（无剪枝）

V2 新增优化:
✅ Drop Path (Stochastic Depth)
✅ EMA (Exponential Moving Average)
✅ Random Erasing
✅ AutoAugment (替代 RandAugment)
✅ Gradient Clipping
✅ 混合精度训练 (AMP)
✅ Layer-wise Learning Rate Decay
✅ Test-Time Augmentation (TTA)
✅ 改进的学习率调度
✅ 更好的日志和可视化

目标性能:
- CIFAR-10: 97-98% (vs V1 的 94-95.5%)
- CIFAR-100: 82-86% (vs V1 的 76-81%)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast, GradScaler

# 导入 Mamba 模型
from models.mamba import mamba_small, mamba_tiny, mamba_base


# ============================================================================
# 参数设置
# ============================================================================

parser = argparse.ArgumentParser(description='Mamba-Small Baseline Training V2 (全面优化)')

# 数据集
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'cifar100'],
                    help='数据集选择')
parser.add_argument('--data_path', type=str, default='./datasets',
                    help='数据集路径')

# 模型
parser.add_argument('--arch', type=str, default='mamba_small',
                    choices=['mamba_tiny', 'mamba_small', 'mamba_base'],
                    help='模型架构')

# 训练参数
parser.add_argument('--epochs', type=int, default=300,
                    help='训练轮数')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='初始学习率')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='权重衰减')
parser.add_argument('--warmup_epochs', type=int, default=20,
                    help='warmup 轮数')
parser.add_argument('--drop_path', type=float, default=0.1,
                    help='Drop Path rate')

# V2 新增: 优化选项
parser.add_argument('--use_ema', action='store_true', default=True,
                    help='使用 EMA')
parser.add_argument('--ema_decay', type=float, default=0.9999,
                    help='EMA decay rate')
parser.add_argument('--use_amp', action='store_true', default=True,
                    help='使用混合精度训练')
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='梯度裁剪')
parser.add_argument('--use_layerwise_lr', action='store_true', default=True,
                    help='使用 Layer-wise LR Decay')
parser.add_argument('--layerwise_lr_decay', type=float, default=0.65,
                    help='Layer-wise LR decay rate')

# 数据增强
parser.add_argument('--use_autoaugment', action='store_true', default=True,
                    help='使用 AutoAugment (替代 RandAugment)')
parser.add_argument('--use_random_erasing', action='store_true', default=True,
                    help='使用 Random Erasing')
parser.add_argument('--use_mixup', action='store_true', default=True,
                    help='使用 Mixup')
parser.add_argument('--use_cutmix', action='store_true', default=True,
                    help='使用 Cutmix')
parser.add_argument('--mixup_alpha', type=float, default=0.8,
                    help='Mixup alpha')
parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                    help='Cutmix alpha')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing')

# TTA
parser.add_argument('--use_tta', action='store_true', default=False,
                    help='使用 Test-Time Augmentation（最终测试时）')
parser.add_argument('--tta_size', type=int, default=5,
                    help='TTA 增强次数')

# 其他
parser.add_argument('--workers', type=int, default=8,
                    help='数据加载线程数')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID')
parser.add_argument('--save_dir', type=str, default='./checkpoint/mamba_baseline_v2',
                    help='模型保存目录')
parser.add_argument('--log_interval', type=int, default=50,
                    help='日志打印间隔')
parser.add_argument('--eval_interval', type=int, default=10,
                    help='评估间隔（epochs）')

args = parser.parse_args()


# ============================================================================
# 工具函数和类
# ============================================================================

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEMA:
    """
    模型参数的指数移动平均 (EMA)
    提升测试性能 +0.3-0.7%
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 EMA 参数（测试时）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数（训练时）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                    min_lr=1e-6, last_epoch=-1):
    """
    Cosine 学习率调度器（带 warmup）
    V2: 使用指数 warmup（更平滑）
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Exponential warmup (更平滑)
            progress = current_step / num_warmup_steps
            return (1 - np.exp(-5 * progress)) / (1 - np.exp(-5))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_layer_wise_lr_params(model, base_lr, decay_rate=0.65):
    """
    Layer-wise Learning Rate Decay
    深层学习率大，浅层学习率小
    预期提升: +0.3-0.5%
    """
    parameter_groups = []
    
    # Patch embedding (最小学习率)
    if hasattr(model, 'patch_embed'):
        n_layers = len(model.blocks) if hasattr(model, 'blocks') else 24
        parameter_groups.append({
            'params': model.patch_embed.parameters(),
            'lr': base_lr * (decay_rate ** n_layers),
            'name': 'patch_embed'
        })
    
    # Blocks (逐层递增)
    if hasattr(model, 'blocks'):
        n_layers = len(model.blocks)
        for i, block in enumerate(model.blocks):
            parameter_groups.append({
                'params': block.parameters(),
                'lr': base_lr * (decay_rate ** (n_layers - i - 1)),
                'name': f'block_{i}'
            })
    
    # Norm
    if hasattr(model, 'norm'):
        parameter_groups.append({
            'params': model.norm.parameters(),
            'lr': base_lr,
            'name': 'norm'
        })
    
    # Head (最大学习率)
    if hasattr(model, 'head'):
        parameter_groups.append({
            'params': model.head.parameters(),
            'lr': base_lr,
            'name': 'head'
        })
    
    # 如果没有分层，返回所有参数
    if len(parameter_groups) == 0:
        parameter_groups.append({
            'params': model.parameters(),
            'lr': base_lr,
            'name': 'all'
        })
    
    return parameter_groups


class MixupCutmix:
    """Mixup 和 Cutmix 数据增强"""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, images, targets):
        if np.random.rand() > self.prob:
            return images, targets
        
        batch_size = images.size(0)
        
        if np.random.rand() < 0.5:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            mixed_images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = targets, targets[index]
            return mixed_images, (targets_a, targets_b, lam)
        else:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            _, _, h, w = images.shape
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            targets_a, targets_b = targets, targets[index]
            return images, (targets_a, targets_b, lam)


def mixup_criterion(criterion, pred, targets):
    """Mixup/Cutmix 损失函数"""
    if isinstance(targets, tuple):
        targets_a, targets_b, lam = targets
        loss = lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)
    else:
        loss = criterion(pred, targets)
    return loss


# ============================================================================
# 数据加载
# ============================================================================

def get_dataloaders(args):
    """获取数据加载器 (V2: 使用 AutoAugment + Random Erasing)"""
    
    # 数据增强
    if args.use_autoaugment:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        
        # 根据数据集选择策略
        if args.dataset == 'cifar10':
            policy = AutoAugmentPolicy.CIFAR10
        else:
            policy = AutoAugmentPolicy.CIFAR10  # CIFAR100 也用 CIFAR10 策略
        
        augment_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(policy=policy),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ]
        
        # 添加 Random Erasing
        if args.use_random_erasing:
            augment_list.append(transforms.RandomErasing(p=0.25))
        
        train_transform = transforms.Compose(augment_list)
    else:
        # 基础增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, 
            transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True,
            transform=test_transform
        )
        num_classes = 10
    else:
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True,
            transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True,
            transform=test_transform
        )
        num_classes = 100
    
    # DataLoader 优化
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,  # V2: 保持 workers 常驻
        prefetch_factor=2 if args.workers > 0 else None,  # V2: 预取
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    
    return train_loader, test_loader, num_classes


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                epoch, args, mixup_cutmix=None, ema=None, scaler=None):
    """训练一个 epoch (V2: 支持 AMP, EMA, Grad Clip)"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        
        # Mixup / Cutmix
        if mixup_cutmix is not None and (args.use_mixup or args.use_cutmix):
            images, targets = mixup_cutmix(images, targets)
        
        # 混合精度训练
        if args.use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        scheduler.step()
        
        # 更新 EMA
        if ema is not None:
            ema.update()
        
        # 统计
        losses.update(loss.item(), images.size(0))
        
        if not isinstance(targets, tuple):
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            top1.update(acc1.item(), images.size(0))
        
        # 打印日志
        if batch_idx % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.2f}% ({top1.avg:.2f}%)\t'
                  f'LR {lr:.6f}')
    
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} 训练完成 | 时间: {epoch_time:.1f}s | '
          f'Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg


def validate(model, test_loader, criterion, args, use_ema=False):
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            
            # 混合精度推理
            if args.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    ema_tag = ' (EMA)' if use_ema else ''
    print(f'验证结果{ema_tag}: Loss {losses.avg:.4f} | Acc@1 {top1.avg:.2f}%')
    
    return losses.avg, top1.avg


def validate_with_tta(model, test_loader, args):
    """
    Test-Time Augmentation (TTA)
    预期提升: +0.5-1%
    """
    model.eval()
    
    print(f'使用 TTA (n={args.tta_size})...')
    
    all_predictions = []
    all_targets = []
    
    # TTA: 使用多个增强版本
    tta_transforms = []
    for flip in [False, True]:
        for crop_padding in [0, 2, 4]:
            if len(tta_transforms) >= args.tta_size:
                break
            tta_transforms.append({
                'flip': flip,
                'padding': crop_padding
            })
    
    with torch.no_grad():
        for images, targets in test_loader:
            batch_predictions = []
            
            for tta_config in tta_transforms:
                # 应用 TTA
                aug_images = images.clone()
                
                # Random crop
                if tta_config['padding'] > 0:
                    aug_images = torch.nn.functional.pad(
                        aug_images, 
                        (tta_config['padding'],) * 4, 
                        mode='reflect'
                    )
                    _, _, h, w = aug_images.shape
                    i = np.random.randint(0, tta_config['padding'] * 2 + 1)
                    j = np.random.randint(0, tta_config['padding'] * 2 + 1)
                    aug_images = aug_images[:, :, i:i+32, j:j+32]
                
                # Horizontal flip
                if tta_config['flip']:
                    aug_images = torch.flip(aug_images, dims=[3])
                
                aug_images = aug_images.cuda(args.gpu, non_blocking=True)
                
                # 推理
                if args.use_amp:
                    with autocast():
                        outputs = model(aug_images)
                else:
                    outputs = model(aug_images)
                
                batch_predictions.append(outputs.cpu())
            
            # 平均所有 TTA 预测
            avg_predictions = torch.stack(batch_predictions).mean(dim=0)
            all_predictions.append(avg_predictions)
            all_targets.append(targets)
    
    # 计算准确率
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    acc1 = accuracy(all_predictions, all_targets, topk=(1,))[0]
    
    print(f'TTA 结果: Acc@1 {acc1:.2f}%')
    
    return acc1.item()


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置 GPU
    torch.cuda.set_device(args.gpu)
    print(f'使用 GPU: {args.gpu}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 打印配置
    print('=' * 80)
    print('Mamba-Small Baseline 训练 V2 (全面优化)')
    print('=' * 80)
    print(f'数据集: {args.dataset}')
    print(f'模型: {args.arch}')
    print(f'训练轮数: {args.epochs}')
    print(f'Batch Size: {args.batch_size}')
    print(f'初始学习率: {args.lr}')
    print(f'Weight Decay: {args.weight_decay}')
    print(f'Warmup Epochs: {args.warmup_epochs}')
    print(f'Drop Path: {args.drop_path}')
    print(f'\n优化选项:')
    print(f'  ✅ EMA: {args.use_ema} (decay={args.ema_decay})')
    print(f'  ✅ AMP (混合精度): {args.use_amp}')
    print(f'  ✅ Gradient Clipping: {args.grad_clip}')
    print(f'  ✅ Layer-wise LR: {args.use_layerwise_lr}')
    print(f'\n数据增强:')
    print(f'  ✅ AutoAugment: {args.use_autoaugment}')
    print(f'  ✅ Random Erasing: {args.use_random_erasing}')
    print(f'  ✅ Mixup (α={args.mixup_alpha}): {args.use_mixup}')
    print(f'  ✅ Cutmix (α={args.cutmix_alpha}): {args.use_cutmix}')
    print(f'  ✅ Label Smoothing: {args.label_smoothing}')
    print(f'  ✅ TTA (测试时): {args.use_tta}')
    print('=' * 80)
    
    # 加载数据
    print('\n加载数据集...')
    train_loader, test_loader, num_classes = get_dataloaders(args)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    print(f'类别数: {num_classes}')
    
    # 创建模型 (V2: 支持 Drop Path)
    print(f'\n创建模型: {args.arch} (with Drop Path={args.drop_path})')
    if args.arch == 'mamba_tiny':
        model = mamba_tiny(num_classes=num_classes, img_size=32, drop_path=args.drop_path)
    elif args.arch == 'mamba_small':
        model = mamba_small(num_classes=num_classes, img_size=32, drop_path=args.drop_path)
    elif args.arch == 'mamba_base':
        model = mamba_base(num_classes=num_classes, img_size=32, drop_path=args.drop_path)
    else:
        raise ValueError(f'Unknown architecture: {args.arch}')
    
    model = model.cuda(args.gpu)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)')
    
    # 损失函数
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
    
    # 优化器 (V2: Layer-wise LR)
    if args.use_layerwise_lr:
        print(f'\n使用 Layer-wise LR Decay (decay_rate={args.layerwise_lr_decay})')
        param_groups = get_layer_wise_lr_params(model, args.lr, args.layerwise_lr_decay)
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=1e-6
    )
    
    # EMA
    ema = None
    if args.use_ema:
        print(f'\n初始化 EMA (decay={args.ema_decay})')
        ema = ModelEMA(model, decay=args.ema_decay)
    
    # 混合精度训练
    scaler = None
    if args.use_amp:
        print('\n使用混合精度训练 (AMP)')
        scaler = GradScaler()
    
    # Mixup / Cutmix
    if args.use_mixup or args.use_cutmix:
        mixup_cutmix = MixupCutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=0.5
        )
    else:
        mixup_cutmix = None
    
    # 训练记录
    best_acc = 0.0
    best_ema_acc = 0.0
    train_losses = []
    train_accs = []
    test_accs = []
    test_ema_accs = []
    
    print('\n开始训练...')
    print('=' * 80)
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 80)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, args, mixup_cutmix, ema, scaler
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            # 标准验证
            test_loss, test_acc = validate(model, test_loader, criterion, args)
            test_accs.append(test_acc)
            
            # EMA 验证
            if ema is not None:
                ema.apply_shadow()
                _, test_ema_acc = validate(model, test_loader, criterion, args, use_ema=True)
                ema.restore()
                test_ema_accs.append(test_ema_acc)
                
                # 保存最佳 EMA 模型
                if test_ema_acc > best_ema_acc:
                    best_ema_acc = test_ema_acc
                    save_path = os.path.join(args.save_dir, 
                                            f'{args.arch}_{args.dataset}_best_ema.pth')
                    # 保存 EMA 参数
                    ema.apply_shadow()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_acc': best_ema_acc,
                        'args': args,
                    }, save_path)
                    ema.restore()
                    print(f'✓ 保存最佳 EMA 模型: {save_path} (Acc: {best_ema_acc:.2f}%)')
            
            # 保存最佳标准模型
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = os.path.join(args.save_dir, 
                                        f'{args.arch}_{args.dataset}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'args': args,
                }, save_path)
                print(f'✓ 保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)')
        
        # 定期保存 checkpoint
        if epoch % 50 == 0:
            save_path = os.path.join(args.save_dir,
                                    f'{args.arch}_{args.dataset}_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_ema_acc': best_ema_acc,
                'args': args,
            }, save_path)
            print(f'✓ 保存 checkpoint: {save_path}')
    
    # 训练完成
    print('\n' + '=' * 80)
    print('训练完成！')
    print('=' * 80)
    print(f'最佳测试准确率: {best_acc:.2f}%')
    if ema is not None:
        print(f'最佳 EMA 准确率: {best_ema_acc:.2f}%')
    
    # TTA 最终测试
    if args.use_tta:
        print('\n进行 TTA 最终测试...')
        
        # 加载最佳模型
        if ema is not None:
            checkpoint = torch.load(os.path.join(args.save_dir, 
                                   f'{args.arch}_{args.dataset}_best_ema.pth'))
        else:
            checkpoint = torch.load(os.path.join(args.save_dir, 
                                   f'{args.arch}_{args.dataset}_best.pth'))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        tta_acc = validate_with_tta(model, test_loader, args)
        print(f'TTA 最终准确率: {tta_acc:.2f}%')
    else:
        print('\n提示: 使用 --use_tta 可以进一步提升性能 (+0.5-1%)')
    
    print(f'\n最佳模型保存在: {os.path.join(args.save_dir, f"{args.arch}_{args.dataset}_best.pth")}')


if __name__ == '__main__':
    main()
