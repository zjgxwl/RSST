"""
Mamba-Small Baseline Training Script
训练 Mamba-Small 的 baseline 性能（无剪枝）

基于 Gemini 建议的现代化训练方案:
- 优化器: AdamW with Cosine LR
- 强数据增强: RandAugment, Mixup, Cutmix
- 训练 300 epochs
- Weight Decay: 0.05
- Patch Size: 4x4 (适配 32x32 CIFAR)
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

# 导入 Mamba 模型
from models.mamba import mamba_small, mamba_tiny, mamba_base


# ============================================================================
# 参数设置
# ============================================================================

parser = argparse.ArgumentParser(description='Mamba-Small Baseline Training')

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

# 训练参数（Gemini 推荐）
parser.add_argument('--epochs', type=int, default=300,
                    help='训练轮数（推荐300）')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='初始学习率（推荐 1e-3 或 5e-4）')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='权重衰减（推荐 0.05，关键参数）')
parser.add_argument('--warmup_epochs', type=int, default=20,
                    help='warmup 轮数')
parser.add_argument('--drop_path', type=float, default=0.1,
                    help='Stochastic Depth / Drop Path rate')

# 数据增强（Gemini 推荐）
parser.add_argument('--use_randaugment', action='store_true', default=True,
                    help='使用 RandAugment')
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

# 其他
parser.add_argument('--workers', type=int, default=4,
                    help='数据加载线程数')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID')
parser.add_argument('--save_dir', type=str, default='./checkpoint/mamba_baseline',
                    help='模型保存目录')
parser.add_argument('--log_interval', type=int, default=50,
                    help='日志打印间隔')
parser.add_argument('--eval_interval', type=int, default=10,
                    help='评估间隔（epochs）')

args = parser.parse_args()


# ============================================================================
# 工具函数
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
                                    min_lr=0, last_epoch=-1):
    """
    Cosine 学习率调度器（带 warmup）
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup: 线性增长
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class MixupCutmix:
    """
    Mixup 和 Cutmix 数据增强
    """
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
            
            # 调整 lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            targets_a, targets_b = targets, targets[index]
            return images, (targets_a, targets_b, lam)


def mixup_criterion(criterion, pred, targets):
    """
    Mixup/Cutmix 损失函数
    """
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
    """获取数据加载器"""
    
    # 数据增强
    if args.use_randaugment:
        from torchvision.transforms import RandAugment
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),  # RandAugment (2, 9)
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010)),
        ])
    else:
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
    else:  # cifar100
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True,
            transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True,
            transform=test_transform
        )
        num_classes = 100
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                epoch, args, mixup_cutmix=None):
    """训练一个 epoch"""
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
        
        # 前向传播
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
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


def validate(model, test_loader, criterion, args):
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    print(f'验证结果: Loss {losses.avg:.4f} | Acc@1 {top1.avg:.2f}%')
    
    return losses.avg, top1.avg


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
    print('Mamba-Small Baseline 训练')
    print('=' * 80)
    print(f'数据集: {args.dataset}')
    print(f'模型: {args.arch}')
    print(f'训练轮数: {args.epochs}')
    print(f'Batch Size: {args.batch_size}')
    print(f'初始学习率: {args.lr}')
    print(f'Weight Decay: {args.weight_decay}')
    print(f'Warmup Epochs: {args.warmup_epochs}')
    print(f'数据增强:')
    print(f'  - RandAugment: {args.use_randaugment}')
    print(f'  - Mixup (α={args.mixup_alpha}): {args.use_mixup}')
    print(f'  - Cutmix (α={args.cutmix_alpha}): {args.use_cutmix}')
    print(f'  - Label Smoothing: {args.label_smoothing}')
    print('=' * 80)
    
    # 加载数据
    print('\n加载数据集...')
    train_loader, test_loader, num_classes = get_dataloaders(args)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    print(f'类别数: {num_classes}')
    
    # 创建模型
    print(f'\n创建模型: {args.arch}')
    if args.arch == 'mamba_tiny':
        model = mamba_tiny(num_classes=num_classes, img_size=32)
    elif args.arch == 'mamba_small':
        model = mamba_small(num_classes=num_classes, img_size=32)
    elif args.arch == 'mamba_base':
        model = mamba_base(num_classes=num_classes, img_size=32)
    else:
        raise ValueError(f'Unknown architecture: {args.arch}')
    
    model = model.cuda(args.gpu)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)')
    
    # 损失函数（带 Label Smoothing）
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)
    
    # 优化器：AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器：Cosine with Warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=1e-6
    )
    
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
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print('\n开始训练...')
    print('=' * 80)
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 80)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, args, mixup_cutmix
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            test_loss, test_acc = validate(model, test_loader, criterion, args)
            test_losses.append(test_acc)
            test_accs.append(test_acc)
            
            # 保存最佳模型
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
                'args': args,
            }, save_path)
            print(f'✓ 保存 checkpoint: {save_path}')
    
    # 训练完成
    print('\n' + '=' * 80)
    print('训练完成！')
    print('=' * 80)
    print(f'最佳测试准确率: {best_acc:.2f}%')
    print(f'最佳模型保存在: {os.path.join(args.save_dir, f"{args.arch}_{args.dataset}_best.pth")}')
    
    # 最终测试
    print('\n进行最终测试...')
    final_loss, final_acc = validate(model, test_loader, criterion, args)
    print(f'最终测试结果: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
