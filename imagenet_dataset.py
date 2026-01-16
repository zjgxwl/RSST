"""
ImageNet数据集加载器
用于在ImageNet上进行ViT剪枝
"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def imagenet_dataloaders(batch_size=128, data_dir='/path/to/imagenet', num_workers=4):
    """
    ImageNet数据加载器
    
    Args:
        batch_size: 批次大小
        data_dir: ImageNet数据集路径，目录结构应为:
                  data_dir/
                  ├── train/
                  │   ├── n01440764/
                  │   ├── n01443537/
                  │   └── ...
                  └── val/
                      ├── n01440764/
                      ├── n01443537/
                      └── ...
        num_workers: 数据加载的进程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    
    # 检查路径是否存在
    if not os.path.exists(traindir):
        raise ValueError(f"训练集路径不存在: {traindir}\n"
                        f"请确保ImageNet数据集已下载到: {data_dir}")
    if not os.path.exists(valdir):
        raise ValueError(f"验证集路径不存在: {valdir}\n"
                        f"请确保ImageNet数据集已下载到: {data_dir}")
    
    # ImageNet标准数据增强
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 验证/测试集变换
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 创建数据集
    print(f"加载ImageNet数据集从: {data_dir}")
    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)
    
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  类别数: {len(train_dataset.classes)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # ImageNet通常不单独分test set，使用val set作为test
    test_loader = val_loader
    
    return train_loader, val_loader, test_loader


def imagenet_subset_dataloaders(batch_size=128, data_dir='/path/to/imagenet', 
                                num_workers=4, subset_size=10000):
    """
    ImageNet子集加载器（用于快速实验）
    
    Args:
        subset_size: 使用的训练样本数（默认10000，约0.78%）
    """
    from torch.utils.data import Subset
    import random
    
    train_loader, val_loader, test_loader = imagenet_dataloaders(
        batch_size=batch_size, 
        data_dir=data_dir, 
        num_workers=num_workers
    )
    
    # 创建训练集子集
    train_dataset = train_loader.dataset
    indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
    train_subset = Subset(train_dataset, indices)
    
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"使用ImageNet子集: {len(train_subset)} 训练样本")
    
    return train_subset_loader, val_loader, test_loader

