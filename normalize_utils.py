"""
替代 advertorch.utils.NormalizeByChannelMeanStd 的简单实现
避免依赖 advertorch 库
"""
import torch
import torch.nn as nn


class NormalizeByChannelMeanStd(nn.Module):
    """
    按通道进行归一化
    等价于 advertorch.utils.NormalizeByChannelMeanStd
    """
    def __init__(self, mean, std):
        """
        Args:
            mean: 均值列表 [R_mean, G_mean, B_mean]
            std: 标准差列表 [R_std, G_std, B_std]
        """
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        
        # 将mean和std reshape为 (1, C, 1, 1) 以便广播
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))
    
    def forward(self, tensor):
        """
        Args:
            tensor: (B, C, H, W) 形状的张量
        
        Returns:
            归一化后的张量
        """
        return (tensor - self.mean) / self.std
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.view(-1).tolist()}, std={self.std.view(-1).tolist()})"

