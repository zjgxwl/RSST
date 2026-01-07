import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SparseConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 将稀疏权重转换为密集形式
        weight_dense = self.weight.to_sparse().to_dense()
        return F.conv2d(x, weight_dense, stride=self.stride, padding=self.padding)

class SparseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SparseConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SparseResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = SparseConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def SparseResNet20():
    return SparseResNet(SparseBasicBlock, [3, 3, 3])

# # 示例使用
# model = SparseResNet20().cuda()  # 将模型移至GPU
# input_data = torch.randn(1, 3, 32, 32).cuda()  # 将输入数据移至GPU
# output = model(input_data)
# print(output.shape)
