import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from advertorch.utils import NormalizeByChannelMeanStd
from normalize_utils import NormalizeByChannelMeanStd  # 自定义实现，功能相同
# from conv_ops import conv_skip_zero_channels  # 注释掉，ViT不需要

def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1_zero_mask = None
        self.conv2_zero_mask = None
        self.zero_channels = {}

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def _initialize_masks(self, zero_channels):
        with torch.no_grad():
            # conv1 掩码
            mask = torch.ones(self.conv1.weight.size(0), device=self.conv1.weight.device)
            conv1_key = 'conv1.weight'
            if conv1_key in zero_channels:
                for idx in zero_channels[conv1_key]:
                    mask[idx] = 0
            self.conv1_zero_mask = mask
            self.zero_channels[conv1_key] = zero_channels.get(conv1_key, [])
            print(f"BasicBlock conv1_zero_mask: {self.conv1_zero_mask}")

            # conv2 掩码
            mask = torch.ones(self.conv2.weight.size(0), device=self.conv2.weight.device)
            conv2_key = 'conv2.weight'
            if conv2_key in zero_channels:
                for idx in zero_channels[conv2_key]:
                    mask[idx] = 0
            self.conv2_zero_mask = mask
            self.zero_channels[conv2_key] = zero_channels.get(conv2_key, [])
            print(f"BasicBlock conv2_zero_mask: {self.conv2_zero_mask}")

    def conv_with_mask(self, x, conv, mask):
        N, C, H, W = x.shape
        out_C = conv.weight.size(0)
        output_H = (H + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        output_W = (W + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
        out = torch.zeros(N, out_C, output_H, output_W, device=x.device).contiguous()

        if mask is None:
            mask = torch.ones(out_C, device=x.device)
            print(f"Warning: Mask is None, using default all-ones mask for {conv.__class__.__name__}")

        mask = mask.to(x.device)
        nonzero_indices = mask.nonzero(as_tuple=True)[0]
        print(f"{conv.__class__.__name__} - Nonzero channels: {len(nonzero_indices)} / {out_C}, Mask: {mask}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if len(nonzero_indices) == out_C:
            out = F.conv2d(x, conv.weight, stride=conv.stride, padding=conv.padding)
            print(f"{conv.__class__.__name__} - Using cuDNN")
        elif len(nonzero_indices) > 0:
            print(f"{conv.__class__.__name__} - Using custom op")
            conv_skip_zero_channels(x.contiguous(), conv.weight.data.contiguous(), out, mask.contiguous(),
                                    conv.stride[0], conv.padding[0])
        else:
            print(f"{conv.__class__.__name__} - All channels zero, skipping")

        end.record()
        torch.cuda.synchronize()
        print(f"{conv.__class__.__name__} time: {start.elapsed_time(end):.6f} ms")
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_with_mask(x, self.conv1, self.conv1_zero_mask)))
        out = self.bn2(self.conv_with_mask(out, self.conv2, self.conv2_zero_mask))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1_zero_mask = None

        self.apply(_weights_init)
        self.zero_channels = {}

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_masks(self):
        with torch.no_grad():
            # conv1 掩码
            mask = torch.ones(self.conv1.weight.size(0), device=self.conv1.weight.device)
            conv1_key = 'conv1.weight'
            if conv1_key in self.zero_channels:
                for idx in self.zero_channels[conv1_key]:
                    mask[idx] = 0
            self.conv1_zero_mask = mask
            print(f"ResNet conv1_zero_mask: {self.conv1_zero_mask}")

            # 更新每一层的掩码
            for layer_idx, layer in enumerate([self.layer1, self.layer2, self.layer3], 1):
                for block_idx, block in enumerate(layer):
                    block_zero_channels = {
                        'conv1.weight': self.zero_channels.get(f'layer{layer_idx}.{block_idx}.conv1.weight', []),
                        'conv2.weight': self.zero_channels.get(f'layer{layer_idx}.{block_idx}.conv2.weight', [])
                    }
                    block._initialize_masks(block_zero_channels)

    def conv_with_mask(self, x, conv, mask):
        N, C, H, W = x.shape
        out_C = conv.weight.size(0)
        output_H = (H + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        output_W = (W + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
        out = torch.zeros(N, out_C, output_H, output_W, device=x.device).contiguous()

        if mask is None:
            mask = torch.ones(out_C, device=x.device)
            print(f"Warning: Mask is None, using default all-ones mask for {conv.__class__.__name__}")

        mask = mask.to(x.device)
        nonzero_indices = mask.nonzero(as_tuple=True)[0]
        print(f"{conv.__class__.__name__} - Nonzero channels: {len(nonzero_indices)} / {out_C}, Mask: {mask}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if len(nonzero_indices) == out_C:
            out = F.conv2d(x, conv.weight, stride=conv.stride, padding=conv.padding)
            print(f"{conv.__class__.__name__} - Using cuDNN")
        elif len(nonzero_indices) > 0:
            print(f"{conv.__class__.__name__} - Using custom op")
            conv_skip_zero_channels(x.contiguous(), conv.weight.data.contiguous(), out, mask.contiguous(),
                                    conv.stride[0], conv.padding[0])
        else:
            print(f"{conv.__class__.__name__} - All channels zero, skipping")

        end.record()
        torch.cuda.synchronize()
        print(f"{conv.__class__.__name__} time: {start.elapsed_time(end):.6f} ms")
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_with_mask(x, self.conv1, self.conv1_zero_mask)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, (out.size(3),))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet20(number_class=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=number_class)

def resnet32(number_class=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=number_class)

def resnet44(number_class=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=number_class)

def resnet56(number_class=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=number_class)

def resnet110(number_class=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=number_class)

def resnet1202(number_class=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=number_class)

    return zero_channels

def load_model(checkpoint_path, device='cpu', num_classes=10):
    model = resnet20(num_classes=num_classes)
    model.eval()

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"{checkpoint_path} not found, using random weights")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Using random weights due to state_dict mismatch")

    model.to(device)
    model.zero_channels = print_nonzeros_and_find_zero_channels(model)
    print(f"Zero channels computed: {model.zero_channels}")
    model._initialize_masks()

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/data/haodonga/RSST-master/cifar10_rsst_output_resnet20_l1_exponents_5/19checkpoint.pth.tar"
    model = load_model(checkpoint_path, device)

    x = torch.randn(1, 3, 32, 32).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")