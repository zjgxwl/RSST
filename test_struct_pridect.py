# import torch
import argparse
from models.resnets import resnet20
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import os
from utils import *

# 打印非零计数表并检查全零通道
def print_nonzeros_and_find_zero_channels(model):
    nonzero = total = 0
    zero_channels = {}  # 用于记录每个层的全零通道

    for name, p in model.named_parameters():
        tensor = p.data.cpu()  # 保留为 PyTorch tensor
        nz_count = torch.count_nonzero(tensor).item()  # 统计非零元素个数
        total_params = torch.numel(tensor)  # 计算全部参数数量
        nonzero += nz_count  # 模型全部非零参数
        total += total_params
        #
        # print(
        #     f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')

        # 检查全零通道
        if len(p.shape) > 1:  # 跳过偏置项，只处理权重
            if len(tensor.shape) == 4:  # 卷积层 (out_channels, in_channels, kernel_h, kernel_w)
                for out_c in range(tensor.shape[0]):  # 遍历每个输出通道
                    if torch.count_nonzero(tensor[out_c]).item() == 0:
                        if name not in zero_channels:
                            zero_channels[name] = []
                        zero_channels[name].append(out_c)
            elif len(tensor.shape) == 2:  # 线性层 (out_features, in_features)
                for out_f in range(tensor.shape[0]):  # 遍历每个输出特征
                    if torch.count_nonzero(tensor[out_f]).item() == 0:
                        if name not in zero_channels:
                            zero_channels[name] = []
                        zero_channels[name].append(out_f)

    # print(
    #     f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')

    # 打印全零通道
    # if zero_channels:
    #     # print("\nAll-zero channels found:")
    #     for layer_name, channels in zero_channels.items():
    #         # print(f"Layer {layer_name} has all-zero channels: {channels}")
    # else:
    #     print("\nNo all-zero channels found.")

    return zero_channels


# 测试模型推理速度
def test_inference_speed(model, input_tensor, device='cuda'):
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    # 测量推理时间
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    inference_time = (end_time - start_time) / 100
    print(f"Average Inference Time per batch: {inference_time:.6f} seconds")
def load_model(checkpoint_path,device='cpu'):
    # 加载模型
    # prepare dataset

    # model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet56(number_class=10)
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)

    # 计算全零通道
    model.zero_channels = print_nonzeros_and_find_zero_channels(model)
    # print(f"Zero channels computed: {model.zero_channels}")

    # 初始化掩码
    model._initialize_masks()

    return model





def main(args):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=args.device)

    # Create dummy input tensor for speed test
    input_tensor = torch.randn(512, 3, 32, 32)  # Example input for ResNet18

    # Print non-zero counts and zero channels
    zero_channels = print_nonzeros_and_find_zero_channels(model)

    # Print model inference speed
    test_inference_speed(model, input_tensor, device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

    ##################################### general setting #################################################
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--arch', type=str, default='res20s', help='model architecture')
    parser.add_argument('--file_name', type=str, default=None, help='dataset index')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='cifar100_rsst_output_resnet20_l1_exp_custom_exponents4', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
    parser.add_argument('--checkpoint', type=str, default='rsst_output_resnet56_l2_exp/0checkpoint.pth.tar', help='checkpoint file')
    parser.add_argument('--init', type=str, default='init_model/cifar100_output_resnet20_l1_x_init.pth.tar',
                        help='init file')

    ##################################### training setting #################################################
    parser.add_argument('--batch_size', type=int, default=3128, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup', default=20, type=int, help='warm up epochs')
    parser.add_argument('--print_freq', default=200, type=int, help='print frequency')
    parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

    ##################################### Pruning setting #################################################
    parser.add_argument('--pruning_times', default=20, type=int, help='overall times of pruning')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
    parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt,pt or pt_trans)')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
    parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
    parser.add_argument('--fc', action="store_true", help="whether rewind fc")
    parser.add_argument('--rewind_epoch', default=24, type=int, help='rewind checkpoint')

    parser.add_argument('--struct', default='rsst', type=str, choices=['refill', 'rsst'],
                        help='overall times of pruning')
    parser.add_argument('--fillback_rate', default=0.0, type=float)
    parser.add_argument('--block_loss_grad', default=False, help="block the grad from loss, only apply weight decay")
    parser.add_argument('--RST_schedule', type=str, default='exp_custom_exponents',
                        choices=['x', 'x^2', 'x^3', 'exp', 'exp_custom', 'exp_custom_exponents'])
    parser.add_argument('--reg_granularity_prune', type=float, default=1, help='正则化阈值')
    parser.add_argument('--criteria', default="l1", type=str, choices=['remain', 'magnitude', 'l1', 'l2', 'saliency'])
    parser.add_argument('--exponents', default=4, type=int, help='此参数用来控制指数函数的曲率')
    parser.add_argument('--device', default='cuda', help='Device to run the model on')
    args = parser.parse_args()
    main(args)
