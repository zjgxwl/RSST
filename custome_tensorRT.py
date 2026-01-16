import os
import time
import random
import shutil
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
# from advertorch.utils import NormalizeByChannelMeanStd
from normalize_utils import NormalizeByChannelMeanStd  # 自定义实现，功能相同
from utils import *
from pruning_utils_2 import *
from pruning_utils_unprune import *
from pruning_utils import prune_model_custom_fillback

parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashionmnist', 'cifar100'], help='dataset')
parser.add_argument('--arch', type=str, default='res20s', help='model architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='tensorrt_test', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save_model', action="store_true", help="whether saving model")

##################################### training setting #################################################
parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--mask_dir', default=None, type=str, help='mask direction for ticket')
parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
parser.add_argument('--fc', action="store_true", help="whether rewind fc")
parser.add_argument('--type', type=str, default=None, choices=['ewp', 'random_path', 'betweenness', 'hessian_abs', 'taylor1_abs','intgrads','identity', 'omp'])
parser.add_argument('--add-back', action="store_true", help="add back weights")
parser.add_argument('--prune-type', type=str, choices=["lt", 'pt', 'st', 'mt', 'trained', 'transfer'])
parser.add_argument('--num-paths', default=50000, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--evaluate-p', type=float, default=0.00)
parser.add_argument('--evaluate-random', action="store_true")
parser.add_argument('--evaluate-full', action="store_true")
parser.add_argument('--reuse', action="store_true")
parser.add_argument('--use-original', action="store_true")
parser.add_argument('--checkpoint', type=str,default='cifar10_rsst_output_resnet20_l1_exponents_1/0checkpoint.pth.tar')
parser.add_argument('--fillback-rate', type=float)

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    args.use_sparse_conv = False
    args.batch_size=32
    print(args)

    print('*'*50)
    print('conv1 included for prune and rewind: {}'.format(args.conv1))
    print('fc included for rewind: {}'.format(args.fc))
    print('*'*50)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)

    criterion = nn.CrossEntropyLoss()
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    except:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    start = time.time()
    current_mask = extract_mask(state_dict)
    print(current_mask.keys())

    combined_state_dict = {}
    for key in state_dict:
        if key in current_mask:
            combined_state_dict[key[:-5]] = current_mask[key] * state_dict[key[:-5] + "_orig"]
        elif not 'orig' in key:
            combined_state_dict[key] = state_dict[key]

    model.load_state_dict(combined_state_dict, strict=False)
    model = model.cuda()
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32).cuda()
    torch.onnx.export(model, dummy_input, "aasimple_conv.onnx", opset_version=11)

    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        for i in range(10):
            with torch.no_grad():
                x = torch.randn((64, 3, 32, 32)).cuda()
                output = model(x)
                del output

    print(prof.display(show_events=False))

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine(onnx_file_path):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = 1

            with open(onnx_file_path, 'rb') as model:
                parser.parse(model.read())

            return builder.build_cuda_engine(network)

    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def do_inference(context, bindings, inputs, outputs, stream):
        [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
        stream.synchronize()

    engine = build_engine("aasimple_conv.onnx")
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 准备输入数据
    data = np.random.random((1, 3, 32, 32)).astype(np.float32)
    np.copyto(inputs[0][0], data.ravel())

    # 推理
    do_inference(context, bindings, inputs, outputs, stream)
    builder = trt.Builder(TRT_LOGGER)
    builder.sparse_weights = True  # 启用稀疏性优化
    # 获取输出
    output = outputs[0][0].reshape((1, 16, 32, 32))
    print(output)

def save_checkpoint(state, is_SA_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))

def load_ticket(model, args):
    # 加载预训练权重
    if args.pretrained:
        initalization = torch.load(args.pretrained, map_location=torch.device('cuda:'+str(args.gpu)))
        if 'init_weight' in initalization.keys():
            print('loading from init_weight')
            initalization = initalization['init_weight']
        elif 'state_dict' in initalization.keys():
            print('loading from state_dict')
            initalization = initalization['state_dict']

        loading_weight = extract_main_weight(initalization, fc=True, conv1=True)
        new_initialization = model.state_dict()
        if not 'normalize.std' in loading_weight:
            loading_weight['normalize.std'] = new_initialization['normalize.std']
            loading_weight['normalize.mean'] = new_initialization['normalize.mean']

        if not (args.prune_type == 'lt' or args.prune_type == 'trained'):
            keys = list(loading_weight.keys())
            for key in keys:
                if key.startswith('fc') or key.startswith('conv1'):
                    del loading_weight[key]

            loading_weight['fc.weight'] = new_initialization['fc.weight']
            loading_weight['fc.bias'] = new_initialization['fc.bias']
            loading_weight['conv1.weight'] = new_initialization['conv1.weight']

        print('*number of loading weight={}'.format(len(loading_weight.keys())))
        print('*number of model weight={}'.format(len(model.state_dict().keys())))
        model.load_state_dict(loading_weight)

def warmup_lr(epoch, step, optimizer, one_epoch_step):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step
    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)
    for p in optimizer.param_groups:
        p['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_model_dataset(args):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    elif args.dataset == 'fashionmnist':
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.FashionMNIST(root=args.data, train=True, download=True, transform=train_transform)
        val_dataset = datasets.FashionMNIST(root=args.data, train=False, download=True, transform=test_transform)
        test_dataset = datasets.FashionMNIST(root=args.data, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = models.resnet18(pretrained=False)  # 这里使用resnet18作为示例模型，实际使用时替换为你需要的模型
    return model, train_loader, val_loader, test_loader

if __name__ == '__main__':
    main()
