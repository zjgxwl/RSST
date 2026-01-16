'''
iterative pruning for supervised task 
with lottery tickets or pretrain tickets 
support datasets: cifar10, Fashionmnist, cifar100
'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
import math
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
from pruning_utils import *

from reg_pruner_files import reg_pruner
import wandb

# 设置 CUDA_VISIBLE_DEVICES 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='sparseresnet20', help='model architecture')
parser.add_argument('--file_name', type=str, default=None, help='dataset index')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='cifar10_rsst_output_sparseresnet20_l1_exp_custom_exponents4', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--init', type=str, default='init_model/cifar10_output_sparseresnet20_l1_x_init.pth.tar', help='init file')

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



parser.add_argument('--struct', default='rsst', type=str, choices=['refill','rsst'], help='overall times of pruning')
parser.add_argument('--fillback_rate', default=0.0, type=float)
parser.add_argument('--block_loss_grad', default=False, help="block the grad from loss, only apply weight decay")
parser.add_argument('--RST_schedule', type=str, default='exp_custom_exponents', choices=['x', 'x^2', 'x^3', 'exp', 'exp_custom','exp_custom_exponents' ])
parser.add_argument('--reg_granularity_prune', type=float, default=1, help='正则化阈值')
parser.add_argument('--criteria', default="l1", type=str, choices=['remain', 'magnitude', 'l1', 'l2', 'saliency'])
parser.add_argument('--exponents', default=4, type=int, help='此参数用来控制指数函数的曲率' )
best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    args.use_sparse_conv = False
    print(args)
    # 初始化WandB
    wdb_name = '_'.join([args.struct, args.RST_schedule, args.criteria, args.arch, args.dataset])
    wandb.init(project='RSST', entity='609354432', name=wdb_name, config=vars(parser.parse_args()))
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
    wandb.watch(model, log='all', log_freq=1, log_graph=True)
    model.cuda()


    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    # 声明一个类用于参数传递
    class passer:
        pass
    #初始化类中属性
    passer.test = validate  # 测试函数
    passer.model = None
    passer.test_loader = val_loader  # 验证数据加载器
    passer.criterion = criterion  # 损失函数
    passer.args = args  # 参数
    passer.current_mask = None
    passer.reg = {}  # 保存每层的正则化项
    passer.reg_indices = []
    passer.refill_mask = None # 上一轮迭代中产生的重组mask
    passer.reg_plot_dict = {} # 收集每一次更新的reg画图

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    print(model.normalize)  
    new_initialization = copy.deepcopy(model.state_dict())
    if not os.path.exists(args.init):
        torch.save(new_initialization, args.init)
    initialization = torch.load(args.init)
    if 'state_dict' in initialization:
        initialization = initialization['state_dict']
    
    initialization['normalize.mean'] = new_initialization['normalize.mean']
    initialization['normalize.std'] = new_initialization['normalize.std']

    print(initialization.keys())
    if not args.prune_type == 'lt':
        keys = list(initialization.keys())
        for key in keys:
            if key.startswith('fc') or key.startswith('conv1'):
                del initialization[key]

        initialization['fc.weight'] = new_initialization['fc.weight']
        initialization['fc.bias'] = new_initialization['fc.bias']
        initialization['conv1.weight'] = new_initialization['conv1.weight']
        model.load_state_dict(initialization)
    else:
        print(initialization.keys())
        model.load_state_dict(initialization)
        
    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        try:
            best_sa = checkpoint['best_sa']
            all_result = checkpoint['result']
        except:
            best_sa = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        start_state = checkpoint['state'] if 'state' in checkpoint else 1

        if start_state>0:
            current_mask = extract_mask(checkpoint['state_dict'])
            # for m in current_mask:
            #     print(current_mask[m].float().mean())
            #print(current_mask)
            prune_model_custom(model, current_mask)
            check_sparsity(model, conv1=False)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
        if 'normalize.mean' not in checkpoint['state_dict']:
            checkpoint['state_dict']['normalize.mean'] = new_initialization['normalize.mean']
            checkpoint['state_dict']['normalize.std'] = new_initialization['normalize.std']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initialization = copy.deepcopy(checkpoint['init_weight']) if 'init_weight' in checkpoint else torch.load(args.init)['state_dict']

        if 'normalize.mean' not in initialization:
            initialization['normalize.mean'] = new_initialization['normalize.mean']
            initialization['normalize.std'] = new_initialization['normalize.std']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)
        all_result = {}
        all_result['train'] = [best_sa]
        all_result['test_ta'] = [best_sa]
        all_result['ta'] = [best_sa]

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        start_epoch = 0
        start_state = 0

    print('######################################## Start Standard Training Iterative Pruning ########################################')


    pruner = None
    for state in range(start_state, args.pruning_times):
        # 监视模型

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        

        remain_weight = check_sparsity(model, conv1=False)
        wandb.log({'remain_weight': remain_weight})
        if state > 0 and passer.args.struct == 'rsst':
            passer.reg_plot_init = 0

            passer.reg_plot = []
            wandb.log({'reg_lambd': passer.reg_plot_init})
            pruner = reg_pruner.Pruner(model, args, passer)  # 初始化剪枝构造函数
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch == args.rewind_epoch and args.prune_type == 'rewind_lt' and state == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{args.rewind_epoch}.pth.tar"))
                initialization = copy.deepcopy(model.state_dict())
            if passer.args.struct == 'rsst':
                acc = train(state, train_loader, model, criterion, optimizer, epoch, passer, pruner)
            else:
                acc = train(state, train_loader, model, criterion, optimizer, epoch, passer=None, pruner=None)
            # evaluate on validation set
            tacc = validate(val_loader, model, criterion)
            # evaluate on test set
            test_tacc = validate(test_loader, model, criterion)

            scheduler.step()
            # 记录训练损失和准确率
            wandb.log({'prune_times': state, 'epoch': epoch,  'accuracy': acc, 'val_accuracy': tacc, 'test_accuracy': test_tacc, 'accuracy': acc})

            all_result['train'].append(acc)
            all_result['ta'].append(tacc)
            all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initialization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        # 在训练后（finetuning）将正则化的项剪掉
        if state > 0 and passer.args.struct == 'rsst':
            # passer.reg_plot_init = 0
            passer.reg_plot_dict[state] = passer.reg_plot
            # print('reg_plot_dict',passer.reg_plot_dict)
            # 绘制正则化lambda折线图
            plt.figure(figsize=(10, 5))  # 可以调整图的大小
            plt.plot(range(len(passer.reg_plot)), passer.reg_plot, marker='o', linestyle='-', color='b')  # 折线图，带圆形标记

            # 添加标题和坐标轴标签
            plt.title("Regularization Parameter Changes Over Iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Lambda (Regularization Parameter)")
            # 可选：添加网格
            plt.grid(True)
            # 显示图表
            plt.show()

            # 遍历模型的所有模块，并为每个卷积层进行处理
            for i, (name, m) in enumerate(model.named_modules()):
                # 判断模块是否为卷积层
                if isinstance(m, nn.Conv2d):
                    # 判断是否处理第一层卷积或者其他卷积层
                    if name != 'conv1':
                        # 获取对应的权重掩码
                        mask = passer.current_mask[name + '.weight_mask']
                        # 将掩码张量重新塑形为二维，其中第一维是通道数
                        mask = mask.view(mask.shape[0], -1)
                        # 计算每个通道的掩码之和
                        count = torch.sum(mask, 1)  # 每个通道的1的数量，[C]

                            # 这一步应该放在正则化后 实现剪枝
                        m.weight.data = initialization[name + ".weight"]
                        mask = mask.view(*passer.current_mask[name + '.weight_mask'].shape)
                        print('pruning layer with custom mask:', name)
                        prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))

        #report result
        validate(val_loader, model, criterion) # extra forward
        # check_sparsity(model, conv1=False)

        print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []
        train_weight = model.state_dict()  # 保存训练后的模型权重
        best_sa = 0
        start_epoch = 0

        pruning_model(model, args.rate, conv1=False)
        remain_weight_after = check_sparsity(model, conv1=False)
        wandb.log({'remain_weight_after':remain_weight_after})
        current_mask = extract_mask(model.state_dict())
        passer.current_mask = current_mask
        remove_prune(model, conv1=False)

        model.load_state_dict(initialization)
        #########################################Refill Method###########################################################
        if args.struct == 'refill':
            print('执行Refill算法')
            model = prune_model_custom_fillback(model, mask_dict=current_mask, train_loader=train_loader,
                                        trained_weight=train_weight, init_weight=initialization,criteria=args.criteria, fillback_rate=args.fillback_rate, return_mask_only=False)
        elif args.struct == 'rsst':
            print('执行RSST算法')
            # rsst剪枝功能 返回refill mask而不剪枝
            mask = prune_model_custom_fillback(model, mask_dict=current_mask, train_loader=train_loader,
                                        trained_weight=train_weight, init_weight=initialization,criteria=args.criteria , fillback_rate=0.0 ,return_mask_only=True)
            # 传递正则化索引
            # passer.current_mask = current_mask
            passer.refill_mask = mask
        else:
            ValueError('错误:没有那个struct算法')


        check_sparsity(model, conv1=False)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

def update_reg(passer, pruner, model, state, i, j):
    #TODO 更新正则化

    for name, m in model.named_modules():
        # 检查模块是否为卷积层或线性层
        if isinstance(m, nn.Conv2d) :
            if name != 'conv1':
                refill_mask = passer.refill_mask[name].flatten()
                current_mask = passer.current_mask[name + '.weight_mask'].flatten()
                if refill_mask.shape != current_mask.shape:
                    raise ValueError("掩码的形状不匹配")
                # 输出需要正则化的项的索引 形状： reg[name][reg_indices[name]] += 正则化补偿
                unpruned_indices = torch.where((refill_mask == 0) & (current_mask == 1))
                unpruned_indices_np = unpruned_indices[0].data.cpu().numpy()
                if passer.args.RST_schedule == 'x':
                    # print('更新正则化参数lambda')
                    pruner.reg[name][unpruned_indices_np] += passer.args.reg_granularity_prune
                    passer.reg_plot_init += passer.args.reg_granularity_prune
                    passer.reg_plot.append(passer.reg_plot_init)

                if passer.args.RST_schedule == 'x^2':
                    pruner.reg_[name][unpruned_indices_np] += passer.args.reg_granularity_prune
                    pruner.reg[name][unpruned_indices_np] = pruner.reg_[name][unpruned_indices_np] ** 2

                if passer.args.RST_schedule == 'x^3':
                    pruner.reg_[name][unpruned_indices_np] += passer.args.reg_granularity_prune
                    pruner.reg[name][unpruned_indices_np] = pruner.reg_[name][unpruned_indices_np] ** 3
                if passer.args.RST_schedule == 'exp':
                    pruner.reg_[name][unpruned_indices_np] += passer.args.reg_granularity_prune
                    pruner.reg[name][unpruned_indices_np] = torch.tensor(
                        np.exp(pruner.reg_[name][unpruned_indices_np].cpu().numpy()), device=pruner.reg_[name].device)
                    passer.reg_plot_init += passer.args.reg_granularity_prune
                    passer.reg_plot_init = np.exp(passer.reg_plot_init)
                    passer.reg_plot.append(passer.reg_plot_init)
                if passer.args.RST_schedule == 'exp_custom':
                    print('更新正则化参数lambda')
                    e = math.exp(1)
                    ceil = 3e-4
                    weight_start = 1 / (e - 1) * passer.args.reg_granularity_prune * (math.exp((i + 1) / j) - 1)
                    pruner.reg[name][unpruned_indices_np] = weight_start

                    passer.reg_plot.append(weight_start)

                if passer.args.RST_schedule == 'exp_custom_exponents':
                    print('更新正则化参数lambda')
                    e = math.exp(1)
                    ceil = 3e-4
                    weight_start = 1 / (e - 1) * passer.args.reg_granularity_prune * (math.exp((i + 1) ** args.exponents / j ** args.exponents) - 1)
                    # weight_start = 1 / (e - 1) * passer.args.reg_granularity_prune * (math.exp((i + 1) / j) - 1)
                    pruner.reg[name][unpruned_indices_np] = weight_start

                    passer.reg_plot.append(weight_start)

def train(state,train_loader, model, criterion, optimizer, epoch, passer, pruner):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        j = len(train_loader)
        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)
        optimizer.zero_grad()
        loss.backward()
        if state > 0  and args.struct == 'rsst':
            # 更新正则化 注意设置更新间隔与更新阈值
            if passer.args.reg_granularity_prune * i < 1 :
                update_reg(passer, pruner, model, state, i, j)
            #修改梯度 加入reg项
            model = apply_reg(pruner, model, passer)
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))

            start = time.time()
    wandb.log({'train_batch_time': end - start, 'loss': losses.avg, 'top1_acc': top1.avg})
    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
def apply_reg(pruner, model, passer):
    # 遍历模型中的所有模块
    print('应用正则化到梯度')
    for name, m in model.named_modules():
        # 检查当前模块名称是否在正则化规则字典中
        if name in pruner.reg:
            # 获取当前模块的正则化权重
            reg = pruner.reg[name]  # [N, C]
            # print('regnc',reg)
            # 如果是按权重,正则化，调整reg的形状与权重完全一致
            reg = reg.view_as(m.weight.data)  # [N, C, H, W]
            # print("reg,",torch.unique(reg))
            # reg_lambda_tensor = torch.unique(reg)

            # 计算L2正则化梯度
            l2_grad = reg * m.weight

            # 根据设置选择是否阻止原始梯度的反向传播
            if passer.args.block_loss_grad:
                # 仅使用L2正则化梯度更新权重梯度
                m.weight.grad = l2_grad
            else:
                # print('将L2正则化梯度添加到原有的权重梯度上')
                m.weight.grad += l2_grad
    # wandb.log({'reg_lambda_tensor': reg_lambda_tensor})
    return model
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

def load_weight(model, initialization, args): 
    print('loading pretrained weight')
    loading_weight = extract_main_weight(initialization)
    
    for key in loading_weight.keys():
        if not (key in model.state_dict().keys()):
            print(key)
            assert False

    print('*number of loading weight={}'.format(len(loading_weight.keys())))
    print('*number of model weight={}'.format(len(model.state_dict().keys())))
    model.load_state_dict(loading_weight)

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

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

if __name__ == '__main__':
    main()


