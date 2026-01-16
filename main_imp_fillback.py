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
import vit_pruning_utils
import vit_structured_pruning  # 新增：结构化剪枝模块
import vit_pruning_utils_head_mlp  # 新增：Head+MLP组合剪枝模块

from reg_pruner_files import reg_pruner
# 注释掉wandb以提升训练速度
# import wandb

# 设置 CUDA_VISIBLE_DEVICES 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 注释掉wandb环境变量设置以提升训练速度
# 设置 WANDB_API_KEY 环境变量
# os.environ["WANDB_API_KEY"] = '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'


parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--arch', type=str, default='res20s', help='model architecture')
parser.add_argument('--vit_pretrained', action='store_true', help='use pretrained model (for ViT)')
parser.add_argument('--file_name', type=str, default=None, help='dataset index')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='cifar100_rsst_output_resnet20_l1_exp_custom_exponents4', type=str)
parser.add_argument('--exp_name', type=str, default=None, help='custom experiment name for wandb (auto-generate if None)')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--init', type=str, default='init_model/cifar100_output_resnet20_l1_x_init.pth.tar', help='init file')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
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
parser.add_argument('--vit_structured', action='store_true', help='use structured pruning for ViT (head-level, not element-wise)')
parser.add_argument('--vit_prune_target', default='head', type=str, choices=['head', 'mlp', 'both'], 
                    help='ViT structured pruning target: head (attention heads), mlp (MLP neurons), both (head+mlp)')
parser.add_argument('--mlp_prune_ratio', default=None, type=float, 
                    help='MLP neuron pruning ratio (default: use same as --rate)')
parser.add_argument('--sorting_mode', default='layer-wise', type=str, choices=['layer-wise', 'global'],
                    help='Sorting mode for ViT structured pruning: layer-wise (each layer independently) or global (all layers mixed)')
best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    args.use_sparse_conv = False
    print(args)
    
    # 注释掉WandB初始化以提升训练速度
    # 初始化WandB - 灵活的实验名称
    # if args.exp_name:
    #     # 用户自定义名称
    #     wdb_name = args.exp_name
    # else:
    #     # 自动生成名称
    #     import datetime
    #     timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    #     
    #     # 基础信息
    #     name_parts = [args.struct, args.arch, args.dataset]
    #     
    #     # 添加关键参数
    #     if args.struct == 'rsst':
    #         name_parts.append(f"sched_{args.RST_schedule}")
    #         name_parts.append(f"reg_{args.reg_granularity_prune}")
    #         if args.RST_schedule == 'exp_custom_exponents':
    #             name_parts.append(f"exp{args.exponents}")
    #     elif args.struct == 'refill':
    #         name_parts.append(f"fill_{args.fillback_rate}")
    #     
    #     name_parts.append(f"crit_{args.criteria}")
    #     name_parts.append(f"rate_{args.rate}")
    #     
    #     # 添加预训练标识
    #     if hasattr(args, 'vit_pretrained') and args.vit_pretrained:
    #         name_parts.append("pretrained")
    #     
    #     # 添加结构化剪枝标识
    #     if hasattr(args, 'vit_structured') and args.vit_structured:
    #         if hasattr(args, 'vit_prune_target'):
    #             name_parts.append(f"struct_{args.vit_prune_target}")
    #         else:
    #             name_parts.append("struct_head")
    #     
    #     # 添加时间戳
    #     name_parts.append(timestamp)
    #     
    #     wdb_name = '_'.join(name_parts)
    # 
    # print(f'WandB实验名称: {wdb_name}')
    # wandb.init(project='RSST', entity='ycx', name=wdb_name, config=vars(parser.parse_args()))
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
    # 注释掉wandb.watch以提升训练速度
    # wandb.watch(model, log='all', log_freq=1, log_graph=True)
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
            # 跳过分类头和conv1（ResNet用fc，ViT用head）
            if key.startswith('fc') or key.startswith('conv1') or key.startswith('head'):
                del initialization[key]

        # 判断模型类型并恢复对应的分类头（互斥选择）
        if 'head.weight' in new_initialization:
            # ViT模型 - 恢复head层
            num_classes = new_initialization['head.weight'].shape[0]
            print(f"✓ 检测到ViT模型，使用新初始化的head层（类别数：{num_classes}）")
            initialization['head.weight'] = new_initialization['head.weight']
            initialization['head.bias'] = new_initialization['head.bias']
            
        elif 'fc.weight' in new_initialization:
            # ResNet模型 - 恢复fc和conv1层
            num_classes = new_initialization['fc.weight'].shape[0]
            print(f"✓ 检测到ResNet模型，使用新初始化的fc层（类别数：{num_classes}）")
            initialization['fc.weight'] = new_initialization['fc.weight']
            initialization['fc.bias'] = new_initialization['fc.bias']
            
            # ResNet还需要恢复conv1
            if 'conv1.weight' in new_initialization:
                initialization['conv1.weight'] = new_initialization['conv1.weight']
        else:
            raise ValueError("❌ 无法识别模型类型：既没有head也没有fc层！请检查模型结构。")
        
        model.load_state_dict(initialization)
    else:
        # lottery ticket (lt) - 也需要处理分类头不匹配的问题
        print(initialization.keys())
        
        # 判断模型类型并处理分类头（互斥选择）
        if 'head.weight' in new_initialization:
            # ViT模型 - 检查head维度是否匹配
            if 'head.weight' in initialization:
                init_classes = initialization['head.weight'].shape[0]
                new_classes = new_initialization['head.weight'].shape[0]
                if init_classes != new_classes:
                    print(f"⚠️  检测到head类别数不匹配（{init_classes} → {new_classes}），使用新的head层")
                    initialization['head.weight'] = new_initialization['head.weight']
                    initialization['head.bias'] = new_initialization['head.bias']
        elif 'fc.weight' in new_initialization:
            
            # ResNet模型 - 检查fc维度是否匹配
            if 'fc.weight' in initialization:
                init_classes = initialization['fc.weight'].shape[0]
                new_classes = new_initialization['fc.weight'].shape[0]
                if init_classes != new_classes:
                    print(f"⚠️  检测到fc类别数不匹配（{init_classes} → {new_classes}），使用新的fc层")
                    initialization['fc.weight'] = new_initialization['fc.weight']
                    initialization['fc.bias'] = new_initialization['fc.bias']
        
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
            if vit_pruning_utils.is_vit_model(model):
                vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
            else:
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
        

        # 根据模型类型选择合适的sparsity检查函数
        if vit_pruning_utils.is_vit_model(model):
            remain_weight = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
        else:
            remain_weight = check_sparsity(model, conv1=False)
        # 注释掉wandb.log以提升训练速度
        # wandb.log({'remain_weight': remain_weight})
        if state > 0 and passer.args.struct == 'rsst':
            passer.reg_plot_init = 0

            passer.reg_plot = []
            # 注释掉wandb.log以提升训练速度
            # wandb.log({'reg_lambd': passer.reg_plot_init})
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
            # 注释掉wandb.log以提升训练速度
            # 记录训练损失和准确率（epoch级别，减少网络请求频率）
            # wandb.log({
            #     'prune_times': state, 
            #     'epoch': epoch,  
            #     'train_accuracy': acc, 
            #     'val_accuracy': tacc, 
            #     'test_accuracy': test_tacc
            # })

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
        
            # 注释掉绘图代码以提升训练速度
            # plt.plot(all_result['train'], label='train_acc')
            # plt.plot(all_result['ta'], label='val_acc')
            # plt.plot(all_result['test_ta'], label='test_acc')
            # plt.legend()
            # plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            # plt.close()

        # 在训练后（finetuning）将正则化的项剪掉
        if state > 0 and passer.args.struct == 'rsst':
            # passer.reg_plot_init = 0
            passer.reg_plot_dict[state] = passer.reg_plot
            # print('reg_plot_dict',passer.reg_plot_dict)
            # 注释掉绘图代码以提升训练速度
            # 绘制正则化lambda折线图
            # plt.figure(figsize=(10, 5))  # 可以调整图的大小
            # plt.plot(range(len(passer.reg_plot)), passer.reg_plot, marker='o', linestyle='-', color='b')  # 折线图，带圆形标记

            # 添加标题和坐标轴标签
            # plt.title("Regularization Parameter Changes Over Iterations")
            # plt.xlabel("Iterations")
            # plt.ylabel("Lambda (Regularization Parameter)")
            # 可选：添加网格
            # plt.grid(True)
            # 保存图表（服务器环境不需要显示）
            # reg_plot_path = os.path.join(args.save_dir, f'reg_plot_state_{state}.png')
            # plt.savefig(reg_plot_path)
            # plt.close()
            # print(f'正则化lambda折线图已保存到: {reg_plot_path}')

            # 遍历模型的所有模块，并为每个卷积层/线性层进行处理
            is_vit = vit_pruning_utils.is_vit_model(model)
            for i, (name, m) in enumerate(model.named_modules()):
                # 判断模块是否为卷积层（CNN）或Linear层（ViT）
                if isinstance(m, nn.Conv2d):
                    # 判断是否处理第一层卷积或者其他卷积层
                    # 对于ViT模型，跳过patch_embed层
                    if name != 'conv1' and not (is_vit and 'patch_embed' in name):
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
                
                elif isinstance(m, nn.Linear) and is_vit:
                    # 处理ViT的Attention和MLP层
                    if 'attn' in name or 'mlp' in name:
                        mask_key = name + '.weight_mask'
                        if mask_key in passer.current_mask:
                            # 这一步应该放在正则化后 实现剪枝
                            m.weight.data = initialization[name + ".weight"]
                            mask = passer.current_mask[mask_key]
                            print('pruning ViT layer with custom mask:', name)
                        prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))

        #report result
        validate(val_loader, model, criterion) # extra forward
        # check_sparsity(model, conv1=False)

        print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []
        
        # ⭐ 在非结构化剪枝之前保存训练后的权重（关键：必须在pruning_model_vit之前！）
        train_weight = model.state_dict()  # 保存完整的训练后权重，用于准结构化剪枝的重要性计算
        
        best_sa = 0
        start_epoch = 0

        # 根据模型类型选择剪枝函数
        is_vit = vit_pruning_utils.is_vit_model(model)
        
        if is_vit:
            # ========== ViT非结构化剪枝 (原有逻辑) ==========
            vit_pruning_utils.pruning_model_vit(model, args.rate, prune_patch_embed=False)
            remain_weight_after = vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
            
            # 提取mask
            current_mask = vit_pruning_utils.extract_mask_vit(model.state_dict())
            passer.current_mask = current_mask
            
            # 移除剪枝重参数化
            vit_pruning_utils.remove_prune_vit(model, prune_patch_embed=False)
            
        else:
            # ========== ResNet剪枝 (原有逻辑) ==========
            pruning_model(model, args.rate, conv1=False)
            remain_weight_after = check_sparsity(model, conv1=False)
            
            # 提取mask
            current_mask = extract_mask(model.state_dict())
            passer.current_mask = current_mask
            
            # 移除剪枝重参数化
            remove_prune(model, conv1=False)
        
        # 注释掉wandb.log以提升训练速度
        # if remain_weight_after is not None:
        #     wandb.log({'remain_weight_after': remain_weight_after})

        model.load_state_dict(initialization)
        
        #########################################Refill/RSST Method###########################################################
        if args.struct == 'refill':
            print('执行Refill算法')
            if is_vit:
                # 判断是否使用准结构化剪枝
                if args.vit_structured:
                    # 确定MLP剪枝率
                    mlp_ratio = args.mlp_prune_ratio if args.mlp_prune_ratio is not None else args.rate
                    
                    if args.vit_prune_target == 'both':
                        print(f'[ViT] 使用Head+MLP组合准结构化剪枝 (Refill)')
                        print(f'  - Head剪枝率: {args.rate}')
                        print(f'  - MLP剪枝率: {mlp_ratio}')
                        model = vit_pruning_utils_head_mlp.prune_model_custom_fillback_vit_head_and_mlp(
                            model, mask_dict=current_mask, train_loader=train_loader,
                            trained_weight=train_weight, init_weight=initialization,
                            criteria=args.criteria, head_prune_ratio=args.rate, 
                            mlp_prune_ratio=mlp_ratio, return_mask_only=False,
                            sorting_mode=args.sorting_mode)
                    elif args.vit_prune_target == 'head':
                        print('[ViT] 使用Head级别准结构化剪枝 (Refill)')
                        model = vit_pruning_utils.prune_model_custom_fillback_vit_by_head(
                            model, mask_dict=current_mask, train_loader=train_loader,
                            trained_weight=train_weight, init_weight=initialization,
                            criteria=args.criteria, prune_ratio=args.rate,
                            return_mask_only=False)
                    elif args.vit_prune_target == 'mlp':
                        raise NotImplementedError('单独MLP剪枝尚未实现，请使用both模式')
                else:
                    print('[ViT] 使用Element-wise非结构化剪枝 (Refill)')
                    model = vit_pruning_utils.prune_model_custom_fillback_vit(
                        model, mask_dict=current_mask, train_loader=train_loader,
                        trained_weight=train_weight, init_weight=initialization,
                        criteria=args.criteria, fillback_rate=args.fillback_rate, 
                        return_mask_only=False)
            else:
                model = prune_model_custom_fillback(
                    model, mask_dict=current_mask, train_loader=train_loader,
                    trained_weight=train_weight, init_weight=initialization,
                    criteria=args.criteria, fillback_rate=args.fillback_rate, 
                    return_mask_only=False)
        elif args.struct == 'rsst':
            print('执行RSST算法')
            # rsst剪枝功能 返回refill mask而不剪枝
            if is_vit:
                # 判断是否使用准结构化剪枝
                if args.vit_structured:
                    # 确定MLP剪枝率
                    mlp_ratio = args.mlp_prune_ratio if args.mlp_prune_ratio is not None else args.rate
                    
                    if args.vit_prune_target == 'both':
                        print(f'[ViT] 使用Head+MLP组合准结构化剪枝 (RSST)')
                        print(f'  - Head剪枝率: {args.rate}')
                        print(f'  - MLP剪枝率: {mlp_ratio}')
                        mask = vit_pruning_utils_head_mlp.prune_model_custom_fillback_vit_head_and_mlp(
                            model, mask_dict=current_mask, train_loader=train_loader,
                            trained_weight=train_weight, init_weight=initialization,
                            criteria=args.criteria, head_prune_ratio=args.rate,
                            mlp_prune_ratio=mlp_ratio, return_mask_only=True,
                            sorting_mode=args.sorting_mode)
                    elif args.vit_prune_target == 'head':
                        print('[ViT] 使用Head级别准结构化剪枝 (RSST)')
                        mask = vit_pruning_utils.prune_model_custom_fillback_vit_by_head(
                            model, mask_dict=current_mask, train_loader=train_loader,
                            trained_weight=train_weight, init_weight=initialization,
                            criteria=args.criteria, prune_ratio=args.rate,
                            return_mask_only=True)
                    elif args.vit_prune_target == 'mlp':
                        raise NotImplementedError('单独MLP剪枝尚未实现，请使用both模式')
                else:
                    print('[ViT] 使用Element-wise非结构化剪枝 (RSST)')
                    mask = vit_pruning_utils.prune_model_custom_fillback_vit(
                        model, mask_dict=current_mask, train_loader=train_loader,
                        trained_weight=train_weight, init_weight=initialization,
                        criteria=args.criteria, fillback_rate=0.0, 
                        return_mask_only=True)
            else:
                mask = prune_model_custom_fillback(
                    model, mask_dict=current_mask, train_loader=train_loader,
                    trained_weight=train_weight, init_weight=initialization,
                    criteria=args.criteria, fillback_rate=0.0, 
                    return_mask_only=True)
            # 传递正则化索引
            # passer.current_mask = current_mask
            passer.refill_mask = mask
        else:
            ValueError('错误:没有那个struct算法')

        # 检查最终剪枝效果
        if vit_pruning_utils.is_vit_model(model):
            vit_pruning_utils.check_sparsity_vit(model, prune_patch_embed=False)
        else:
            check_sparsity(model, conv1=False)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

def update_reg(passer, pruner, model, state, i, j):
    """
    更新正则化参数
    
    对于准结构化剪枝（包括ViT的head级别剪枝），正则化仍然适用：
    - refill_mask标记哪些weights/heads应该被剪枝（mask=0）
    - 正则化逐渐压缩这些weights，实现渐进式剪枝
    """
    # 如果没有refill_mask或current_mask，跳过
    if passer.refill_mask is None or passer.current_mask is None:
        return
    
    is_vit = vit_pruning_utils.is_vit_model(model)

    for name, m in model.named_modules():
        # 检查模块是否为卷积层或线性层
        should_process = False
        if isinstance(m, nn.Conv2d) and not is_vit:
            if name != 'conv1':
                should_process = True
        elif isinstance(m, nn.Linear) and is_vit:
            if 'attn' in name or 'mlp' in name:
                should_process = True
        
        if should_process:
                # 检查mask是否存在
                if name not in passer.refill_mask:
                    continue
                if name + '.weight_mask' not in passer.current_mask:
                    continue
                    
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
    # 减少wandb.log频率：只在epoch结束时记录，避免频繁网络请求
    # wandb.log({'train_batch_time': end - start, 'loss': losses.avg, 'top1_acc': top1.avg})
    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
def apply_reg(pruner, model, passer):
    # 遍历模型中的所有模块
    # print('应用正则化到梯度')
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


