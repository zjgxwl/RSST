import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
from dlth_utils import plot_weights_heatmap, Timer
import matplotlib.pyplot as plt
pjoin = os.path.join

class Pruner(MetaPruner):
    def __init__(self, model, args, passer):
        # 调用父类构造函数
        super(Pruner, self).__init__(model, args, passer)

        # 与正则化相关的变量初始化
        self.reg = {}  # 保存每层的正则化项
        self.delta_reg = {}  # 保存正则化项的变化
        self.hist_mag_ratio = {}  # 历史幅值比率
        self.n_update_reg = {}  # 更新正则化项的次数
        self.iter_update_reg_finished = {}  # 完成更新正则化项的迭代次数
        self.iter_finish_pick = {}  # 完成选择的迭代次数
        self.iter_stabilize_reg = math.inf  # 稳定正则化的迭代阈值
        self.original_w_mag = {}  # 原始权重的幅度
        self.original_kept_w_mag = {}  # 保留权重的原始幅度
        self.ranking = {}  # 权重的排名
        self.pruned_wg_L1 = {}  # L1剪枝后的权重组
        self.all_layer_finish_pick = False  # 所有层是否完成选择
        self.w_abs = {}  # 权重的绝对值
        self.mag_reg_log = {}  # 幅值正则化日志

        self.current_mask = passer.current_mask # user传递当前模型的mask

        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                # 初始化正则化项
                self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()

        # 深拷贝正则化项，用于更新和比较
        self.reg_ = copy.deepcopy(self.reg)
    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg != 'weight': # weight is too slow
            self._update_mag_ratio(m, name, self.w_abs[name])
        
        pruned = self.pruned_wg[name]
        if self.args.RST_schedule == 'x':
            if self.args.wg == "channel":
                self.reg[name][:, pruned] += self.args.reg_granularity_prune
            elif self.args.wg == "filter":
                self.reg[name][pruned, :] += self.args.reg_granularity_prune
            elif self.args.wg == 'weight':
                self.reg[name][pruned] += self.args.reg_granularity_prune
            else:
                raise NotImplementedError

        if self.args.RST_schedule == 'x^2':
            if self.args.wg == 'weight':
                self.reg_[name][pruned] += self.args.reg_granularity_prune
                self.reg[name][pruned] = self.reg_[name][pruned]**2
            else:
                raise NotImplementedError

        if self.args.RST_schedule == 'x^3':
            if self.args.wg == 'weight':
                self.reg_[name][pruned] += self.args.reg_granularity_prune
                self.reg[name][pruned] = self.reg_[name][pruned]**3
            else:
                raise NotImplementedError

        # 如果权重类型为'weight'，不使用幅值比率条件检查
        if self.args.wg == 'weight':
            finish_update_reg = False
        else:
            # 默认设置为完成更新状态
            finish_update_reg = True
            # 遍历历史幅值比率
            for k in self.hist_mag_ratio:
                # 如果任何一层的幅值比率低于设定阈值，则标记为未完成更新
                if self.hist_mag_ratio[k] < self.args.mag_ratio_limit:
                    finish_update_reg = False
        # 返回是否结束更新的判断，条件是是否已经完成更新或者任何一层的正则化参数的最大值超过上限
        return finish_update_reg or self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self):
        # 遍历模型中的所有模块
        for name, m in self.model.named_modules():
            # 检查模块是否为卷积层或线性层
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 获取当前层的索引
                cnt_m = self.layers[name].layer_index
                # 获取当前层的修剪率
                pr = self.pr[name]

                # 检查是否完成了当前层的正则化更新
                if name in self.iter_update_reg_finished.keys():
                    continue

                # 在特定的迭代间隔打印更新状态
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("[%d] Update reg for layer '%s'. Pr = %s. Iter = %d"
                                  % (cnt_m, name, pr, self.total_iter))

                # 获取当前层的重要性得分（本例中使用L1范数）
                self.w_abs[name] = self._get_score(m)

                # 更新正则化函数，包括更新这一层的正则化以及判断是否停止更新
                if self.args.method == "RST":
                    finish_update_reg = self._greg_1(m, name)
                else:
                    self.logprint("Wrong '--method' argument, please check.")
                    exit(1)

                # 检查当前层的修剪状态
                if finish_update_reg:
                    # 如果完成了更新正则化阶段，保持当前的正则化以稳定权重大小
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # 检查所有层是否都完成了正则化更新
                    self.prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, nn.Conv2d) or isinstance(mm, nn.Linear):
                            if n not in self.iter_update_reg_finished:
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint(
                            "==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
                        self._save_model(mark='just_finished_update_reg')

                # 更新完正则化后，打印以检查
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("    reg_status: min = %.5f ave = %.5f max = %.5f" %
                                  (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))



    def _resume_prune_status(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.model = state['model'].cuda()
        self.model.load_state_dict(state['state_dict'])
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune, 
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        self.optimizer.load_state_dict(state['optimizer'])
        self.prune_state = state['prune_state']
        self.total_iter = state['iter']
        self.iter_stabilize_reg = state.get('iter_stabilize_reg', math.inf)
        self.reg = state['reg']
        self.hist_mag_ratio = state['hist_mag_ratio']

    def _save_model(self, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'prune_state': self.prune_state, # we will resume prune_state
                'arch': self.args.arch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'iter_stabilize_reg': self.iter_stabilize_reg,
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': self.optimizer.state_dict(),
                'reg': self.reg,
                'hist_mag_ratio': self.hist_mag_ratio,

        }
        self.save(state, is_best=False, mark=mark)

    def prune(self):
        # 设置模型为训练模式
        self.model = self.model.train()
        # 配置SGD优化器
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.args.lr_pick if self.args.__dict__.get(
                                       'AdaReg_only_picking') else self.args.lr_prune,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)

        # 从保存点恢复模型、优化器状态以及剪枝状态
        self.total_iter = -1
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)  # 恢复剪枝状态
            self._get_kept_wg_L1()  # 从恢复的模型获取保留和剪枝的权重组
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                self.args.resume_path, self.total_iter, self.prune_state))

        acc1 = acc5 = 0
        # 计算正则化更新的总迭代次数
        '''
        self.args.reg_upper_limit:
        这个参数定义了正则化力度的上限。它可能是一个阈值，超过这个阈值之后不再增加正则化的力度。
        self.args.reg_granularity_prune:
        此参数定义了正则化力度的增长粒度。它控制了正则化力度在达到上限之前的调整细节。
        self.args.update_reg_interval:
        这个参数指定了正则化力度更新的间隔。也就是说，每隔多少次迭代更新一次正则化力度。
        self.args.stabilize_reg_interval:
        此参数定义了在最终稳定正则化力度之前的额外迭代次数。这是在完成所有正则化更新后用于进一步稳定模型的迭代次数。
        '''
        total_iter_reg = self.args.reg_upper_limit / self.args.reg_granularity_prune * self.args.update_reg_interval + self.args.stabilize_reg_interval
        timer = Timer(total_iter_reg / self.args.print_interval)
        while True:
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter

                # 在特定间隔进行模型测试
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (before update) [prune_state = %s, method = %s]" %
                                  (acc1, acc5, total_iter, self.prune_state, self.args.method))

                # 在特定间隔保存模型
                if total_iter % self.args.save_interval == 0:
                    self._save_model(acc1, acc5)
                    self.logprint('Periodically save model done. Iter = {}'.format(total_iter))

                if total_iter % self.args.print_interval == 0:
                    self.logprint("")
                    self.logprint("Iter = %d [prune_state = %s, method = %s] "
                                  % (total_iter, self.prune_state, self.args.method) + "-" * 40)

                # 前向传播
                self.model.train()
                y_ = self.model(inputs)

                # 如果当前处于更新正则化状态，并且达到更新间隔，则更新正则化
                if self.prune_state == "update_reg" and total_iter % self.args.update_reg_interval == 0:
                    self._update_reg()

                # 计算损失，并进行反向传播
                loss = self.criterion(y_, targets)
                self.optimizer.zero_grad()
                loss.backward()

                # 在更新前应用正则化到梯度
                self.apply_reg()
                self.optimizer.step()

                # 日志打印
                if total_iter % self.args.print_interval == 0:
                    # 检查批归一化层状态
                    if self.args.verbose:
                        for name, m in self.model.named_modules():
                            if isinstance(m, nn.BatchNorm2d):
                                # 获取此BN层关联的卷积层
                                ix = self.all_layers.index(name)
                                for k in range(ix - 1, -1, -1):
                                    if self.all_layers[k] in self.layers:
                                        last_conv = self.all_layers[k]
                                        break
                                mask_ = [0] * m.weight.data.size(0)
                                for i in self.kept_wg[last_conv]:
                                    mask_[i] = 1
                                wstr = ' '.join(['%.3f (%s)' % (x, y) for x, y in zip(m.weight.data, mask_)])
                                bstr = ' '.join(['%.3f (%s)' % (x, y) for x, y in zip(m.bias.data, mask_)])
                                logstr = f'{last_conv} BN weight: {wstr}\nBN bias: {bstr}'
                                self.logprint(logstr)

                    # 检查训练准确率
                    _, predicted = y_.max(1)
                    correct = predicted.eq(targets).sum().item()
                    train_acc = correct / targets.size(0)
                    self.logprint("After optim update current_train_loss: %.4f current_train_acc: %.4f" % (loss.item(), train_acc))

                # 在特定状态和间隔后改变剪枝状态
                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    # --- 检查准确率以确认 '_prune_and_build_new_model' 方法正常工作
                    # 已检查，正常工作！
                    '''
                    for name, m in self.model.named_modules():
                        if isinstance(m, self.learnable_layers):
                            # 获取被剪枝的滤波器
                            pruned_filter = self.pruned_wg[name]
                            # 将这些滤波器的权重设置为0
                            m.weight.data[pruned_filter] *= 0
                            # 找到与当前层相邻的批归一化层
                            next_bn = self._next_bn(self.model, m)
                        elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                            # 将批归一化层中对应的权重和偏置也设置为0
                            m.weight.data[pruned_filter] *= 0
                            m.bias.data[pruned_filter] *= 0

                    # 在剪枝操作前测试模型的准确率
                    acc1_before, *_ = self.test(self.model)
                    # 执行剪枝并重建模型
                    self._prune_and_build_new_model()
                    # 在剪枝操作后测试模型的准确率
                    acc1_after, *_ = self.test(self.model)
                    # 打印剪枝前后的准确率
                    print(acc1_before, acc1_after)
                    # 退出程序
                    exit()
                    # ---
                    '''
                    model_before_removing_weights = copy.deepcopy(self.model)
                    self._prune_and_build_new_model()
                    self.logprint("'stabilize_reg' is done. Pruned, go to 'finetune'. Iter = %d" % total_iter)
                    return model_before_removing_weights, copy.deepcopy(self.model)

                # 使用计时器对象估算并打印剩余训练时间
                if total_iter % self.args.print_interval == 0:
                    self.logprint(f"predicted_finish_time of reg: {timer()}")

    def _plot_mag_ratio(self, w_abs, name):
        fig, ax = plt.subplots()
        max_ = w_abs.max().item()
        w_abs_normalized = (w_abs / max_).data.cpu().numpy()
        ax.plot(w_abs_normalized)
        ax.set_ylim([0, 1])
        ax.set_xlabel('filter index')
        ax.set_ylabel('relative L1-norm ratio')
        layer_index = self.layers[name].layer_index
        shape = self.layers[name].size
        ax.set_title("layer %d iter %d shape %s\n(max = %s)" 
            % (layer_index, self.total_iter, shape, max_))
        # out = pjoin(self.logger.logplt_path, "%d_iter%d_w_abs_dist.jpg" %
        #                         (layer_index, self.total_iter))
        # fig.savefig(out)
        # plt.close(fig)
        # np.save(out.replace('.jpg', '.npy'), w_abs_normalized)
