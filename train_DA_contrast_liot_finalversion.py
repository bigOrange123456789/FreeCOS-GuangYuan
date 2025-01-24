# from __future__ import division
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import config
from network import Network, Network_UNet, SingleUNet, Single_IBNUNet, Single_contrast_UNet

from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, CosinLR
from utils.evaluation_metric import computeF1, compute_allRetinal
from Datasetloader.dataset import CSDataset
from common.logger import Logger
import csv
from utils.loss_function import DiceLoss, Contrastloss, ContrastRegionloss, ContrastRegionloss_noedge, \
    ContrastRegionloss_supunsup, ContrastRegionloss_NCE, ContrastRegionloss_AllNCE, ContrastRegionloss_quaryrepeatNCE, Triplet
from base_model.discriminator import PredictDiscriminator, PredictDiscriminator_affinity

def create_csv(path, csv_head):
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)

def check_feature(sample_set_sup, sample_set_unsup):
    """
    sample_sets_sup：  有监督合成图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
    sample_sets_unsup：无监督真实图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
    feature N,dims，Has bug or debuff because of zeros
    """
    flag = True
    Single = False
    queue_len = 500
    # sample_set_sup['sample_easy_pos'], sample_set_sup['sample_easy_neg'], sample_set_unsup['sample_easy_pos'], sample_set_unsup['sample_easy_neg']
    with torch.no_grad(): #不进行梯度计算
        if (    'sample_easy_pos' not in sample_set_sup.keys() or
                'sample_easy_neg' not in sample_set_unsup.keys() or
                'sample_easy_pos' not in sample_set_unsup.keys()):
            flag = False
            quary_feature = None
            pos_feature = None
            neg_feature = None
        else:
            quary_feature = sample_set_sup['sample_easy_pos']   #合成图像 全部易正样本的特征
            pos_feature   = sample_set_unsup['sample_easy_pos'] #真实图像 全部易正样本的特征
            flag = True
            '''
                sample_easy_pos [2000, 64]
                quary_feature.shape=[2000, 64]
                pos_feature.shape  =[579,  64] 真实图像中血管所占的像素数量可能过少？
            '''

        if 'sample_easy_neg' in sample_set_sup.keys() and 'sample_easy_neg' in sample_set_unsup.keys():
            neg_unlabel = sample_set_unsup['sample_easy_neg'] #合成图像 全部易负样本的特征 #shape=[212, 64] #合成图像中的背景像素过少？
            neg_label   = sample_set_sup['sample_easy_neg']   #真实图像 全部易负样本的特征 #shape=[2000, 64]

            neg_feature = torch.cat((neg_unlabel[:min(queue_len // 2, neg_unlabel.shape[0]), :],
                                     neg_label[:min(queue_len // 2, neg_label.shape[0]), :]), dim=0)
            # neg_feature.shape:[212+250,64]=[462, 64]
            '''
            负样本特征,形状是(N, D)，其中N是样本数量，D是特征维度。
            queue_len：这是一个预先定义的变量，代表想要拼接的张量的最大长度。
            min(queue_len // 2, neg_unlabel.shape[0])：要取出的最大样本数量。
            沿着第0维拼接起来，但拼接的数量受到queue_len的限制。
            '''
    return quary_feature, pos_feature, neg_feature, flag
    # quary_feature: 合成图像 全部易正样本的特征 [<=2000, 64]=[2000, 64]
    # pos_feature:   真实图像 全部易正样本的特征 [<=2000, 64]=[579,  64]
    # neg_feature:   合成和真实的图像 部分易负样本的特征 [<=500, 64]=[462, 64]


def train(epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
          optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
          average_negregion):
    '''
    epoch,                      已完成的批次数     0
    Segment_model,              分割模型
    predict_Discriminator_model,判别器模型
    dataloader_supervised,      有监督数据集
    dataloader_unsupervised,    无监督数据集
    optimizer_l, optimizer_D,   优化器
    lr_policy, lrD_policy,      学习率调整的策略    <engine.lr_policy.WarmUpPolyLR>
    criterion,                  评价标准           DiceLoss() type=<utils.loss_function.DiceLoss>
    total_iteration,            总迭代次数         总epoch数=nepochs * niters_per_epoch
    average_posregion,          平均正区域         torch.zeros((1, 128)),暂时不知道这个对象的作用
    average_negregion
    '''
    if torch.cuda.device_count() > 1: #如果有多个CUDA
        Segment_model.module.train()
        predict_Discriminator_model.module.train()
        '''
            .module: 这个属性通常在模型被封装或复制时出现。
                并行化处理后，原始模型会被封装在一个新的对象中，而这个新对象会有一个.module属性指向原始的模型。
            .train(): 将模型设置为训练模式。
        '''
    else: #将模型设置为cuda模式
        print("start_model_train")
        Segment_model.train()
        predict_Discriminator_model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
    '''进度条
            range(config.niters_per_epoch):
                这部分代码生成一个迭代器，其范围是从0到config.niters_per_epoch（不包括config.niters_per_epoch）。
                niters_per_epoch表示每个epoch（一轮训练）中的迭代次数。
            file=sys.stdout:
                这个参数指定了进度条信息输出的目标文件对象。sys.stdout代表标准输出流，即命令行界面或终端。
            bar_format=bar_format:
                bar_format是一个字符串，用于自定义进度条的显示格式。
                这个字符串包含了特殊的占位符，tqdm会将这些占位符替换为实际的进度信息。
                允许用户根据需要定制进度条的外观，比如显示百分比、预计剩余时间等。
    '''
    dataloader = iter(dataloader_supervised) # 有监督的数据加载器
    unsupervised_dataloader = iter(dataloader_unsupervised) # 无监督的数据加载器
    bce_loss = nn.BCELoss()
    sum_loss_seg = 0
    sum_adv_loss = 0
    sum_Dsrc_loss = 0
    sum_Dtar_loss = 0
    sum_totalloss = 0
    sum_contrastloss = 0
    sum_celoss = 0
    sum_diceloss = 0
    source_label = 0 # 源数据域:合成血管 #用于对抗学习，源数据域的标签
    target_label = 1 #目标数据域:真实血管 #对抗学习的目的是让神经网络能够屏蔽数据域的差异，让两个数据域都能输出正确的结果
    criterion_contrast = ContrastRegionloss_quaryrepeatNCE()
    print('begin train')
    ''' supervised part '''
    for idx in pbar:
        current_idx = epoch * config.niters_per_epoch + idx
        damping = (1 - current_idx / total_iteration) # 剩余工作量(damping本意指阻尼)
        start_time = time.time()
        optimizer_l.zero_grad()
        optimizer_D.zero_grad()
        '''
            优化器（Optimizer）：根据梯度来更新的参数。
            反向传播（Backpropagation）：计算每个参数的梯度。
            -----
            .zero_grad()：将模型参数的梯度清零。
            默认情况下，在一个迭代（batch）中多次调用反向传播，梯度会被累加。
            因此需要清零梯度，以确保梯度计算不会受到之前迭代梯度的影响。
        '''
        try:
            minibatch = next(dataloader) #获取一个batch的有监督数据
        except StopIteration: #如果加载完一个epoch，就重新初始化加载器
            dataloader = iter(dataloader_supervised)
            minibatch = next(dataloader)

        imgs = minibatch['img']       #imgs: [4, 4, 256, 256] #每个batch有4张图片
        gts  = minibatch['anno_mask'] #gts:  [4, 1, 256, 256]
        imgs = imgs.cuda(non_blocking=True)
        gts  = gts.cuda(non_blocking=True)
        '''
            non_blocking=True：将数据移至GPU时，使用异步操作。
                这意味着数据移动操作不会阻塞（即等待数据完全传输完毕）当前线程的执行。
                这可以允许程序在等待数据传输的同时执行其他操作，从而提高程序的总体效率。
                然而，需要注意的是，如果后续操作立即依赖于这些数据，并且数据尚未完全传输到GPU，则可能会导致未定义行为或错误。
        '''
        with torch.no_grad(): #禁用梯度计算
        # 这个上下文管理器用于暂时禁用梯度计算。在这个代码块内部执行的所有操作都不会被记录在PyTorch的计算图中，因此不会计算梯度。
        # 这通常用于推理（inference）或评估（evaluation）阶段，以减少内存消耗并加速计算。
            weight_mask = gts.clone().detach()
            # 这行代码克隆了gts张量，克隆是为了避免直接修改原始张量。
            # detach()方法分离了克隆的张量，使其不再与计算图相关联。
            # 分离是为了确保后续操作不会影响梯度计算（尽管在这个torch.no_grad()上下文中梯度计算已被禁用）。
            weight_mask[weight_mask == 0] = 0.1 # 值为0的元素设为0.1
            weight_mask[weight_mask == 1] = 1   # 值为1的元素保持不变
            criterion_bce = nn.BCELoss(weight=weight_mask)
            '''
                血管区域太小，背景区域太大。因此分别给这两个区域使用了不同的权重。
                weight (Tensor, optional) : 给定到每个批次元素的损失的手动重新缩放权重。如果给定，则必须是大小为 nbatch 的张量。
                这意味着在计算损失时，不同类别的样本将对总损失有不同的贡献，这有助于处理类别不平衡问题。
            '''
        try: #获取一个batch的无监督数据
            unsup_minibatch = next(unsupervised_dataloader)
            # 这段代码尝试从unsupervised_dataloader迭代器中获取下一个数据批次。
        except StopIteration: #如果迭代器已经耗尽（引发了StopIteration异常）
            unsupervised_dataloader = iter(dataloader_unsupervised) # 则重新初始化迭代器
            unsup_minibatch = next(unsupervised_dataloader) # 再次尝试获取下一个数据批次

        unsup_imgs = unsup_minibatch['img']
        # unsup_minibatch:{
        #   'img_name': ['**.png', ...],
        #   'img':tensor
        # }
        unsup_imgs = unsup_imgs.cuda(non_blocking=True) # unsup_imgs:[4, 4, 256, 256]

        # Start train fake vessel 开始训练假血管
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False #不在反向传播过程中计算其梯度。
        pred_sup_l,  sample_set_sup,   flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        # pred_sup_l： 类0-1标签的预测结果
        # sample_sets：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        loss_ce = 0.1 * criterion_bce(pred_sup_l, gts) # 根据预测结果和标签计算CE损失 # retinal是5 XCAD是0.1 crack是5 # For retinal :5 For XCAD:0.1 5 for crack
        loss_dice = criterion(pred_sup_l, gts) # 根据预测结果和标签计算Dice损失
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True, fake=False) #mask影响正负样本的获取
        D_seg_target_out = predict_Discriminator_model(pred_target) #计算对抗损失 #判别是否为 直线合成血管 or 曲线标注血管
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out), #无监督曲线标注血管的预测结果
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())
        quary_feature, pos_feature, neg_feature, flag = check_feature(sample_set_sup, sample_set_unsup)
        '''
        in:
            sample_sets_sup：  有监督合成图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
            sample_sets_unsup：无监督真实图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        out:
            quary_feature: 合成图像 全部易正样本的特征 [<=2000, 64]=[2000, 64]
            pos_feature:   真实图像 全部易正样本的特征 [<=2000, 64]=[579,  64]
            neg_feature:   合成和真实的图像 部分易负样本的特征 [<=500, 64]=[462, 64]
        '''

        if flag:
            loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
        else:
            loss_contrast = 0
        weight_contrast = 0.04  # 0.04 for NCE allpixel/0.01maybe same as dice
        loss_seg = loss_dice + loss_ce
        sum_loss_seg += loss_seg.item()
        loss_contrast_sum = weight_contrast * (loss_contrast)
        sum_contrastloss += loss_contrast_sum

        loss_adv = (loss_adv_target * damping) / 4 + loss_dice + loss_ce + weight_contrast * (loss_contrast)
        loss_adv.backward(retain_graph=False)
        loss_adv_sum = (loss_adv_target * damping) / 4
        sum_adv_loss += loss_adv_sum.item()

        sum_celoss += loss_ce
        sum_diceloss += loss_dice.item()
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = True
        pred_sup_l = pred_sup_l.detach()
        D_out_src = predict_Discriminator_model(pred_sup_l)

        loss_D_src = bce_loss(F.sigmoid(D_out_src),
                              torch.FloatTensor(D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 8
        loss_D_src.backward(retain_graph=False)
        sum_Dsrc_loss += loss_D_src.item()

        pred_target = pred_target.detach()
        D_out_tar = predict_Discriminator_model(pred_target)

        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda())
        loss_D_tar = loss_D_tar / 8  # bias
        loss_D_tar.backward(retain_graph=False)
        sum_Dtar_loss += loss_D_tar.item()
        optimizer_l.step()
        optimizer_D.step()

        lr = lr_policy.get_lr(current_idx)  # lr change
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_l.param_groups[1]['lr'] = lr
        # for i in range(2, len(optimizer_l.param_groups)):   没用
        #     optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D
        # for i in range(2, len(optimizer_D.param_groups)):  没用
        #     optimizer_D.param_groups[i]['lr'] = Lr_D

        sum_contrastloss += loss_contrast_sum
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss_seg=%.4f' % loss_seg.item() \
                    + ' loss_D_tar=%.4f' % loss_D_tar.item() \
                    + ' loss_D_src=%.4f' % loss_D_src.item() \
                    + ' loss_adv=%.4f' % loss_adv.item() \
                    + 'loss_ce=%.4f' % loss_ce \
                    + 'loss_dice=%.4f' % loss_dice.item() \
                    + 'loss_contrast=%.4f' % loss_contrast_sum

        sum_totalloss = sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
        pbar.set_description(print_str, refresh=False)

        end_time = time.time()

    train_loss_seg = sum_loss_seg / len(pbar)
    train_loss_Dtar = sum_Dtar_loss / len(pbar)
    train_loss_Dsrc = sum_Dsrc_loss / len(pbar)
    train_loss_adv = sum_adv_loss / len(pbar)
    train_loss_ce = sum_celoss / len(pbar)
    train_loss_dice = sum_diceloss / len(pbar)
    train_loss_contrast = sum_contrastloss / len(pbar)
    train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
    return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion


# evaluate(epoch, model, dataloader_val,criterion,criterion_consist)
def evaluate(epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    if torch.cuda.device_count() > 1:
        Segment_model.module.eval()
        predict_Discriminator_model.module.eval()
    else:
        Segment_model.eval()
        predict_Discriminator_model.eval()
    with torch.no_grad():
        val_sum_loss_sup = 0
        val_sum_f1 = 0
        val_sum_pr = 0
        val_sum_re = 0
        val_sum_sp = 0
        val_sum_acc = 0
        val_sum_jc = 0
        val_sum_AUC = 0
        F1_best = 0
        print('begin eval')
        ''' supervised part '''
        for val_idx, minibatch in enumerate(val_target_loader):
            start_time = time.time()
            val_imgs = minibatch['img']
            val_gts = minibatch['anno_mask']
            val_imgs = val_imgs.cuda(non_blocking=True)
            val_gts = val_gts.cuda(non_blocking=True)
            # NCHW
            val_pred_sup_l, sample_set_unsup, _ = Segment_model(val_imgs, mask=None, trained=False, fake=False)

            max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0)
            val_max_l = max_l.float()
            val_loss_sup = criterion(val_pred_sup_l, val_gts)

            current_validx = epoch * config.niters_per_epoch + val_idx
            val_loss = val_loss_sup
            val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC = compute_allRetinal(val_max_l,
                                                                                                     val_pred_sup_l,
                                                                                                     val_gts)
            val_sum_loss_sup += val_loss_sup.item()
            val_sum_f1 += val_f1
            val_sum_pr += val_precision
            val_sum_re += val_recall
            val_sum_AUC += val_AUC
            val_sum_sp += val_Sp
            val_sum_acc += val_Acc
            val_sum_jc += val_jc
            end_time = time.time()
        val_mean_f1 = val_sum_f1 / len(val_target_loader)
        val_mean_pr = val_sum_pr / len(val_target_loader)
        val_mean_re = val_sum_re / len(val_target_loader)
        val_mean_AUC = val_sum_AUC / len(val_target_loader)
        val_mean_acc = val_sum_acc / len(val_target_loader)
        val_mean_sp = val_sum_sp / len(val_target_loader)
        val_mean_jc = val_sum_jc / len(val_target_loader)
        val_loss_sup = val_sum_loss_sup / len(val_target_loader)
        return val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup


def main():
    # os.getenv('debug'): None
    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser()
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口

    args = parser.parse_args()
    cudnn.benchmark = True #benchmark的意思是基准
    # set seed
    seed = config.seed  # 12345
    torch.manual_seed(seed) # manual_seed的意思是人工种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")

    CSDataset.initialize(datapath=config.datapath)
    print('config.datapath:',config.datapath)#"./Data/XCAD"

    dataloader_supervised = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT # benchmark的本译是基准
                                                       config.batch_size, # 4
                                                       config.nworker,    # 8
                                                       'train',
                                                       config.img_mode,   # crop
                                                       config.img_size,   # 256
                                                       'supervised')
    # 有监督使用的数据是什么？gray中为血管的灰度图、gt中为标签。
    dataloader_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT
                                                         config.batch_size, # 4
                                                         config.nworker,    # 8
                                                         'train',
                                                         config.img_mode,   # crop
                                                         config.img_size,   # 256
                                                         'unsupervised')
    # 无监督使用的应该是img文件夹中的原始图片。

    dataloader_val = CSDataset.build_dataloader(config.benchmark,       # XCAD_LIOT
                                                config.batch_size_val,  # 1
                                                config.nworker,         # 8
                                                'val',
                                                'same',
                                                None,
                                                'supervised')
    # 无监督使用的应该是img文件夹中的原始图片。
    print("Dataloader.....")
    criterion = DiceLoss()  # try both loss BCE and DICE # 尝试损失BCE和DICE
    # dice(x,y) = 1 - 2 * (x*y) / (x+y) # 相同为-1,不同为1

    # define and init the model # 定义并初始化模型
    # Single or not single # 单个或非单个
    BatchNorm2d = nn.BatchNorm2d # <BatchNorm2d>
    Segment_model = Single_contrast_UNet(4, config.num_classes) # config.num_classes=1

    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_,
                # nn.init.kaiming_normal_: <function kaiming_normal_>
                BatchNorm2d,        # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
                config.bn_eps,      # config.bn_eps: 1e-05
                config.bn_momentum, # config.bn_momentum: 0.1
                mode='fan_in', nonlinearity='relu')
    # define the learning rate
    base_lr = config.lr      # 0.04 # 学习率
    base_lr_D = config.lr_D  # 0.04 # dropout?

    params_list_l = []
    params_list_l = group_weight(
        params_list_l, #一个list对象，内部的成员为tensor对象
        Segment_model.backbone, # 分割网络的主干
        BatchNorm2d,    # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
        base_lr)        # base_lr: 0.01
    # optimizer for segmentation_L   # 分割优化器_L
    print("config.weight_decay", config.weight_decay)
    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),
                                   lr=base_lr_D, betas=(0.9, 0.99))

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch  # nepochs=137  niters=C.max_samples // C.batch_size
    print("total_iteration", total_iteration)
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    lrD_policy = WarmUpPolyLR(base_lr_D, config.lr_power, total_iteration,
                              config.niters_per_epoch * config.warm_up_epoch)

    average_posregion = torch.zeros((1, 128)) # average_posregion ：平均正区？
    average_negregion = torch.zeros((1, 128)) # average_negregion ：平均负区？
    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.device_count() > 1:
        Segment_model = Segment_model.cuda()
        Segment_model = nn.DataParallel(Segment_model)
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda()
        predict_Discriminator_model = nn.DataParallel(predict_Discriminator_model)
        # Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda() # 预测判别模型，我猜这是判别器
    else:
        Segment_model = Segment_model
        predict_Discriminator_model = predict_Discriminator_model

    best_val_f1 = 0
    best_val_AUC = 0
    Logger.initialize(config, training=True)
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1.csv'
    csv_head = ["epoch", "total_loss", "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"]
    create_csv(val_score_path, csv_head)
    for epoch in range(config.state_epoch, config.nepochs): #按照预先设定的回合数量执行，似乎不会提前中止
        # train_loss_sup, train_loss_consis, train_total_loss
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion \
            = train(#训练
                epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
                optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
                average_negregion)
        print("train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},train_loss_contrast:{}".format(
                train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss,
                train_loss_contrast))
        print("train_loss_dice:{},train_loss_ce:{}".format(train_loss_dice, train_loss_ce))
        # val_mean_f1, val_mean_pr, val_mean_re, val_mean_f1, val_mean_pr, val_mean_re,val_loss_sup
        val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup \
            = evaluate(#验证
                epoch, Segment_model, predict_Discriminator_model, dataloader_val,
                criterion)  # evaluate(epoch, model, val_target_loader,criterion, criterion_cps)
        # val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re,val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup
        data_row_f1score = [str(epoch), str(train_total_loss), str(val_mean_f1.item()), str(val_mean_AUC),
                            str(val_mean_pr.item()), str(val_mean_re.item()), str(val_mean_acc), str(val_mean_sp),
                            str(val_mean_jc)]
        print("val_mean_f1",  val_mean_f1.item())
        print("val_mean_AUC", val_mean_AUC)
        print("val_mean_pr",  val_mean_pr.item())
        print("val_mean_re",  val_mean_re.item())
        print("val_mean_acc", val_mean_acc.item())
        write_csv(val_score_path, data_row_f1score)
        if val_mean_f1 > best_val_f1:
            best_val_f1 = val_mean_f1
            Logger.save_model_f1_S(Segment_model, epoch, val_mean_f1, optimizer_l)
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_f1, optimizer_D)
        # if val_mean_AUC > best_val_AUC:
        #     best_val_AUC = val_mean_AUC
        #     Logger.save_model_f1_S(Segment_model, epoch, val_mean_AUC, optimizer_l)
        #     Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_AUC, optimizer_D)

if __name__ == '__main__':
    main()

'''
    2. 训练脚本
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS
    python train_DA_contrast_liot_finalversion.py 
    #(CUDA_VISIBLE_DEVICES=0 python train_DA_contrast_liot_DRIVE_finalversion.py for DRIVE)
'''
