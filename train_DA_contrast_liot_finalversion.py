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

import numpy as np
from PIL import Image
from skimage import measure

def asymmetric_loss(x, y, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
# def asymmetric_loss(x, y, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
    gamma_neg=config.gamma_neg
    gamma_pos=config.gamma_pos
    """"
    用于计算不平衡损失
    源自谢驰师哥的代码：https://github.com/charles-xie/CQL
    Parameters
    ----------
    x: input logits     [4, 1, 256, 256]
    tensor([[[[0.7753, 0.7117, 0.5816,  ...,
    y: targets (multi-label binarized vector)   [4, 1, 256, 256]
    tensor([[[[0., 0., 0.,  ...,
    """
    # criterion_bce(x=pred_sup_l, y=gts) # 根据预测结果x和标签y计算CE损失
    pos_inds = y.eq(1).float() #转换为布尔型,再转换为浮点型
    num_pos = pos_inds.float().sum() #标签中血管所占像素的总数
    '''
    x.shape=[4, 1, 256, 256]
    y.shape=[4, 1, 256, 256]
    num_pos=tensor(19957.)
    '''

    # Calculating Probabilities
    # x_sigmoid = torch.sigmoid(x)
    x_sigmoid = x  # x.shape=[4, 1, 256, 256]
    xs_pos = x_sigmoid #血管概率
    xs_neg = 1 - x_sigmoid #背景概率
    # print(xs_pos.shape, xs_neg.shape,xs_neg)

    # Asymmetric Clipping 不对称剪裁(这个操作的作用是什么？)
    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1) # 将所有背景概率都增大clip(5%)

    # Basic CE calculation
    los_pos = y * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        if disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False) # 接下来不去计算梯度
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True) # 接下来恢复计算梯度
        loss *= one_sided_w
    # loss.shape=[4, 1, 256, 256] <torch.Tensor>

    if num_pos == 0:
        return -loss.sum()
    else:
        return -loss.sum() / num_pos

def bce_loss_lzc(x, y, eps=1e-8):

    # Calculating Probabilities
    # x_sigmoid = torch.sigmoid(x)
    x_sigmoid = x  # x.shape=[4, 1, 256, 256]
    xs_pos = x_sigmoid #血管概率
    xs_neg = 1 - x_sigmoid #背景概率
    # print(xs_pos.shape, xs_neg.shape,xs_neg)

    # Basic CE calculation
    los_pos = y * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]

    return -loss.mean()
class BCELoss_lzc(nn.Module):
    def __init__(
        self,
        weight=None,
        eps=1e-8,
        gamma_neg=0,
        gamma_pos=0,
        disable_torch_grad_focal_loss=True
    ):
        super(BCELoss_lzc, self).__init__()
        self.weight=weight
        self.eps=eps
        self.gamma_neg=gamma_neg #用于Focal Loss
        self.gamma_pos=gamma_pos #用于Focal Loss
        self.disable_torch_grad_focal_loss=disable_torch_grad_focal_loss #用于Focal Loss

    def forward(self, x , y ) :
        xs_pos = x #血管概率
        xs_neg = 1 - x #背景概率

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]


        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False) # 接下来不去计算梯度
            pt = xs_neg * y  + xs_pos * (1 - y)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow( pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True) # 接下来恢复计算梯度
            loss *= one_sided_w

        if not self.weight is None:#原本BCE自带的加权功能
            loss=loss*self.weight

        return -loss.mean()

class ConnectivityAnalyzer:
    def __init__(
            self,
            mask_tensor
    ):
        self.mask_tensor=mask_tensor
        self.allObj=torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
                                  torch.zeros_like(mask_tensor))
        self.mainObj=self.getMainObj(self.allObj)
    def connectivityLoss(self):
        # score_all=mask_tensor*self.getAllObj(mask_tensor)
        # score_main=mask_tensor*self.getMainObj(mask_tensor)
        score_all = self.mask_tensor * self.allObj
        score_main = self.mask_tensor * self.mainObj
        def compute(m):
            # 计算每张图片的像素和
            # 由于每张图片是单通道的，我们直接对最后一个两个维度求和
            pixel_sums = m.sum(dim=(2, 3))  # shape 将变为 [4, 1]

            # 由于 pixel_sums 的形状是 [4, 1]，我们可以通过 squeeze() 方法去掉单通道维度
            # 这不是必需的，但可以使后续操作更清晰
            pixel_sums_squeezed = pixel_sums.squeeze(1)  # shape 变为 [4]

            # 计算所有图片像素和的平均值
            return pixel_sums_squeezed.mean()  # 得到一个标量
        score_all  = compute(score_all)
        score_main = compute(score_main)
        eps=1e-8
        return score_all/(score_main+eps)
    def getAllObj(self, mask_tensor):
        return torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
                                     torch.zeros_like(mask_tensor))

    def getMainObj(self, mask_tensor):
        mask_tensor=mask_tensor.cpu()
        # mask_tensor = torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
        #                              torch.zeros_like(mask_tensor))

        # 将PyTorch张量转换为NumPy数组，保持单通道维度
        mask_array = mask_tensor.numpy().astype(np.uint8)

        # 创建一个空列表来存储处理后的MASK，保持与输入相同的shape
        processed_masks = []

        # 遍历每张MASK图片（保持单通道维度）
        for mask in mask_array:
            # 挤压掉单通道维度以进行连通性检测，但之后要恢复
            mask_squeeze = mask.squeeze()
            if mask_squeeze.sum()==0:#这个对象为空
                processed_masks.append(mask)
                continue

            # 进行连通性检测，返回标记数组和连通区域的数量
            labeled_array, num_features = measure.label(mask_squeeze, connectivity=1, return_num=True)

            # 创建一个字典来存储每个标签的像素数
            region_sizes = {}
            for region in range(1, num_features + 1):
                # 计算每个连通区域的像素数
                region_size = np.sum(labeled_array == region)
                region_sizes[region] = region_size

            # 找到像素数最多的连通区域及其标签
            max_region = max(region_sizes, key=region_sizes.get)

            # 创建一个新的MASK，只保留像素数最多的连通区域，并恢复单通道维度
            processed_mask = np.zeros_like(mask)
            processed_mask[0, labeled_array == max_region] = 1

            # 将处理后的MASK添加到列表中
            processed_masks.append(processed_mask)

        # 将处理后的MASK列表转换回PyTorch张量
        processed_masks_tensor = torch.tensor(processed_masks, dtype=torch.float32)

        # 检查shape是否保持不变
        assert processed_masks_tensor.shape == mask_tensor.shape, "Processed masks tensor shape does not match original."

        if torch.cuda.is_available():# 检查CUDA是否可用
            device = torch.device("cuda")  # 创建一个表示GPU的设备对象
        else:
            device = torch.device("cpu")  # 如果没有GPU，则使用CPU

        return processed_masks_tensor.to(device)

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
          average_negregion,isFirstEpoch):
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
    else: #将模型设置为训练模式
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
        # if config.ASL:(失败了，使用这种方法准确度变为0)
        #     criterion_bce = asymmetric_loss
        with torch.no_grad(): #禁用梯度计算
            weight_mask = gts.clone().detach()
            weight_mask[weight_mask == 0] = 0.1 # 值为0的元素设为0.1
            weight_mask[weight_mask == 1] = 1   # 值为1的元素保持不变
            if True: # if config.ASL:
                criterion_bce = BCELoss_lzc(
                    weight=weight_mask,
                    gamma_pos=config.gamma_pos,
                    gamma_neg=config.gamma_neg)
            else:
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

        # Start train fake vessel 开始训练合成血管
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False #首先优化分割器、不优化判别器
        pred_sup_l,  sample_set_sup,   flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        # pred_sup_l： 类0-1标签的预测结果
        # sample_sets：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        loss_ce = 0.1 * criterion_bce(pred_sup_l, gts) # 根据预测结果和标签计算CE损失 # retinal是5 XCAD是0.1 crack是5 # For retinal :5 For XCAD:0.1 5 for crack
        loss_dice = criterion(pred_sup_l, gts) # 根据预测结果和标签计算Dice损失
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True, fake=False) #mask影响正负样本的获取
        D_seg_target_out = predict_Discriminator_model(pred_target) #计算对抗损失 #判别是否为 直线合成血管 or 曲线标注血管
        # pred_target      shape=[4, 1, 256, 256]
        # D_seg_target_out shape=[4, 1, 8, 8]
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out), #无监督曲线标注血管的预测结果
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())
        quary_feature, pos_feature, neg_feature, flag = check_feature(sample_set_sup, sample_set_unsup)
        '''
        in:
            sample_sets_sup:  有监督合成图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
            sample_sets_unsup:无监督真实图片用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        out:
            quary_feature: 合成图像 全部易正样本的特征 [<=2000, 64]=[2000, 64]
            pos_feature:   真实图像 全部易正样本的特征 [<=2000, 64]=[579,  64]
            neg_feature:   合成和真实的图像 部分易负样本的特征 [<=500, 64]=[462, 64]
        '''

        if flag:
            if hasattr(Segment_model, 'learnable_scalar'):
                loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature, Segment_model.learnable_scalar)
            else:
                loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
            '''
                quary_feature: 合成图像 全部易正样本的特征 [<=2000, 64]=[2000, 64]       ->len=579 (这里的正样本成对使用)
                pos_feature:   真实图像 全部易正样本的特征 [<=2000, 64]=[579,  64]       ->len=579 
                neg_feature:   合成和真实的图像 部分易负样本的特征 [<=500, 64]=[462, 64]   ->len=462
            '''
        else:
            loss_contrast = 0

        # if hasattr(Segment_model, 'learnable_scalar'):
        #     print('learnable_scalar1 ', Segment_model.learnable_scalar,Segment_model.learnable_scalar.grad)
        weight_contrast = 0.04  #对比损失的权重 # 0.04 for NCE allpixel/0.01maybe same as dice
        loss_seg = loss_dice + loss_ce #(当前batch的)合成监督损失
        sum_loss_seg += loss_seg.item() #(本次的epoch的)合成监督损失
        loss_contrast_sum = weight_contrast * (loss_contrast) #(当前batch的)加权后的对比损失
        sum_contrastloss += loss_contrast_sum #(本次的epoch的)加权后的对比损失

        loss_adv = (loss_adv_target * damping) / 4 + loss_dice + loss_ce + weight_contrast * (loss_contrast) #对抗+监督+对比
        if config.pseudo_label and isFirstEpoch==False:
            gts_pseudo = unsup_minibatch['anno_mask']#原始数据
            gts_pseudo = gts_pseudo.cuda(non_blocking=True)
            with torch.no_grad():  # 禁用梯度计算
                weight_mask = gts_pseudo.clone().detach()
                weight_mask[weight_mask == 0] = 0.1  # 值为0的元素设为0.1
                weight_mask[weight_mask == 1] = 1  # 值为1的元素保持不变
                criterion_bce2 = BCELoss_lzc(
                        weight=weight_mask,
                        gamma_pos=config.gamma_pos,
                        gamma_neg=config.gamma_neg)
            loss_pseudo = 0.01 * criterion_bce2(pred_target, gts_pseudo)
            loss_adv = loss_adv + loss_pseudo

        if config.connectivityLoss:#使用连通损失
            loss_conn1 = ConnectivityAnalyzer(pred_sup_l).connectivityLoss() #合成监督
            loss_conn2 = ConnectivityAnalyzer(pred_target).connectivityLoss() #无监督
            loss_conn = loss_conn1 + loss_conn2
            if config.pseudo_label and isFirstEpoch == False:
                loss_conn3 = ConnectivityAnalyzer(pred_target).connectivityLoss() #伪监督
                loss_conn = loss_conn + loss_conn3
            # print("\t",loss_adv , loss_conn)
            loss_adv = loss_adv + loss_conn*0.1

        loss_adv.backward(retain_graph=False) # #计算分割网络参数的梯度,并累加到网络参数的.grad属性中
        # if hasattr(Segment_model, 'learnable_scalar'):
        #     print('learnable_scalar2 ', Segment_model.learnable_scalar, Segment_model.learnable_scalar.grad)
        '''
            retain_graph=False：
                在默认情况下（即retain_graph=False），在计算完梯度后会释放用于计算梯度的计算图（graph）。
                这对于大多数情况来说已经足够了，因为每次参数更新后，我们通常都会重新计算损失和梯度。
                如果在同一个计算图中进行多次反向传播，设置retain_graph=True来保留计算图。
        '''
        loss_adv_sum = (loss_adv_target * damping) / 4 #(当前batch的)对抗损失
        sum_adv_loss += loss_adv_sum.item() #(本次的epoch的)对抗损失

        sum_celoss += loss_ce #(本次的epoch的)CE损失
        sum_diceloss += loss_dice.item() #(本次的epoch的)DICE损失
        for param in predict_Discriminator_model.parameters(): # 开启判别器的优化
            param.requires_grad = True #将判别器中的参数设置为需要计算梯度
        pred_sup_l = pred_sup_l.detach() #这样可以确保接下来不优化分割器
        # pred_sup_l： 类0-1标签的预测结果 shape=[4, 1, 256, 256]
        '''
            .detach()：这是PyTorch张量的一个方法，它的作用是创建一个新的张量，这个新张量与原始张量共享数据但不共享梯度历史。
            换句话说，.detach()方法会“分离”出原始张量的一个副本，这个副本在自动微分过程中不会被考虑用于梯度计算。
            这通常用于当你想要使用张量的值但不希望其梯度影响反向传播时。
        '''
        D_out_src = predict_Discriminator_model(pred_sup_l) #D_out_src.shape=[4, 1, 8, 8]

        loss_D_src = bce_loss(F.sigmoid(D_out_src), #判别器的目标：有监督合成图片->源数据域
                              torch.FloatTensor(D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 8 #损失函数加权
        loss_D_src.backward(retain_graph=False) #计算判别器参数的梯度,并累加到网络参数的.grad属性中
        sum_Dsrc_loss += loss_D_src.item() #(本次的epoch的)源数据域 判决器损失

        pred_target = pred_target.detach()
        D_out_tar = predict_Discriminator_model(pred_target) # 判别 无监督真实图片

        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda()) # 判别器的目标：无监督真实图片->目标数据域
        loss_D_tar = loss_D_tar / 8  # bias #损失函数加权
        loss_D_tar.backward(retain_graph=False) #计算判别器参数的梯度,并累加到网络参数的.grad属性中
        sum_Dtar_loss += loss_D_tar.item() #(本次的epoch的)目标数据域 判别器损失
        optimizer_l.step() # 根据梯度更新分割器的参数
        # if hasattr(Segment_model, 'learnable_scalar'):
        #     print('learnable_scalar3 ', Segment_model.learnable_scalar, Segment_model.learnable_scalar.grad)
        optimizer_D.step() # 根据梯度更新判别器的参数

        # lr_policy, lrD_policy,      学习率调整的策略    <engine.lr_policy.WarmUpPolyLR>
        lr = lr_policy.get_lr(current_idx)  # lr change #调整学习率
        optimizer_l.param_groups[0]['lr'] = lr #分割网络
        optimizer_l.param_groups[1]['lr'] = lr #BN
        # for i in range(2, len(optimizer_l.param_groups)):   没用
        #     optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D #判别器
        # for i in range(2, len(optimizer_D.param_groups)):  没用
        #     optimizer_D.param_groups[i]['lr'] = Lr_D

        sum_contrastloss += loss_contrast_sum #(本次的epoch的)对比损失
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss_seg=%.4f' % loss_seg.item() \
                    + ' loss_D_tar=%.4f' % loss_D_tar.item() \
                    + ' loss_D_src=%.4f' % loss_D_src.item() \
                    + ' loss_adv=%.4f' % loss_adv.item() \
                    + ' loss_ce=%.4f' % loss_ce \
                    + ' loss_dice=%.4f' % loss_dice.item() \
                    + ' loss_contrast=%.4f' % loss_contrast_sum
        # if idx%config.idxBatchPrint==0:
        pbar.set_description(print_str, refresh=False) # 输出本batch的各项损失

        sum_totalloss = sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
        # 这个单batch的总损失在计算后没有被使用


        end_time = time.time()

    train_loss_seg = sum_loss_seg / len(pbar)           #伪监督(CE+DICE)
    train_loss_Dtar = sum_Dtar_loss / len(pbar)         #判别器-合成目标数据
    train_loss_Dsrc = sum_Dsrc_loss / len(pbar)         #判别器-合成源数据
    train_loss_adv = sum_adv_loss / len(pbar)           #对抗
    train_loss_ce = sum_celoss / len(pbar)              #CE
    train_loss_dice = sum_diceloss / len(pbar)          #DICE
    train_loss_contrast = sum_contrastloss / len(pbar)  #对比
    train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
    return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion


# evaluate(epoch, model, dataloader_val,criterion,criterion_consist)
def evaluate(epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    '''
    epoch,                      已完成的批次数     0
    Segment_model,              分割模型
    predict_Discriminator_model,判别器模型
    val_target_loader,          验证集加载器
    criterion,                  评价标准           DiceLoss() type=<utils.loss_function.DiceLoss>
    '''
    if torch.cuda.device_count() > 1:
        Segment_model.module.eval()
        predict_Discriminator_model.module.eval()
    else:#将模型设置为评价模式
        Segment_model.eval()
        predict_Discriminator_model.eval()
    with torch.no_grad(): #不进行梯度计算
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
            val_imgs = minibatch['img']     #图片数据
            val_gts = minibatch['anno_mask']#手工标签
            val_imgs = val_imgs.cuda(non_blocking=True)
            val_gts = val_gts.cuda(non_blocking=True)
            # NCHW
            val_pred_sup_l, sample_set_unsup, _ = Segment_model(val_imgs, mask=None, trained=False, fake=False)
            '''
            fake的含义是区分 真实血管/合成血管
                fake=T/F -> masks真标签/预测标签
                因为fake只影响对比学习，所以只在训练时有用、在评估时没用。
            '''

            max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0)
            val_max_l = max_l.float() # 1.<-1; 0.<-0;
            val_loss_sup = criterion(val_pred_sup_l, val_gts) #监督损失

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


class Predictor():
    def __init__(
        self,
        Segment_model,
        dataloader_val,
        dataloader_unsupervised
    ):
        self.Segment_model=Segment_model
        self.dataloader_val=dataloader_val
        self.dataloader_unsup=dataloader_unsupervised
        if True:  # 加载保存的状态字典
            self.loadParm()

    def loadParm(self):
        checkpoint_path = 'logs/best_Segment.pt'  # os.path.join(cls.logpath, 'best_Segment.pt')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
        self.Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    def is_pattern_connected(self, mask_tensor):
        # 将PyTorch张量转换为NumPy数组，保持单通道维度
        mask_array = mask_tensor.numpy().astype(np.uint8)

        # 创建一个空列表来存储处理后的MASK，保持与输入相同的shape
        processed_masks = []

        # 遍历每张MASK图片（保持单通道维度）
        for mask in mask_array:
            # 挤压掉单通道维度以进行连通性检测，但之后要恢复
            mask_squeeze = mask.squeeze()

            # 进行连通性检测，返回标记数组和连通区域的数量
            labeled_array, num_features = measure.label(mask_squeeze, connectivity=1, return_num=True)

            # 创建一个字典来存储每个标签的像素数
            region_sizes = {}
            for region in range(1, num_features + 1):
                # 计算每个连通区域的像素数
                region_size = np.sum(labeled_array == region)
                region_sizes[region] = region_size

            # 找到像素数最多的连通区域及其标签
            max_region = max(region_sizes, key=region_sizes.get)

            # 创建一个新的MASK，只保留像素数最多的连通区域，并恢复单通道维度
            processed_mask = np.zeros_like(mask)
            processed_mask[0, labeled_array == max_region] = 1

            # 将处理后的MASK添加到列表中
            processed_masks.append(processed_mask)

        # 将处理后的MASK列表转换回PyTorch张量
        processed_masks_tensor = torch.tensor(processed_masks, dtype=torch.float32)

        # 检查shape是否保持不变
        assert processed_masks_tensor.shape == mask_tensor.shape, "Processed masks tensor shape does not match original."
        return processed_masks_tensor

    def inference(self, loader , path ) :
        if torch.cuda.device_count() > 1:
            self.Segment_model.module.eval()
        else:  # 将模型设置为评价模式
            self.Segment_model.eval()
        with torch.no_grad():  # 不进行梯度计算
            os.makedirs(path, exist_ok=True)
            for val_idx, minibatch in enumerate(loader):
                val_imgs = minibatch['img']  # 图片的梯度数据
                val_img_name = minibatch['img_name']  # 图片名称
                val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
                val_pred_sup_l, sample_set_unsup, _ = self.Segment_model(val_imgs, mask=None, trained=False, fake=False)
                if True:
                    val_pred_sup_l = torch.where(val_pred_sup_l > 0.5, torch.ones_like(val_pred_sup_l),
                         torch.zeros_like(val_pred_sup_l))
                else:
                    val_pred_sup_l = ConnectivityAnalyzer(val_pred_sup_l).mainObj# val_pred_sup_l=self.is_pattern_connected(val_pred_sup_l.cpu())

                val_pred_sup_l = val_pred_sup_l * 255

                # 将tensor转换为numpy数组，并调整形状以匹配PIL的输入要求（N, H, W）
                images_np = val_pred_sup_l.to('cpu').numpy().squeeze(axis=1).astype(np.uint8)
                # 保存每张图片到本地文件
                for i, image in enumerate(images_np):
                    # 使用PIL创建图像对象，并保存为灰度图
                    img_pil = Image.fromarray(image, mode='L')  # 'L'模式表示灰度图
                    # img_pil.save("logs/"+val_img_name[i])  # 保存图片，文件名可以根据需要调整
                    # img_pil.save(os.path.join('logs', config.logname + ".log", "inference", val_img_name[i]))
                    img_pil.save(os.path.join(path, val_img_name[i]))
    def lastInference(self) :
        path = os.path.join('logs', config.logname + ".log", "inference")
        os.makedirs(path, exist_ok=True)
        self.inference(self.dataloader_val, path)
    def nextInference(self) :
        path = os.path.join('logs', config.logname + ".log", "unsup_temp")
        os.makedirs(path, exist_ok=True)
        self.inference(self.dataloader_unsup, path)


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

    dataloader_supervised,_ = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT # benchmark的本译是基准
                                                       config.batch_size, # 4
                                                       config.nworker,    # 8
                                                       'train',
                                                       config.img_mode,   # crop
                                                       config.img_size,   # 256
                                                       'supervised')
    # 有监督使用的数据是什么？gray中为血管的灰度图、gt中为标签。
    dataloader_unsupervised,dataset_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT
                                                         config.batch_size, # 4
                                                         config.nworker,    # 8
                                                         'train',
                                                         config.img_mode,   # crop
                                                         config.img_size,   # 256
                                                         'unsupervised')
    # 无监督使用的应该是img文件夹中的原始图片。

    dataloader_val,_ = CSDataset.build_dataloader(config.benchmark,       # XCAD_LIOT
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
    BatchNorm2d = nn.BatchNorm2d # <BatchNorm2d> #BN会重置均值和方差，新的均值和新的方差都是可学习的参数
    Segment_model = Single_contrast_UNet(4, config.num_classes) # config.num_classes=1
    # 我猜BN不放在Segment_model中的原因是：训练和评估这两种模式在使用的时候会有差异

    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_,
                # nn.init.kaiming_normal_: <function kaiming_normal_>
                BatchNorm2d,        # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
                config.bn_eps,      # config.bn_eps: 1e-05
                config.bn_momentum, # config.bn_momentum: 0.1
                mode='fan_in', nonlinearity='relu')
    # define the learning rate
    base_lr = config.lr      # 0.04 # 学习率
    base_lr_D = config.lr_D  # 0.04 # dropout?

    predictor = Predictor(Segment_model, dataloader_val, dataloader_unsupervised)

    params_list_l = []
    params_list_l = group_weight(
        params_list_l, #一个list对象 #用于存储tensor对象
        Segment_model.backbone, # 分割网络的主干
        BatchNorm2d,    # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
        base_lr)        # base_lr: 0.01
    if hasattr(Segment_model, 'learnable_scalar'): #用于优化对比学习的一个边缘间隔margin参数
        params_list_l.append(dict(params=Segment_model.learnable_scalar, lr=base_lr))
    # optimizer for segmentation_L   # 分割优化器_L
    print("config.weight_decay", config.weight_decay)
    optimizer_l = torch.optim.SGD(params_list_l,#分割网络中的全部参数
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),#判别器中的全部参数
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
    # inference(Segment_model, dataloader_val)
    for epoch in range(config.state_epoch, config.nepochs): #从state_epoch到nepochs-1 # 按照预先设定的回合数量执行，不会提前中止
        # train_loss_sup, train_loss_consis, train_total_loss
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion \
            = train(#训练
                epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
                optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
                average_negregion, dataset_unsupervised.isFirstEpoch)
        print("train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},train_loss_contrast:{}".format(
                train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss,train_loss_contrast))
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
        write_csv(val_score_path, data_row_f1score) # 保存在验证集下的实验结果
        if val_mean_f1 > best_val_f1: # F1分数是精确率和召回率的调和平均数
            best_val_f1 = val_mean_f1
            Logger.save_model_f1_S(Segment_model, epoch, val_mean_f1, optimizer_l) #保存到best_Segment.pt中
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_f1, optimizer_D) #保存到best_Dis.pt中

        # if val_mean_AUC > best_val_AUC:
        #     best_val_AUC = val_mean_AUC
        #     Logger.save_model_f1_S(Segment_model, epoch, val_mean_AUC, optimizer_l)
        #     Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_AUC, optimizer_D)
        if config.pseudo_label:
            predictor.nextInference()
            dataset_unsupervised.isFirstEpoch=False #已经保存了伪标签数据

    predictor.lastInference()# inference(Segment_model, dataloader_val)

if __name__ == '__main__':
    main()

'''
    2. 训练脚本
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS
    python train_DA_contrast_liot_finalversion.py 
    #(CUDA_VISIBLE_DEVICES=0 python train_DA_contrast_liot_DRIVE_finalversion.py for DRIVE)
'''
