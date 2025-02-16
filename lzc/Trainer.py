import sys
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

from utils.loss_function import ContrastRegionloss_quaryrepeatNCE

from lzc.BCELoss_lzc import BCELoss_lzc
from lzc.ConnectivityAnalyzer import ConnectivityAnalyzer
from lzc.csv_lib import create_csv,write_csv,getPath_csv


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

class Trainer():
    def __init__(
        self,Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
          optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
          average_negregion
    ):
        '''
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

        self.Segment_model = Segment_model
        self.predict_Discriminator_model = predict_Discriminator_model
        self.dataloader_supervised = dataloader_supervised
        self.dataloader_unsupervised = dataloader_unsupervised
        self.optimizer_l = optimizer_l
        self.optimizer_D = optimizer_D
        self.lr_policy = lr_policy
        self.lrD_policy = lrD_policy
        self.criterion = criterion
        self.total_iteration = total_iteration
        self.average_posregion = average_posregion
        self.average_negregion = average_negregion

        self.bce_loss = nn.BCELoss()
        self.source_label = 0  # 源数据域:合成血管 #用于对抗学习，源数据域的标签
        self.target_label = 1  # 目标数据域:真实血管 #对抗学习的目的是让神经网络能够屏蔽数据域的差异，让两个数据域都能输出正确的结果
        self.criterion_contrast = ContrastRegionloss_quaryrepeatNCE()

        self.csv_loss_path = getPath_csv() + '/' + 'train_loss.csv'
        self.lossNameList = [
            "loss_D_tar",
            "loss_D_src",
            # "loss_adv",
            "loss_ce",
            "loss_dice",

            # 【1.合成监督、2.对抗、3.对比、4.伪监督、5.连通损失】
            "loss_seg_w",
            "loss_adv_w",
            "loss_contrast_w",
            "loss_pseudo_w",
            "loss_conn_w",

            "loss_adv",
            "loss_contrast",
            "loss_pseudo",
            "loss_conn"]
        csv_head = ["epoch"]
        for i in self.lossNameList:
            csv_head.append(i)
        create_csv(self.csv_loss_path, csv_head)

    def train(self,epoch,isFirstEpoch):
        Segment_model = self.Segment_model
        predict_Discriminator_model = self.predict_Discriminator_model
        dataloader_supervised = self.dataloader_supervised
        dataloader_unsupervised = self.dataloader_unsupervised
        optimizer_l = self.optimizer_l
        optimizer_D = self.optimizer_D
        total_iteration = self.total_iteration
        average_posregion = self.average_posregion
        average_negregion = self.average_negregion

        if torch.cuda.device_count() > 1:  # 如果有多个CUDA
            Segment_model.module.train()
            predict_Discriminator_model.module.train()
            '''
                .module: 这个属性通常在模型被封装或复制时出现。
                    并行化处理后，原始模型会被封装在一个新的对象中，而这个新对象会有一个.module属性指向原始的模型。
                .train(): 将模型设置为训练模式。
            '''
        else:  # 将模型设置为训练模式
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
        dataloader = iter(dataloader_supervised)  # 有监督的数据加载器
        unsupervised_dataloader = iter(dataloader_unsupervised)  # 无监督的数据加载器

        sum_loss_seg = 0
        sum_adv_loss = 0
        sum_Dsrc_loss = 0
        sum_Dtar_loss = 0
        sum_totalloss = 0
        sum_contrastloss = 0
        sum_celoss = 0
        sum_diceloss = 0
        # sum_pseudo = 0
        # sum_conn =0
        total_loss={}
        for i in self.lossNameList:
            total_loss[i]=0

        print('begin train')
        ''' supervised part '''
        for idx in pbar:
            current_idx = epoch * config.niters_per_epoch + idx
            damping = (1 - current_idx / total_iteration)  # 剩余工作量(damping本意指阻尼) #1->0
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
                minibatch = next(dataloader)  # 获取一个batch的有监督数据
            except StopIteration:  # 如果加载完一个epoch，就重新初始化加载器
                dataloader = iter(dataloader_supervised)
                minibatch = next(dataloader)

            try:  # 获取一个batch的无监督数据
                unsup_minibatch = next(unsupervised_dataloader)
                # 这段代码尝试从unsupervised_dataloader迭代器中获取下一个数据批次。
            except StopIteration:  # 如果迭代器已经耗尽（引发了StopIteration异常）
                unsupervised_dataloader = iter(dataloader_unsupervised)  # 则重新初始化迭代器
                unsup_minibatch = next(unsupervised_dataloader)  # 再次尝试获取下一个数据批次

            loss = self.__forward(minibatch,unsup_minibatch,damping,isFirstEpoch)
            self.__backward( loss['loss_adv'] , loss['loss_D_src'] + loss['loss_D_tar'] )
            lr = self.__step(current_idx)

            for i in total_loss:
                total_loss[i] += loss[i]

            # 【1.合成监督、2.对抗、3.对比、4.伪监督、5.连通损失】
            # 1.合成监督
            sum_loss_seg += loss['loss_seg_w'].item()  # (本次的epoch的)合成监督损失
            sum_celoss += loss['loss_ce']  # (本次的epoch的)CE损失
            sum_diceloss += loss['loss_dice'].item()  # (本次的epoch的)DICE损失
            # 2.对抗
            sum_adv_loss += loss['loss_adv_w'].item()  # (本次的epoch的)对抗损失
            # 3.对比
            sum_contrastloss += loss['loss_contrast_w'].item()  # (本次的epoch的)加权后的对比损失
            # # 4.伪监督pseudo
            # sum_pseudo += loss['loss_contrast_w'].item()
            # # 5.连通conn
            # sum_conn += loss['loss_contrast_w'].item()
            
            sum_Dsrc_loss += loss['loss_D_src'].item()  # (本次的epoch的)源数据域 判决器损失
            sum_Dtar_loss += loss['loss_D_tar'].item()  # (本次的epoch的)目标数据域 判别器损失
            sum_contrastloss += loss['loss_contrast_w']  # (本次的epoch的)对比损失
            sum_totalloss += loss['loss_adv'].item() #sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
            # 这个单batch的总损失在计算后没有被使用

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_seg=%.4f' % loss['loss_seg_w'].item() \
                        + ' loss_D_tar=%.4f' % loss['loss_D_tar'].item() \
                        + ' loss_D_src=%.4f' % loss['loss_D_src'].item() \
                        + ' loss_adv=%.4f' % loss['loss_adv'].item() \
                        + ' loss_ce=%.4f' % loss['loss_ce'] \
                        + ' loss_dice=%.4f' % loss['loss_dice'].item() \
                        + ' loss_contrast=%.4f' % loss['loss_contrast_w']
            # if idx%config.idxBatchPrint==0:
            pbar.set_description(print_str, refresh=False)  # 输出本batch的各项损失

        lossSave=[epoch]
        # for i in total_loss:
        for i in self.lossNameList:
            lossSave.append( total_loss[i] / len(pbar) )
        write_csv(self.csv_loss_path, lossSave)
        

        train_loss_seg = sum_loss_seg / len(pbar)  # 伪监督(CE+DICE)
        train_loss_Dtar = sum_Dtar_loss / len(pbar)  # 判别器-合成目标数据
        train_loss_Dsrc = sum_Dsrc_loss / len(pbar)  # 判别器-合成源数据
        train_loss_adv = sum_adv_loss / len(pbar)  # 对抗
        train_loss_ce = sum_celoss / len(pbar)  # CE
        train_loss_dice = sum_diceloss / len(pbar)  # DICE
        train_loss_contrast = sum_contrastloss / len(pbar)  # 对比
        train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
        return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion

    def __forward(self,minibatch,unsup_minibatch,damping,isFirstEpoch):
        Segment_model = self.Segment_model
        predict_Discriminator_model = self.predict_Discriminator_model
        criterion = self.criterion

        bce_loss = self.bce_loss
        source_label = self.source_label
        target_label = self.target_label
        criterion_contrast = self.criterion_contrast

        imgs = minibatch['img']  # imgs: [4, 4, 256, 256] #每个batch有4张图片
        gts = minibatch['anno_mask']  # gts:  [4, 1, 256, 256]
        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)

        with torch.no_grad():  # 禁用梯度计算
            weight_mask = gts.clone().detach()
            weight_mask[weight_mask == 0] = 0.1  # 值为0的元素设为0.1
            weight_mask[weight_mask == 1] = 1  # 值为1的元素保持不变
            # 血管区域太小，背景区域太大。因此分别给这两个区域使用了不同的权重。
            if True:  # if config.ASL:
                criterion_bce = BCELoss_lzc(
                    weight=weight_mask,
                    gamma_pos=config.gamma_pos,
                    gamma_neg=config.gamma_neg)
            else:
                criterion_bce = nn.BCELoss(weight=weight_mask)

        unsup_imgs = unsup_minibatch['img']
        # unsup_minibatch:{
        #   'img_name': ['**.png', ...],
        #   'img':tensor
        # }
        unsup_imgs = unsup_imgs.cuda(non_blocking=True)  # unsup_imgs:[4, 4, 256, 256]

        # Start train fake vessel 开始训练合成血管
        pred_sup_l, sample_set_sup, flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True,fake=False)
        # mask影响正负样本的获取

        # 1.(当前batch的)合成监督损失
        # pred_sup_l： 类0-1标签的预测结果
        # sample_sets：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        loss_ce = criterion_bce(pred_sup_l, gts)  # 根据预测结果和标签计算CE损失 # retinal是5 XCAD是0.1 crack是5 # For retinal :5 For XCAD:0.1 5 for crack
        loss_dice = criterion(pred_sup_l, gts)  # 根据预测结果和标签计算Dice损失

        # 2.对抗损失
        D_seg_target_out = predict_Discriminator_model(pred_target)  # 计算对抗损失 #判别是否为 直线合成血管 or 曲线标注血管
        # pred_target      shape=[4, 1, 256, 256]
        # D_seg_target_out shape=[4, 1, 8, 8]
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out),  # 无监督曲线标注血管的预测结果
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())

        # 3.对比学习损失
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
                loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature,
                                                   Segment_model.learnable_scalar)
            else:
                loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
            '''
                quary_feature: 合成图像 全部易正样本的特征 [<=2000, 64]=[2000, 64]       ->len=579 (这里的正样本成对使用)
                pos_feature:   真实图像 全部易正样本的特征 [<=2000, 64]=[579,  64]       ->len=579 
                neg_feature:   合成和真实的图像 部分易负样本的特征 [<=500, 64]=[462, 64]   ->len=462
            '''
        else: loss_contrast = 0

        # 4.伪监督损失
        if config.pseudo_label and isFirstEpoch == False:
            gts_pseudo = unsup_minibatch['anno_mask']  # 原始数据
            gts_pseudo = gts_pseudo.cuda(non_blocking=True)

            weight_mask2 = gts_pseudo.clone().detach()
            weight_mask2[weight_mask2 == 0] = 0.1  # 值为0的元素设为0.1
            weight_mask2[weight_mask2 == 1] = 1  # 值为1的元素保持不变

            with torch.no_grad():  # 禁用梯度计算
                criterion_bce2 = BCELoss_lzc(
                    weight=weight_mask2,
                    gamma_pos=config.gamma_pos,
                    gamma_neg=config.gamma_neg)
            loss_pseudo = criterion_bce2(pred_target, gts_pseudo)
        else: loss_pseudo = 0

        # 5.连通性损失
        if config.connectivityLoss:  # 使用连通损失
            loss_conn1 = ConnectivityAnalyzer(pred_sup_l).connectivityLoss(config.connectivityLossType)  # 合成监督
            loss_conn2 = ConnectivityAnalyzer(pred_target).connectivityLoss(config.connectivityLossType)  # 无/伪监督
            loss_conn = loss_conn1 + loss_conn2
            # if config.pseudo_label and isFirstEpoch == False:
            #     loss_conn3 = ConnectivityAnalyzer(pred_target).connectivityLoss(config.connectivityLossType)  # 伪监督
            #     loss_conn = loss_conn + loss_conn3
        else: loss_conn=0

        # 【1.合成监督、2.对抗、3.对比、4.伪监督、5.连通损失】
        def useW_seg(l_d,l_c,c):
            l = l_d * c["weight_dice"] + l_c * c["weight_ce"]
            if c["damping"]=="increase":
                l=l*(1-damping)
            elif c["damping"]=="reduce":
                l=l*damping
            elif not c["damping"]=="constant":
                print("The damping parameter in the configuration file is invalid! (配置文件中的damping参数不合法!)")
                exit(0)
            return l
        def useW(l,c):
            if c["damping"]=="increase":
                l=l*(1-damping)
            elif c["damping"]=="reduce":
                l=l*damping # damping:剩余工作量(阻尼) #1->0
            elif not c["damping"]=="constant":
                print("The damping parameter in the configuration file is invalid! (配置文件中的damping参数不合法!)")
                exit(0)
            return l*c["weight"]
        
            
        loss_seg_w = useW_seg(loss_dice,loss_ce,config.seg) # damping:剩余工作量(阻尼) #1->0
        # loss_seg_w = loss_dice + loss_ce * 0.1
        loss_adv_w = useW(loss_adv_target, config.adv) 
        # loss_adv_w = loss_adv_target * damping * 0.25 
        # damping的取值范围是: 1到0
        loss_contrast_w = loss_contrast * 0.04  # (当前batch的)加权后的对比损失 # weight_contrast = 0.04  # 对比损失的权重
        loss_pseudo_w = loss_pseudo * ( 1 - damping ) * 0.01
        loss_conn_w = loss_conn * 0.1

        loss_adv = loss_seg_w + loss_adv_w + loss_contrast_w + loss_pseudo_w + loss_conn_w

        ############################################################################################################
        # 【1.合成监督】
        pred_sup_l = pred_sup_l.detach() # 这样可以确保接下来不优化分割器
        # pred_sup_l： 类0-1标签的预测结果 shape=[4, 1, 256, 256]
        # “分离”出一个副本，这个副本在自动微分过程中不会被考虑用于梯度计算。
        D_out_src = predict_Discriminator_model(pred_sup_l)  # D_out_src.shape=[4, 1, 8, 8]

        loss_D_src = bce_loss(F.sigmoid(D_out_src),  # 判别器的目标：有监督合成图片->源数据域
                              torch.FloatTensor(D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 8  # 损失函数加权
        # loss_D_src.backward(retain_graph=False)  # 计算判别器参数的梯度,并累加到网络参数的.grad属性中

        # 【2.无/伪监督】
        pred_target = pred_target.detach()
        D_out_tar = predict_Discriminator_model(pred_target)  # 判别 无监督真实图片

        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda())  # 判别器的目标：无监督真实图片->目标数据域
        loss_D_tar = loss_D_tar / 8  # bias #损失函数加权

        return {
            "loss_D_tar":loss_D_tar,
            "loss_D_src":loss_D_src,
            # "loss_adv":loss_adv,
            "loss_ce":loss_ce,
            "loss_dice":loss_dice,

            # 【1.合成监督、2.对抗、3.对比、4.伪监督、5.连通损失】
            "loss_seg_w":loss_seg_w,
            "loss_adv_w":loss_adv_w,
            "loss_contrast_w":loss_contrast_w,
            "loss_pseudo_w":loss_pseudo_w,
            "loss_conn_w":loss_conn_w,

            "loss_adv":loss_adv,
            "loss_contrast":loss_contrast,
            "loss_pseudo":loss_pseudo,
            "loss_conn":loss_conn
        }

    def __backward(self,loss_adv,loss_D):
        predict_Discriminator_model = self.predict_Discriminator_model

        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False  # 首先优化分割器、不优化判别器
        loss_adv.backward(retain_graph=False)  # 计算分割网络参数的梯度,并累加到网络参数的.grad属性中

        for param in predict_Discriminator_model.parameters():  # 开启判别器的优化
            param.requires_grad = True # 将判别器中的参数设置为需要计算梯度
        loss_D.backward(retain_graph=False)  # 计算判别器参数的梯度,并累加到网络参数的.grad属性中

    def __step(self,current_idx):
        optimizer_l = self.optimizer_l
        optimizer_D = self.optimizer_D
        lr_policy = self.lr_policy
        lrD_policy = self.lrD_policy

        optimizer_l.step()  # 根据梯度更新分割器的参数
        # if hasattr(Segment_model, 'learnable_scalar'):
        #     print('learnable_scalar3 ', Segment_model.learnable_scalar, Segment_model.learnable_scalar.grad)
        optimizer_D.step()  # 根据梯度更新判别器的参数

        # lr_policy, lrD_policy,      学习率调整的策略    <engine.lr_policy.WarmUpPolyLR>
        lr = lr_policy.get_lr(current_idx)  # lr change #调整学习率
        optimizer_l.param_groups[0]['lr'] = lr  # 分割网络
        optimizer_l.param_groups[1]['lr'] = lr  # BN
        # for i in range(2, len(optimizer_l.param_groups)):   没用
        #     optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D  # 判别器
        # for i in range(2, len(optimizer_D.param_groups)):  没用
        #     optimizer_D.param_groups[i]['lr'] = Lr_D
        return lr

