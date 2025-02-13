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


from lzc.BCELoss_lzc import BCELoss_lzc
from lzc.ConnectivityAnalyzer import ConnectivityAnalyzer

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

from lzc.Predictor import Predictor
from lzc.Trainer import Trainer

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
    # Segment_model = Single_contrast_UNet(4, config.num_classes) # config.num_classes=1
    if config.inputType=="Origin":
        n_channels = 1
    elif config.inputType=="LIOT":
        n_channels = 4
    elif config.inputType == "NewLIOT":
        n_channels = 4
    elif config.inputType == "NewLIOT2":
        n_channels = 4
    else:
        print("配置文件中的inputType参数不合法！")
        exit(0)
    Segment_model = Single_contrast_UNet(n_channels, config.num_classes)
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
    # predictor.showInput()#测试代码

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
    trainer = Trainer(Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
                optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
                average_negregion)
    # inference(Segment_model, dataloader_val)
    for epoch in range(config.state_epoch, config.nepochs): # 从state_epoch到nepochs-1 # 按照预先设定的回合数量执行，不会提前中止
        # train_loss_sup, train_loss_consis, train_total_loss
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion \
            =trainer.train(epoch, dataset_unsupervised.isFirstEpoch)
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
