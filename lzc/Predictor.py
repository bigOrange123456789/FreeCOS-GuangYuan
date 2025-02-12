import os

import torch

from config import config

import numpy as np
from PIL import Image

from lzc.ConnectivityAnalyzer import ConnectivityAnalyzer

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
        if False:  # 加载保存的状态字典
            self.loadParm()

    def loadParm(self):
        checkpoint_path = 'logs/best_Segment.pt'  # os.path.join(cls.logpath, 'best_Segment.pt')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
        self.Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    def __inference(self, loader , path ) :
        os.makedirs(path, exist_ok=True)
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
                if config.onlyMainObj: #True:
                    val_pred_sup_l = ConnectivityAnalyzer(val_pred_sup_l).mainObj
                else:
                    val_pred_sup_l = torch.where(val_pred_sup_l > 0.5, torch.ones_like(val_pred_sup_l),
                         torch.zeros_like(val_pred_sup_l))

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
        self.__inference(self.dataloader_val, path)
    def nextInference(self) :
        path = os.path.join('logs', config.logname + ".log", "unsup_temp")
        self.__inference(self.dataloader_unsup, path)

    def showInput(self):
        path = os.path.join('logs', config.logname + ".log", "liot")
        os.makedirs(path, exist_ok=True)
        loader=self.dataloader_unsup
        with torch.no_grad():  # 不进行梯度计算
            for val_idx, minibatch in enumerate(loader):
                val_img_name = minibatch['img_name']  # 图片名称
                val_imgs = minibatch['img_copy']  # 图片的梯度数据
                val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
                # print("val_imgs",val_imgs.shape)
                # print("val_imgs[0, 0, :, :]",val_imgs[0, 0, :, :])

                val_imgs_t = minibatch['img_test']*255  # 图片的梯度数据
                # val_imgs_t = minibatch['img_copy']  # 图片的梯度数据
                # val_imgs_t =val_imgs_t*255

                # gray=minibatch['img']
                # print("gray",gray.shape,type(gray))

                # val_imgs = val_imgs * 255

                # 将tensor转换为numpy数组，并调整形状以匹配PIL的输入要求（N, H, W）
                # images_np = val_imgs[:,0,:,:].to('cpu').numpy()#.squeeze(axis=1).astype(np.uint8)
                images_np = val_imgs[:, 0, :, :].to('cpu').numpy().astype(np.uint8)
                images_np_t = val_imgs_t[:, 0, :, :].to('cpu').numpy().astype(np.uint8)
                # print("images_np_t:",images_np_t)
                # print("images_np_t[0,:,:]",images_np_t[0,:,:])
                # 保存每张图片到本地文件
                for i, image in enumerate(images_np):
                    # 使用PIL创建图像对象，并保存为灰度图
                    img_pil = Image.fromarray(image, mode='L')  # 'L'模式表示灰度图
                    img_pil.save(os.path.join(path, "input."+val_img_name[i]))

                    img_pil = Image.fromarray(images_np_t[i,:,:], mode='L')  # 'L'模式表示灰度图
                    img_pil.save(os.path.join(path, "test." + val_img_name[i]))


