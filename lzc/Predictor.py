import os

import torch

from config import config

import numpy as np
from PIL import Image
from skimage import measure

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
        self.inference(self.dataloader_val, path)
    def nextInference(self) :
        path = os.path.join('logs', config.logname + ".log", "unsup_temp")
        self.inference(self.dataloader_unsup, path)

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


