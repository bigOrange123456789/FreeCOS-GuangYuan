import torch
import numpy as np
from skimage import measure

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
