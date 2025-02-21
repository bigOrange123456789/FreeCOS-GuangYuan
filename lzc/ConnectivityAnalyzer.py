import torch
import numpy as np
from skimage import measure

class ConnectivityAnalyzer:
    def __init__(
            self,
            mask_tensor,
    ):
        self.mask_tensor=mask_tensor
        self.allObj=torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
                                  torch.zeros_like(mask_tensor))
        self.mainObj=self.getMainObj(self.allObj)
    def connectivityLoss(self,type):
        if type=='ratio':
            return self.__connectivityLossRatio()
        elif type=='entropy':
            return self.__connectivityLossEntropy()
        else:
            print('配置文件中的connectivityLossType参数不合法！')
            exit(0)
    def __connectivityLossEntropy(self):
        return self.entropyList.mean()
    def __connectivityLossRatio(self):#旧版连通性损失函数的计算方法
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
        mask_tensor=mask_tensor.cpu() # 我猜.转换到CPU当中之后就不会计算梯度了
        # mask_tensor = torch.where(mask_tensor > 0.5, torch.ones_like(mask_tensor),
        #                              torch.zeros_like(mask_tensor))

        # 将PyTorch张量转换为NumPy数组，保持单通道维度
        mask_array = mask_tensor.numpy().astype(np.uint8)

        # 创建一个空列表来存储处理后的MASK，保持与输入相同的shape
        processed_masks = []

        entropyList=[]
        # 遍历每张MASK图片（保持单通道维度）
        for mask in mask_array:
            # 挤压掉单通道维度以进行连通性检测，但之后要恢复
            mask_squeeze = mask.squeeze()
            if mask_squeeze.sum()==0:#这个对象为空
                processed_masks.append(mask)
                continue

            # 进行连通性检测，返回标记数组和连通区域的数量
            labeled_array, num_features = measure.label(mask_squeeze, connectivity=1, return_num=True)
            entropy=self.__computeEntropy( labeled_array)
            entropyList.append(entropy)
            '''
    这行代码调用了measure.label函数，对输入的mask_squeeze数组进行处理。
    mask_squeeze：这是一个二维或三维的数组（通常是二值的，即只包含0和1），表示某种形式的图像或空间数据。
    connectivity=1：意味着只有直接相邻的像素（即上下左右）被视为连通的。这个参数影响如何定义连通组件。
    return_num=True：这个参数指示函数除了返回标记数组外，还要返回检测到的连通组件（或特征）的数量。
    标记数组中的每个连通组件都会被赋予一个唯一的标签（从1开始）。
            '''

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

        self.entropyList=torch.tensor(entropyList).to(device)

        return processed_masks_tensor.to(device)
    def __computeEntropy(self, labeled_array):
        # 使用 measure.regionprops 函数计算每个连通组件的属性
        props = measure.regionprops(labeled_array)

        # 创建一个字典来存储每个连通组件的标签和对应的像素数量
        component_sizes = {prop.label: prop.area for prop in props}

        # 如果需要，也可以将组件大小和标签以列表形式返回
        # labels = list(component_sizes.keys())
        sizes = list(component_sizes.values())


        probs=np.array(sizes)/np.sum(sizes)

        # 为了避免对0取对数（因为0的对数是没有定义的），我们将非常小的概率值替换为一个非常小的正数（例如1e-10）
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1.0 - epsilon)

        # 归一化概率分布（虽然使用Dirichlet分布生成的概率分布已经归一化，但此步骤确保万无一失）
        probs /= np.sum(probs)

        # 计算信息熵
        entropy = -np.sum(probs * np.log2(probs))
        return entropy


