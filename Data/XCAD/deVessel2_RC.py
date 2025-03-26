
import numpy as np
import PIL.Image as Image
import os.path
import cv2

class DeVessel():
    def save(self,img,path):
        img_FDA_Image=Image.fromarray((img).astype('uint8')).convert('L')
        img_FDA_Image.save(path, format='PNG')

    def load(self,file_path):
        background_img = Image.open( file_path ).convert('L') #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img)
        return np.asarray(background_array, np.float32) 
    
    def __init__(self , config):
        # self.path = path
        self.config = config
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]

    
    def process(self, inPath, outPath):
        self.path = inPath
        img=self.load(self.path)
        shape = img.shape
        self.width=shape[0]
        self.height=shape[1]
        # print("shape",shape)
        # shape (512, 512)
        # shape (605, 700)
        # exit(0)
        img_deVessel3=self.deVessel3(img)
        self.save(img_deVessel3,outPath)

    def deVessel(self,img):
        img = np.expand_dims(img, axis=2) # (512, 512) -> (512, 512, 1)
        img = img.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        # 傅里叶变换 # get fft of both source and target
        fft_trg_np = np.fft.fft2( img, axes=(-2, -1) )

        def low_freq_mutate_np2(amp_src): #L=0~255
            # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
            a_src = np.fft.fftshift( amp_src, axes=(-2, -1) ) # np.zeros_like(amp_trg)
            # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
            _, h, w = a_src.shape # h=512 w=512       
            for i in range(h):
                    for j in range(w):
                        k = ( (i-h/2)**2 + (j-w/2)**2 )**0.5
                        if k>10 and k<90 and not (k>50 and k<55): #(10,100) 
                            a_src[0,i,j]=0

            a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
            return a_src
    
        # extract amplitude and phase of both ffts
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate_np2( amp_trg )
        pha_src_ = low_freq_mutate_np2( pha_trg )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部
        print("src_in_trg",src_in_trg.shape,src_in_trg)

        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)

        return img_FDA 
    
    def deVessel2(self,img,noise):
        img = np.expand_dims(img, axis=2) # (512, 512) -> (512, 512, 1)
        noise = np.expand_dims(noise, axis=2) # (512, 512) -> (512, 512, 1)
        img = img.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
        noise = noise.transpose((2, 0, 1))

        # 傅里叶变换 # get fft of both source and target
        fft_trg_np1 = np.fft.fft2( img, axes=(-2, -1) )
        fft_trg_np2 = np.fft.fft2( noise, axes=(-2, -1) )

        def low_freq_mutate_np2(amp_src1,amp_src2): #L=0~255
            # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
            a_src1 = np.fft.fftshift( amp_src1, axes=(-2, -1) ) # np.zeros_like(amp_trg)
            a_src2 = np.fft.fftshift( amp_src2, axes=(-2, -1) ) 
            # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
            _, h, w = a_src1.shape # h=512 w=512       
            for i in range(h):
                    for j in range(w):
                        k = ( (i-h/2)**2 + (j-w/2)**2 )**0.5
                        # if k>10 and k<90 and not (k>50 and k<55): #(10,100) 
                        if k>10 and k<100 : #(10,90) 
                            # a_src1[0,i,j]=0
                            a_src1[0,i,j]=a_src2[0,i,j]

            return np.fft.ifftshift( a_src1, axes=(-2, -1) )
    
        # extract amplitude and phase of both ffts
        amp_trg1, pha_trg1 = np.abs(fft_trg_np1), np.angle(fft_trg_np1)
        amp_trg2, pha_trg2 = np.abs(fft_trg_np2), np.angle(fft_trg_np2)

        # mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate_np2( amp_trg1 ,amp_trg2)
        pha_src_ = low_freq_mutate_np2( pha_trg1 ,pha_trg2)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部
        # print("src_in_trg",src_in_trg.shape,src_in_trg)

        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)

        return img_FDA
     
    def addNoise(self, img0):
        if self.gamma==0:
            return img0*0
        def gaussian_kernel(size, sigma=1.0):
            
            # 创建一个二维网格
            x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
            g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
            return g / g.sum()  # 归一化，使核的总和为1

        def apply_gaussian_blur(image, kernel):
            """
            对图像应用高斯模糊
            :param image: 输入图像（灰度图像）
            :param kernel: 高斯核
            :return: 模糊后的图像
            """
            # 获取图像和核的大小
            h, w = image.shape
            k_h, k_w = kernel.shape
            
            # 计算边界填充的宽度
            pad_h = k_h // 2
            pad_w = k_w // 2
            
            # 对图像进行边界填充
            padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
            
            # 创建输出图像
            blurred_image = np.zeros_like(image, dtype=np.float64)
            
            # 应用高斯核
            for i in range(h):
                for j in range(w):
                    # 提取当前像素周围的区域
                    region = padded_image[i:i+k_h, j:j+k_w]
                    # 应用高斯核并累加结果
                    blurred_image[i, j] = np.sum(region * kernel)
            
            return blurred_image

        # print(kernel)
        # print(np.sum(kernel))
        # exit(0)
        img0 = img0[0,:,:]
        # 2.高斯模糊
        # img_FDA_guassian = cv2.GaussianBlur(img0, (13, 13), 0) #：高斯核的大小必须是正奇数（如 3、5、7、9 等）。
        # img_FDA_guassian = cv2.GaussianBlur(img0, (13, 13), 0)
        if True:
            img1 = cv2.GaussianBlur(img0, (3, 3), 0)
        else:
            kernel=gaussian_kernel(5,1)
            img1 = apply_gaussian_blur(img0, kernel)
        # 3.添加点噪声
        # noise_map = np.random.uniform(-5, 5, img_FDA_guassian.shape)
        param=0#0.1
        noise_map = np.random.normal(0, np.pi*param, img0.shape)
        img2 = img1 + noise_map
        result = np.clip(img2, -np.pi, np.pi)
        arr_expanded = np.expand_dims(result, axis=0)
        return self.gamma*arr_expanded
    
    def deVessel3(self,img):
        # testFlag=True
        ##########################    1.计算noise   ##########################
        img = np.expand_dims(img, axis=2) # (512, 512) -> (512, 512, 1)
        img = img.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
            
        fft_trg_np = np.fft.fft2( img, axes=(-2, -1) )# 傅里叶变换

        # extract amplitude and phase of both ffts
        amp, pha = np.abs(fft_trg_np), np.angle(fft_trg_np)
        if False:
            print(type(pha))
            print(pha.shape)
            print("max:",np.max(pha))
            print("min:",np.min(pha))
            print(np.pi)
            # 计算梯度
            grad = np.gradient(pha[0,:,:]) # grad 是一个包含三个梯度数组的元组，分别对应于每个轴的梯度
            max_grad_row = np.max(np.abs(grad[0]))# grad[1] 是沿第二个轴（行轴，大小为 605）的梯度
            max_grad_col = np.max(np.abs(grad[1]))# grad[2] 是沿第三个轴（列轴，大小为 700）的梯度
            print(max_grad_row)
            print(max_grad_col)

        # mutated fft of source
        if False:
            fft_0 = amp * np.exp( 1j * np.zeros_like(pha) )
        elif False:
            # print(pha.shape,"shape2")
            # exit(0)
            pha_noise = self.addNoise(pha)
            fft_0 = amp * np.exp( 1j * pha_noise )
        else:
            if self.gamma==0:#XCAD
                fft_0 = amp #* np.exp( 1j * pha*0 )
            elif self.gamma==1:#STARE
                fft_0 = amp*0 #* np.exp( 1j * pha )
            else:
                print("gamma数值非法!")
                exit(0)

        # 逆傅里叶变换 # get the mutated image
        src_0 = np.fft.ifft2( fft_0, axes=(-2, -1) )
        src_0 = np.real(src_0) # 从复数数组中提取实部。
            
        noise = np.clip(src_0, 0, 255.)#限制像素的最小值为0、最大值为255
        if False:#testFlag:
            print(noise.shape,type(noise))
            # exit(0)
            self.save(noise[0,:,:],"./test/img/test_noise.jpg")
        ##########################    2.计算noise   ##########################
        # 傅里叶变换 # get fft of both source and target
        fft_trg_np2 = np.fft.fft2( noise, axes=(-2, -1) )

        def low_freq_mutate_np2(amp_src1,amp_src2): #L=0~255
            a_src1 = np.fft.fftshift( amp_src1, axes=(-2, -1) ) # np.zeros_like(amp_trg)
            a_src2 = np.fft.fftshift( amp_src2, axes=(-2, -1) ) 
            # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
            _, h, w = a_src1.shape # h=512 w=512       
            # print(256,self.width,self.height)
            for i in range(h):
                    for j in range(w):
                        x=(i-h/2)/self.width
                        y=(j-w/2)/self.height
                        k = ( x**2 + y**2 )**0.5
                        # if k>self.alpha and k<self.beta : 
                        if abs(x)>self.alpha or abs(y)>self.alpha : 
                            a_src1[0,i,j]=a_src2[0,i,j]#将血管区域用噪声填充
            return np.fft.ifftshift( a_src1, axes=(-2, -1) )
    
        # extract amplitude and phase of both ffts
        amp_trg2, pha_trg2 = np.abs(fft_trg_np2), np.angle(fft_trg_np2)

        # mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate_np2( amp ,amp_trg2)
        pha_src_ = low_freq_mutate_np2( pha ,pha_trg2)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部
        # print("src_in_trg",src_in_trg.shape,src_in_trg)

        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)

        return img_FDA 

    def dePhase(self,img):
        img = np.expand_dims(img, axis=2) # (512, 512) -> (512, 512, 1)
        img = img.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
        
        # 傅里叶变换 # get fft of both source and target
        fft_trg_np = np.fft.fft2( img, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp, pha = np.abs(fft_trg_np), np.angle(fft_trg_np)
        pha = np.zeros_like(pha)

        # mutated fft of source
        fft_src_ = amp * np.exp( 1j * pha )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。

        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
        return img_FDA


config_XCAD={
    "alpha":10/512, 
    "beta":100/512,
    "gamma":0,

    "root":"../../../DataSet-images/XCAD_FreeCOS/train/",
    "in":"bg_lzc",
    "out":"bg_RC"
}
config_30XCA={
    "alpha":10/512, 
    "beta":100/512,
    "gamma":0,

    "root":"../../../DataSet-images/30XCA/train/",
    "in":"img",
    "out":"bg"
}
config_XTARE={
    "alpha":10/700, 
    "beta":200/700,
    "gamma":1,

    "root":"../../../DataSet-images/STARE_sim/train/",
    "in":"img",
    "out":"bg"
}
def run():
    config=config_XCAD
    devessel = DeVessel(config)
    root=config["root"]
    inPath = os.path.join(root, config["in"])
    outPath = os.path.join(root, config["out"])
    if not os.path.exists(outPath):  # 检查文件夹是否存在
        os.makedirs(outPath)  # 创建文件夹

    listName = os.listdir(inPath)
    for i in range(len(listName)):
        print(i,"\t",len(listName))
        filename = listName[i]
        file_path = os.path.join(inPath, filename)  # 获取完整路径

        # devessel = DeVessel(file_path)
        devessel.process(file_path,os.path.join(outPath, filename))

def test():
    name="im0001.ppm.jpg"
    inpath="./test/img/"+name
    devessel = DeVessel(config_XTARE)
    devessel.process(inpath,"./test/img/test_bg.jpg")

if __name__ == '__main__':
    run()

