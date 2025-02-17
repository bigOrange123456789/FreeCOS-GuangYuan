
import numpy as np
import PIL.Image as Image
import os.path


class Test_lzc6():
    def save(self,img,path):
        img_FDA_Image=Image.fromarray((img).astype('uint8')).convert('L')
        img_FDA_Image.save(path, format='PNG')

    def __init__(self):
        root="../../../DataSet-images/XCAD_FreeCOS/train/"
        self.background_path = os.path.join(root, "img")
        self.background_path2 = os.path.join(root, "img2")
        if not os.path.exists(self.background_path2):  # 检查文件夹是否存在
            os.makedirs(self.background_path2)  # 创建文件夹

        listName = os.listdir(self.background_path)
        for i in range(len(listName)):
            print(i,"\t",len(listName))
            filename = listName[i]
            file_path = os.path.join(self.background_path, filename)  # 获取完整路径

            background_img = Image.open( file_path ).convert('L') #背景图(真实造影图) # <PIL.Image.Image>
            background_array = np.array(background_img)
            im_trg = np.asarray(background_array, np.float32) 

            # img_dePhase=self.dePhase(im_trg)
            # self.save(img_dePhase, "img_dePhase.png")

            # img_deVessel1=self.deVessel(im_trg)
            # self.save(img_deVessel1,"img_deVessel1.png")

            # img_deVessel2=self.deVessel2(im_trg,img_dePhase)
            # self.save(img_deVessel2,"img_deVessel2.png")

            img_deVessel3=self.deVessel3(im_trg)
            self.save(img_deVessel3,os.path.join(self.background_path2, filename))

            # vessel = Image.open(os.path.join("vessel.png")).convert('L')
            # vessel = np.asarray(vessel, np.float32)

            # result=255.*(img_deVessel2/255. )*(vessel/255.)
            # self.save(result,"result.png")

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
     
    def deVessel3(self,img):
        ##########################    1.计算noise   ##########################
        img = np.expand_dims(img, axis=2) # (512, 512) -> (512, 512, 1)
        img = img.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
            
        fft_trg_np = np.fft.fft2( img, axes=(-2, -1) )# 傅里叶变换

        # extract amplitude and phase of both ffts
        amp, pha = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutated fft of source
        fft_0 = amp * np.exp( 1j * np.zeros_like(pha) )

        # 逆傅里叶变换 # get the mutated image
        src_0 = np.fft.ifft2( fft_0, axes=(-2, -1) )
        src_0 = np.real(src_0) # 从复数数组中提取实部。
            
        noise = np.clip(src_0, 0, 255.)#限制像素的最小值为0、最大值为255
        ##########################    2.计算noise   ##########################
        # 傅里叶变换 # get fft of both source and target
        fft_trg_np2 = np.fft.fft2( noise, axes=(-2, -1) )

        def low_freq_mutate_np2(amp_src1,amp_src2): #L=0~255
            a_src1 = np.fft.fftshift( amp_src1, axes=(-2, -1) ) # np.zeros_like(amp_trg)
            a_src2 = np.fft.fftshift( amp_src2, axes=(-2, -1) ) 
            # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
            _, h, w = a_src1.shape # h=512 w=512       
            for i in range(h):
                    for j in range(w):
                        k = ( (i-h/2)**2 + (j-w/2)**2 )**0.5
                        if k>10 and k<100 : 
                            a_src1[0,i,j]=a_src2[0,i,j]
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
       
Test_lzc6()
