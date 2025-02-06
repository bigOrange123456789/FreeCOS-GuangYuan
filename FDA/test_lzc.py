
import numpy as np
import cv2
import PIL.Image as Image
import os.path
import torch

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    # print("amp_src",amp_src.shape,type(amp_src),amp_src)
    # print("amp_trg",amp_trg.shape,type(amp_trg),amp_trg)
    # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
    # print("a_src",a_src.shape,a_trg.shape)

    _, h, w = a_src.shape # h=512 w=512
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int) # b=153
    c_h = np.floor(h/2.0).astype(int) # 256
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # 傅里叶变换 # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )
    '''
    numpy库中用于计算二维快速傅里叶变换的函数。
        axes参数指定了要进行FFT运算的轴。
            axes=(-2, -1)表示对输入数组的倒数第二维和倒数第一维进行FFT运算。
            对应于(0, 1),即对整个图像进行二维FFT。
            对于彩色图像,对每个颜色通道分别进行二维FFT,而不会对通道之间进行FFT运算。
        输出：复数数组。每个元素表示输入图像的一个频率分量，其幅度和相位分别表示该频率分量的强度和相位。
    '''

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # 逆傅里叶变换 # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
    return src_in_trg

def low_freq_mutate_np2( amp_src, amp_trg, L=128 ): #L=0~255
    # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
    # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

    _, h, w = a_src.shape # h=512 w=512
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int) # b=153
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np2( src_img, trg_img, L=128 ):
    # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # 傅里叶变换 # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np2( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # 逆傅里叶变换 # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
    return src_in_trg


def low_freq_mutate_np3( amp_src, amp_trg, L=128 ): #L=0~255
    # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
    # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

    _, h, w = a_src.shape # h=512 w=512
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int) # b=153
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src[:,h1:h2,w1:w2] = a_trg # a_trg[:,0:255,0:255]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np3( src_img, trg_img, L=128 ):
    # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # 傅里叶变换 # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np3( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # 逆傅里叶变换 # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
    return src_in_trg

class Test_lzc():

    def __init__(self):
        root="../../DataSet-images/XCAD_FreeCOS/train/"
        self.img_path        = os.path.join(root, "fake_grayvessel_width")
        self.background_path = os.path.join(root, "img")
        self.ann_path        = os.path.join(root, "fake_gtvessel_width")
        
        self.load_frame_fakevessel_gaussian04("0.png","000_PPA_44_PSA_00_8.png")

    def read_img(self, img_name):
        # maybe png
        return Image.open(os.path.join(self.img_path, img_name)).convert('L')
    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')
    def read_mask(self, img_name):
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name)).convert('L'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask


    def load_frame_fakevessel_gaussian01(self,img_name,background_name):
        img = self.read_img(img_name) # 读取人工血管图
        anno_mask = self.read_mask(img_name) # 读取人工血管的标签图
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>

        # 1.FDA：在人工血管图中添加背景
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_src = np.asarray(img, np.float32) # <PIL.Image.Image> -- <numpy.ndarray> #转换为NumPy，并且指定类型

        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_src = np.expand_dims(im_src, axis=2) # (512, 512) -> (512, 512, 1) #增加一个维度
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)

        im_src = im_src.transpose((2, 0, 1)) #  (512, 512, 1) -> (1, 512, 512)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
        # 2.高斯模糊
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)
        '''
        进行高斯模糊处理。 (注意这里有一个拼写错误，“guassian”应该是“gaussian”)
            img_FDA：原始图像。
            (13, 13)：高斯核的大小。在这个例子中，核的大小是13x13像素。核的大小会影响模糊的程度；较大的核会产生更强烈的模糊效果。
            0：表示σ值在X和Y方向上的标准差。当这个参数为0时，它会根据核的大小自动计算。标准差决定了高斯函数的宽度，从而影响模糊的程度。
        '''
        # 3.添加点噪声
        noise_map = np.random.uniform(-5,5,img_FDA_guassian.shape)
        '''
        生成一个与img_FDA_guassian图像形状相同的噪声，其中的元素值在-5到5之间均匀分布:
            [-5,5)：抽取样本的范围。
            img_FDA_guassian.shape：输出数组的形状。
        '''
        img_FDA_guassian = img_FDA_guassian + noise_map
        img_FDA_guassian = np.clip(img_FDA_guassian, 0, 255.)
        '''
            img_FDA_guassian [[ ....]] 
            type=<class 'numpy.ndarray'> 
            shape=(512, 512)
        '''

        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')
        '''
        img_FDA_guassian.astype('uint8')：转换为无符号8位整型（uint8）。
            这是必要的，因为PIL库处理图像时通常期望图像数据是以这种格式存储的。
            以确保所有的值都在0到255的范围内（uint8能表示的最小值和最大值）。
        Image.fromarray(...)：这一步使用PIL库的Image.fromarray函数将NumPy数组转换为PIL图像对象。
            这个函数接受一个NumPy数组作为输入，并返回一个PIL图像对象，该对象可以用于进一步的图像处理或保存。
        .convert('L')：这一步将PIL图像对象转换为灰度图像。
            如果原始图像已经是灰度图像，这一步则不会改变图像的内容。
        '''

        org_img_size = img.size


        output_path = './temp/output_image_01.png'
        img_FDA_Image.save(output_path, format='PNG')
    
    def load_frame_fakevessel_gaussian02(self,img_name,background_name):
        img = self.read_img(img_name) # 读取人工血管图
        anno_mask = self.read_mask(img_name) # 读取人工血管的标签图
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>

        # 1.FDA：在人工血管图中添加背景
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_src = np.asarray(img, np.float32) # <PIL.Image.Image> -- <numpy.ndarray> #转换为NumPy，并且指定类型

        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_src = np.expand_dims(im_src, axis=2) # (512, 512) -> (512, 512, 1) #增加一个维度
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)

        im_src = im_src.transpose((2, 0, 1)) #  (512, 512, 1) -> (1, 512, 512)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序
        print("im_trg",im_trg.shape,type(im_trg))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
        
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        output_path = './temp/output_image_02.png'
        img_FDA_Image.save(output_path, format='PNG')

    
        
    def load_frame_fakevessel_gaussian03(self,img_name,background_name):

        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        im_src=np.zeros_like(im_trg)
        N=100
        for i in range(N):
            src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.5*i/N) #源图片是人工血管、目标图片是背景图
            print(i/N)
            img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
            img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
            img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
            img_FDA_Image.save('./temp/t/'+str(i)+'.png', format='PNG')
        
    def deVessel(self,im,L,name):
        im_0=np.zeros_like(im)
        src_in_trg = FDA_source_to_target_np2(im_0, im, L=0.5*L) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save(name, format='PNG')
    def load_frame_fakevessel_gaussian04(self,img_name,background_name):
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.99)+'.png')
        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.999)+'.png')
        N=255
        for i in range(N):
            print("test",i)
            self.deVessel(im_trg,i*2,'./temp/t/'+str(i)+'.png')
        
class Test_lzc2():

    def __init__(self):
        root="../../DataSet-images/XCAD_FreeCOS/train/"
        self.img_path        = os.path.join(root, "fake_grayvessel_width")
        self.background_path = os.path.join(root, "img")
        self.ann_path        = os.path.join(root, "fake_gtvessel_width")
        
        self.load_frame_fakevessel_gaussian04("0.png","000_PPA_44_PSA_00_8.png")

    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def low_freq_mutate_np(self, amp_src, amp_trg, L=128 ): #L=0~255
        # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
        a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
        # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

        _, h, w = a_src.shape # h=512 w=512
        b = (  np.floor(np.amin((h,w))*L)  ).astype(int) # b=153
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1
        assert h1==w1
        assert h2==w2
        h1 = 0
        h2 = 511
        w1 = h1
        w2 = h2
        print(h1,h2)


        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
        a_src = a_trg
        # a_src[:,h1:h2,w1:w2] = a_trg # a_trg[:,0:255,0:255]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src

    def FDA_source_to_target_np(self, src_img, trg_img, L=128 ):
        # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

        src_img_np = src_img #.cpu().numpy()
        trg_img_np = trg_img #.cpu().numpy()

        # 傅里叶变换 # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np( amp_src, amp_trg, L=L )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
        return src_in_trg
    
    def deVessel(self,im,L,name):
        im_0=np.zeros_like(im)
        src_in_trg = self.FDA_source_to_target_np(im_0, im, L=0.5*L) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save(name, format='PNG')
    def load_frame_fakevessel_gaussian04(self,img_name,background_name):
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.99)+'.png')
        self.deVessel(im_trg,1,'./temp/t/'+str(1)+'.png')
        return
        N=255
        for i in range(N):
            print("test",i)
            self.deVessel(im_trg,i*2,'./temp/t/'+str(i)+'.png')    

class Test_lzc3():

    def __init__(self):
        root="../../DataSet-images/XCAD_FreeCOS/train/"
        self.img_path        = os.path.join(root, "fake_grayvessel_width")
        self.background_path = os.path.join(root, "img")
        self.ann_path        = os.path.join(root, "fake_gtvessel_width")
        
        self.load_frame_fakevessel_gaussian04("0.png","000_PPA_44_PSA_00_8.png")

    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def low_freq_mutate_np(self, amp_src, amp_trg, L=128 ): #L=0~255
        # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
        a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
        # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

        _, h, w = a_src.shape # h=512 w=512
        b = (  np.floor(np.amin((h,w,L)))  ).astype(int) # b=153
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1
        assert h1==w1
        assert h2==w2
        # h1 = 0
        # h2 = 511
        # w1 = h1
        # w2 = h2
        # print(h1,h2)


        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
        a_src = a_trg
        # a_src[:,h1:h2,w1:w2] = a_trg # a_trg[:,0:255,0:255]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src

    def FDA_source_to_target_np(self,  trg_img, L=128 ):
        # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

        src_img_np = np.zeros_like(trg_img) #src_img #.cpu().numpy() #0
        trg_img_np = trg_img #.cpu().numpy()

        # 傅里叶变换 # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)


        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np( np.zeros_like(amp_trg), amp_trg, L=L )
        pha_src_ = self.low_freq_mutate_np( np.zeros_like(pha_trg), pha_trg, L=L )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
        return src_in_trg
    
    def deVessel(self,im,L,name):
        src_in_trg = self.FDA_source_to_target_np( im, L=L) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save(name, format='PNG')
    def load_frame_fakevessel_gaussian04(self,img_name,background_name):
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.99)+'.png')
        # self.deVessel(im_trg,1,'./temp/t/'+str(1)+'.png')
        # return
        N=255
        for i in range(N):
            print("test",i)
            self.deVessel(im_trg,i,'./temp/t/'+str(i)+'.png')   
class Test_lzc4():

    def __init__(self):
        root="../../DataSet-images/XCAD_FreeCOS/train/"
        self.img_path        = os.path.join(root, "fake_grayvessel_width")
        self.background_path = os.path.join(root, "img")
        self.ann_path        = os.path.join(root, "fake_gtvessel_width")
        
        self.load_frame_fakevessel_gaussian04("0.png","000_PPA_44_PSA_00_8.png")

    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def low_freq_mutate_np(self, amp_src, amp_trg, L=128 ): #L=0~255
        # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
        a_src = np.zeros_like(amp_trg)
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
        # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

        _, h, w = a_src.shape # h=512 w=512
        b = (  np.floor(np.amin((h,w,L)))  ).astype(int) # b=153
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1
        assert h1==w1
        assert h2==w2
        # h1 = 0+255
        # h2 = 511-255
        # w1 = h1
        # w2 = h2
        # print(h1,h2)


        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
        # a_src = a_trg
        # a_src[:,h1:h2,w1:w2] = a_trg # a_trg[:,0:255,0:255]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src

    def FDA_source_to_target_np(self,  trg_img, L=128 ):
        # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

        # 傅里叶变换 # get fft of both source and target
        fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)


        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np( np.zeros_like(amp_trg), amp_trg, L=L )
        pha_src_ = self.low_freq_mutate_np( np.zeros_like(pha_trg), pha_trg, L=L )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
        return src_in_trg
    
    def deVessel(self,im,L,name):
        src_in_trg = self.FDA_source_to_target_np( im, L=L) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save(name, format='PNG')
    def load_frame_fakevessel_gaussian04(self,img_name,background_name):
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.99)+'.png')
        # self.deVessel(im_trg,1,'./temp/t/'+str(1)+'.png')
        # return
        N=257
        for i in range(N):
            print("test",i)
            self.deVessel(im_trg,i,'./temp/t/'+str(i)+'.png')   
            # exit(0)
            
class Test_lzc5():

    def __init__(self):
        root="../../DataSet-images/XCAD_FreeCOS/train/"
        self.img_path        = os.path.join(root, "fake_grayvessel_width")
        self.background_path = os.path.join(root, "img")
        self.ann_path        = os.path.join(root, "fake_gtvessel_width")
        
        self.load_frame_fakevessel_gaussian04("0.png","000_PPA_44_PSA_00_8.png")

    def read_background(self,img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def low_freq_mutate_np(self, amp_src, amp_trg, L=128 ): #L=0~255
        # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
        a_src = np.zeros_like(amp_trg)
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
        # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。

        _, h, w = a_src.shape # h=512 w=512
        b = (  np.floor(np.amin((h,w,L)))  ).astype(int) # b=153
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1
        assert h1==w1
        assert h2==w2

        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
        # a_src = a_trg
        # a_src[:,h1:h2,w1:w2] = a_trg # a_trg[:,0:255,0:255]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src

    def FDA_source_to_target_np(self,  trg_img, L=128 ):
        # 0~255  (1, 512, 512) <class 'numpy.ndarray'>

        # 傅里叶变换 # get fft of both source and target
        fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)


        # mutate the amplitude part of source with target
        amp_src_ = amp_trg # self.low_freq_mutate_np( np.zeros_like(amp_trg), amp_trg, L=L )
        pha_src_ = self.low_freq_mutate_np( np.zeros_like(pha_trg), pha_trg, L=L )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src_ )

        # 逆傅里叶变换 # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg) # 从复数数组中提取实部。
        return src_in_trg
    
    def deVessel(self,im,L,name):
        src_in_trg = self.FDA_source_to_target_np( im, L=L) #源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)#应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA,axis = 0) # (1, 512, 512) -> (512, 512)
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save(name, format='PNG')
    def load_frame_fakevessel_gaussian04(self,img_name,background_name):
        background_img = self.read_background(background_name) #背景图(真实造影图) # <PIL.Image.Image>
        background_array = np.array(background_img) # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_trg = np.asarray(background_array, np.float32) #转化前后类型都是<numpy.ndarray> (512, 512)
        im_trg = np.expand_dims(im_trg, axis=2) # (512, 512) -> (512, 512, 1)
        im_trg = im_trg.transpose((2, 0, 1)) # 通过转置操作，改变维度顺序

        self.dePhase(im_trg)
        exit(0)

        # self.deVessel(im_trg,0.99,'./temp/t/'+str(0.99)+'.png')
        # self.deVessel(im_trg,1,'./temp/t/'+str(1)+'.png')
        # return
        N=257
        for i in range(N):
            print("test",i)
            self.deVessel(im_trg,i,'./temp/t/'+str(i)+'.png')   
            # exit(0)
    
    def dePhase(self,img):
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
        # return img_FDA
            
        img_FDA_Image=Image.fromarray((img_FDA).astype('uint8')).convert('L')
        img_FDA_Image.save("test.png", format='PNG')
    

Test_lzc4()

