import numpy as np
import PIL.Image as Image
import os.path
import cv2
import random

class GetTree():
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
        self.start()

    def __merge(self,image,image_rand):
        img1=image
        img2=image_rand
        # 检查图片形状是否正确
        if img1.shape != (512, 512) or img2.shape != (512, 512):
            raise ValueError("图片的形状必须为 (512, 512)")
        img_merge = np.minimum(img1, img2)
        return img_merge
    def __merge2(self,image,image_rand,bg,bg_rand):
        img1=self.load(image)
        img2=self.load(image_rand)
        bg1=self.load(bg)
        bg2=self.load(bg_rand)
        def get_rand( target_shape=(512, 512), scale_factor=1/2):
            # 缩放图像
            scaled_shape = (int(target_shape[0] * scale_factor), int(target_shape[1] * scale_factor))
            
            # 计算需要填充的边距
            height, width = scaled_shape
            target_height, target_width = target_shape
            top_pad = np.random.randint(0, target_height - height + 1)
            bottom_pad = target_height - height - top_pad
            left_pad = np.random.randint(0, target_width - width + 1)
            right_pad = target_width - width - left_pad
            return (top_pad, bottom_pad, left_pad, right_pad,scaled_shape)
        def random_pad_image(image, padding_param1,value):
            top_pad, bottom_pad, left_pad, right_pad,scaled_shape = padding_param1
            # 缩放图像
            scaled_image = cv2.resize(image, scaled_shape, interpolation=cv2.INTER_NEAREST)
            
            # 填充图像
            padded_image = cv2.copyMakeBorder(scaled_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=value)
            
            return padded_image
        # 随机生成填充参数
        padding_param1 = get_rand(scale_factor=self.config["scale_factor"])  # 只获取填充参数

        # 使用相同的填充参数对两张图像进行填充
        img1 = random_pad_image(img1,padding_param1,255)
        bg1 = random_pad_image(bg1 ,padding_param1,0)
        

        padding_param2 = get_rand(scale_factor=self.config["scale_factor"])  # 只获取填充参数
        img2 = random_pad_image(img2,padding_param2,255)
        bg2  = random_pad_image(bg2 ,padding_param2,0)
        
        
        return np.minimum(img1, img2), np.maximum(bg1, bg2) # np.minimum(bg1, bg2)


    def start(self):
        
        # 设置文件夹路径
        path_in_gray = self.config["root"]+self.config["in1"]  
        path_out_gray = self.config["root"]+self.config["out1"]  
        path_in_bg = self.config["root"]+self.config["in2"]  
        path_out_bg = self.config["root"]+self.config["out2"]  

        # 获取文件夹中所有文件
        files = os.listdir(path_in_gray)

        # 遍历文件夹中的每个文件
        for file in files:
            # 检查文件是否是图片格式（可根据需要扩展支持的格式）
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
  
                
                # 使用 OpenCV 读取图片
                # image = np.ones((512, 512), dtype=np.float32)
                file1=file
                file2=random.choice(files)
                image      = os.path.join(path_in_gray, file1) 
                image_rand = os.path.join(path_in_gray, file2) 
                bg      = os.path.join(path_in_bg, file1) 
                bg_rand = os.path.join(path_in_bg, file2) 
                # image_merge = self.__merge(image,image_rand)
                image_merge,bg_merge = self.__merge2(image,image_rand,bg,bg_rand)
                self.save(image_merge,os.path.join(path_out_gray, file))
                self.save(bg_merge   ,os.path.join(path_out_bg  , file))
                exit(0)
                # 检查图片是否成功读取
                if image is not None:
                    # 输出图片的 NumPy 格式数据
                    print(f"图片 {file} 的 NumPy 格式数据：")
                    print(image)
                    print(f"图片 {file} 的形状：{image.shape}")
                    print(f"图片 {file} 的数据类型：{image.dtype}")
                else:
                    print(f"无法读取图片 {file}")





config={

    "root":"../../../DataSet-images/30XCA/train/",
    "in1" :"fake_grayvessel_bend",
    "out1":"fake_grayvessel_bend_tree",
    "in2" :"fake_gtvessel_bend",
    "out2":"fake_gtvessel_bend_tree",

    "scale_factor":0.85
}

def run():
    GetTree(config)


if __name__ == '__main__':
    run()