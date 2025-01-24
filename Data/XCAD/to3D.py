import numpy as np
import random
import math
from PIL import Image
import os



class ProcessImage():
    def __init__(self, image):
        self.label_bg=255
        # 将PIL.Image.Image对象转换为NumPy数组
        self.image = np.array(image)
        self.h=self.image.shape[0]
        self.w=self.image.shape[1]
        # print(self.image.shape)
        self.initThickness()
        self.initLight()
    def isBg(self,x):
         bgColorList=[255]
         return x in bgColorList
    def isVessel(self,y,x):
        bgColorList=[255]
        # if x-1>0:       flag10=(not self.image[y,x-1] in bgColorList)
        # else:           flag10=False
        # if x+1<self.w:  flag12=(not self.image[y,x+1] in bgColorList)
        # else:           flag12=False

        # if y-1>0:       flag01=(not self.image[y-1,x] in bgColorList)
        # else:           flag01=False
        # if y+1<self.h:  flag21=(not self.image[y+1,x] in bgColorList)
        # else:           flag21=False
        
        # flag11=(not self.image[y,x] in bgColorList)
        # return flag10 or flag11 or flag12 or flag01 or flag21
        return (not self.image[y,x] in bgColorList)

    def calculate_directional_width(self, x, y, angle):
        """
        计算每个像素点处曲线在指定方向上的宽度。
        输入：
            self.image: 二维灰度图像,numpy数组,曲线区域值小于1,背景区域值等于1。
            angle: 指定方向的倾斜角度（以度为单位），正角度表示逆时针旋转。
            x,y:检测像素点的坐标
        输出：
            width_image: 二维数组，每个像素点的值表示该点处曲线在指定方向上的宽度。
        """
        image=self.image
        height=self.h
        width = self.w

        # width_image = np.zeros_like(image, dtype=np.float32)  # 创建一个与原图相同大小的全零数组

        # 将角度转换为弧度
        angle_rad = np.radians(angle)
        # 计算方向向量
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        
        directional_width=0
        directional_coord=0
        if self.isVessel(y, x) :  # 如果当前像素属于曲线区域
                    # 沿着指定方向扩展
                    distance = 0
                    current_x, current_y = x, y
                    while (0 <= current_x < width and 0 <= current_y < height and
                        self.isVessel(int(current_y), int(current_x)) ):
                        current_x += direction_vector[0]
                        current_y += direction_vector[1]
                        distance += 1
                    current_x1, current_y1=current_x, current_y

                    # 沿着相反方向扩展
                    current_x, current_y = x, y
                    while (0 <= current_x < width and 0 <= current_y < height and
                        self.isVessel(int(current_y), int(current_x)) ):
                        current_x -= direction_vector[0]
                        current_y -= direction_vector[1]
                        distance += 1
                    current_x2, current_y2=current_x, current_y

                    # 计算方向宽度
                    directional_width = distance
                    # width_image[y, x] = directional_width
                    center_x=(current_x1+current_x2)/2
                    center_y=(current_y1+current_y2)/2
                    directional_coord=( (x-center_x)**2+(y-center_y)**2 )**0.5

        return directional_width,directional_coord
    def calculate_directional0_width(self, x, y):
        horizontal_w=0
        coord_w=0
        if not self.isBg(self.image[y, x]):  # 如果当前像素属于曲线区域
            # 向左扩展
            left = x
            while left > 0 and not self.isBg(self.image[y, left - 1]):
                left -= 1
                    
            # 向右扩展
            right = x
            while right < self.w - 1 and not self.isBg(self.image[y, right + 1]):
                right += 1
                    
            # 计算横向宽度
            horizontal_w = right - left + 1
            center_w=(right+left)/2
            coord_w=abs(x-center_w)
        return horizontal_w,coord_w
    
    def calculate_directional90_width(self, x, y):
        horizontal_h=0
        coord_h=0
        if not self.isBg(self.image[y, x]):  # 如果当前像素属于曲线区域
                    # 向下扩展
                    down = y
                    while down > 0 and not self.isBg(self.image[down - 1,x]):
                        down -= 1
                    
                    # 向右扩展
                    up = y
                    while up < self.h - 1 and not self.isBg(self.image[down + 1,x]):
                        up += 1
                    
                    # 计算横向宽度
                    horizontal_h = up - down + 1
                    center_h=(up+down)/2
                    coord_h=abs(y-center_h)

        return horizontal_h,coord_h
    def initThickness(self):
        radius    = np.zeros((self.h, self.w))#np.zeros_like(self.image)
        distance  = np.zeros((self.h, self.w))#点(x,y)到圆心的距离
        Num=8
        # 遍历每个像素点
        for y in range(self.h):
            for x in range(self.w):
                # print(self.image[y, x])
                if self.isVessel(y, x):  # 如果当前像素属于曲线区域
                    
                    horizontal0=float('inf') #浮点数最大值
                    distance0=0
                    if True: #最小值
                        for i in range(Num):
                            angle=i*180/Num
                            horizontal_angle,coord_angle=self.calculate_directional_width(x, y, angle)
                            if horizontal_angle<horizontal0:
                                horizontal0=horizontal_angle
                                distance0=coord_angle
                    elif False:#均值
                        radius_list=[]
                        distance_list=[]

                        for i in range(Num):
                            angle=i*180/Num
                            horizontal_angle,coord_angle=self.calculate_directional_width(x, y, angle)
                            radius_list.append(horizontal_angle)
                            distance_list.append(coord_angle)
                        horizontal0=np.mean(radius_list)
                        distance0=np.mean(distance_list)
                    else:#自定义权重
                        radius_list=[]
                        distance_list=[]

                        for i in range(Num):
                            angle=i*180/Num
                            horizontal_angle,coord_angle=self.calculate_directional_width(x, y, angle)
                            radius_list.append(horizontal_angle)
                            distance_list.append(coord_angle)
                        
                        # 获取排序索引
                        sorted_indices = np.argsort(radius_list)

                        # 根据排序索引重新排列values
                        radius_list = np.array(radius_list)[sorted_indices]
                        distance_list = np.array(distance_list)[sorted_indices]

                        weight=[32,4,1,0.2, 0.01,0,0,0]
                        if len(weight)!=Num:
                             print("weight的长度必须和Num相等!")
                             exit(0)
                        weight=np.array(weight)/np.sum(weight)

                        horizontal0=0
                        distance0=0
                        for i in range(Num):
                            horizontal0=horizontal0+weight[i]*radius_list[i]
                            distance0=distance0+weight[i]*distance_list[i]

                        
                    # horizontal_h,coord_h=self.calculate_directional_width(x, y, 90)
                    # horizontal_h,coord_h=self.calculate_directional0_width(x, y)
                    # print(horizontal_h,coord_h)
                    radius[y, x ]   = horizontal0
                    distance[y, x ] = distance0
        self.thickness  = 2*(radius**2-distance**2)**0.5
    def initLight(self):
         light0=1
         delta=random.uniform(0.01, 0.02) #0.02 #0.01
         self.light=light0*np.exp(-1*self.thickness*delta)
        #  print(self.light)

def saveImg(img,path):
    Image.fromarray((img).astype('uint8')).convert('L').save(path)


import time
# 记录开始时间
start_time = time.time()

Img_dir = "./train/fake_grayvessel_width//"
Save_dir  ="./train/vessel3D_lzc//"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
files = os.listdir(Img_dir)
i = 0
list_RGB= []
for image_dir in files:
    i=i+1
    print(str(i)+"/"+str(len(files)),"\timage_dir",image_dir)
    image =  Image.open(Img_dir+image_dir).convert("L")

    myProcess=ProcessImage(image)
    # saveImg(path)
    if False:
        saveImg(myProcess.thickness,'thickness.jpg')
        saveImg(255*myProcess.light,'light.jpg')
        exit(0)
    saveImg(255*myProcess.light,Save_dir + image_dir)
    # exit(0)
    # image_array = np.asarray(image)
    # print("image_array",image_array.shape)
    # save_path = Save_dir + image_dir
    # Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)
print('后续要通过与make_fakevessel.py相结合来提高生成图片的质量')

# 记录结束时间
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.6f} 秒")