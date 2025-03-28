import numpy as np
import random
# random.seed(42)#感觉这个随机数种子没有起作用
from PIL import Image
from scipy.ndimage import label, generate_binary_structure

from BendFun import BendFun
config_XCAD={
    "inferenceSize":{
        "w":1024,
        "h":1024
    },
    "resultSize":{
        "w":512,
        "h":512
    },
    "ratioMin":2.5/100
}
config_XCAD_Test={
    "inferenceSize":{
        "w":1024,
        "h":1024
    },
    "resultSize":{
        "w":512,
        "h":512
    },
    "ratioMin":2.5/100
}
config_STARE={
    "inferenceSize":{
        "w":1400,
        "h":1210
    },
    "resultSize":{
        "w":700,
        "h":605
    },
    "ratioMin":2.5/100
}
config_STARE_Test={
    "inferenceSize":{
        "w":700,
        "h":610
    },
    "resultSize":{
        "w":350,
        "h":305
    }
}
config = config_STARE
TestFlag = False #是否快速生成低质量图片
#######################   开始创建一个numpy对象的图片   #########################
class Img():
    def __init__(self, width=1024, height = 1024):
        # if TestFlag:
        #     width=552
        #     height=552
        width  = config["inferenceSize"]["w"]
        height = config["inferenceSize"]["h"]
    # def __init__(self, width=512*2+10, height = 512*2+10):
        self.image = np.zeros((height, width), dtype=np.uint8) #标签图
        # self.radius = np.zeros((width, height), dtype=np.float32)#半径图，用于后续计算光影图
        # self.distance = np.zeros((width, height), dtype=np.float32)
        # self.thickness = np.zeros((width, height), dtype=np.uint8) 
        self.thickness = np.zeros((height, width))
        # self.light = np.zeros((width, height), dtype=np.float32)#光影图
        self.end_x=None
        self.end_y=None
    # def setThichness(self):

    def find_center_of_pattern(self):
        """
        找到图案的中心点。
        
        参数:
            image: numpy数组,表示图像。
        
        返回:
            center_x, center_y: 图案的中心点坐标。
        """
        # 找到所有非零像素点的坐标
        print(type(self.image),np.sum(self.image))
        y_coords, x_coords = np.nonzero(self.image)
        
        # 计算中心点
        print("x_coords",x_coords)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return center_x, center_y
    
    def extract_rectangle(self, center_x, center_y):
        size_w = config["resultSize"]["w"]
        size_h = config["resultSize"]["h"]
        half_size_w = size_w // 2
        half_size_h = size_h // 2
        # half_size = size // 2
        height, width = self.image.shape #height:610, width:700
        # print("height, width",height, width)
        # exit(0)
        
        # 计算矩形框的边界
        x_start = center_x - half_size_h # x_start = center_x - half_size
        y_start = center_y - half_size_w # y_start = center_y - half_size
        x_end = center_x + half_size_h # x_end = center_x + half_size
        y_end = center_y + half_size_w # y_end = center_y + half_size

        if x_start < 0 or y_start < 0 or x_end > height or y_end > width:
            print("边界超出图像范围")
            if x_start < 0:
                x_start = 0
                x_end = size_h # x_end = size
            if y_start < 0:
                y_start = 0
                y_end = size_w # y_end = size
            if x_end > height:
                x_end = height
                x_start = height - size_h # x_start = width - size
            if y_end > width:
                y_end = width
                y_start = width - size_w # y_start = height - size
        # 提取矩形框
        # self.image = self.image[y_start:y_end, x_start:x_end]
        # self.thickness = self.thickness[y_start:y_end, x_start:x_end]
        # print("a,self.image",self.image.shape,[width,height])
        self.image = self.image[x_start:x_end, y_start:y_end]
        self.thickness = self.thickness[x_start:x_end, y_start:y_end]
        # print("b,self.image",self.image.shape)
        
        return self.image
    
    def calculate_pixel_ratio(self):
        # 计算颜色为255的像素数量
        num_white_pixels = np.sum(self.image == 255)
        
        # 计算图片的总像素数量
        total_pixels = self.image.size
        
        return num_white_pixels / total_pixels # 计算占比
    
    def getLight(self):
        light0 = 1
        delta = random.uniform(0.009, 0.012) # random.uniform(0.009, 0.0012) # delta = 0.01 # 0.1 # 0.02 # 0.01 #感觉这里应该是一个错误的BUG
        self.light=light0*np.exp(-1*self.thickness*delta)
        return self.light

    def save(self, output_path = "output_image.png", output_path2 = "output_image2.png"): # 保存图像 
        if not TestFlag: # if True: # 
            center_x, center_y = self.find_center_of_pattern() 
            self.extract_rectangle( center_x, center_y) 

        connected = self.is_pattern_connected() 
        ratio = self.calculate_pixel_ratio() 
        # print("连通情况为：",connected) 
        # print("calculate_pixel_ratio",ratio) 
        if (connected and ratio>config["ratioMin"]) or TestFlag: 
            if True:
                pic = Image.fromarray(self.image, mode='L')  # 'L' 表示灰度图像 
                pic.save(output_path) 
            light=self.getLight()
            print('sum',np.sum(light))
            # pic = Image.fromarray((255.*light).astype(np.int32), mode='L')  # 'L' 表示灰度图像 
            # pic2 = Image.fromarray((self.thickness).astype('uint8'), mode='L')  # 'L' 表示灰度图像 
            pic2 = Image.fromarray((255*light).astype('uint8'), mode='L')  # 'L' 表示灰度图像 
            pic2.save(output_path2) 
            print(f"图像已保存到 {output_path2}") 
            return True 
        else: 
            print("图片没有保存") 
            return False 

    def draw(self,new_length0,width,x,y,theta):
        w, h = self.image.shape
        # print(new_length,width,x,y,theta)
        # 77.18 2 -152 -100 81.31 
        # theta=np.radians(45)
        theta=np.radians(theta) # 将角度制转换为弧度值
        x=w/2+x
        y=h/2+y
        self.end_x = x + new_length0 * np.cos(theta)-w/2
        self.end_y = y + new_length0 * np.sin(theta)-h/2
        new_length=new_length0+width/2. #width
        """
            在numpy数组上绘制一条线段。
            参数:
                image: numpy数组,表示图像。
                x, y: 线段起点的坐标。
                new_length: 线段的长度。
                width: 线段的宽度。
                theta: 线段与水平方向的逆时针夹角（以弧度为单位）。
        """
        # 计算线段的两个端点
        end_x = x + new_length * np.cos(theta)
        end_y = y + new_length * np.sin(theta)
        # self.end_x=end_x-h/2
        # self.end_y=end_y-w/2
        
        # 计算线段的中心点
        center_x = (x + end_x) / 2
        center_y = (y + end_y) / 2
        
        # 计算线段的方向向量
        direction_x = np.cos(theta)
        direction_y = np.sin(theta)
        
        # 计算线段的法向量
        normal_x = -direction_y
        normal_y = direction_x
        
        # 遍历图像中的每个像素
        h, w = self.image.shape
        # print("a:",center_x,center_y)
        MaxBendWeight=0.3
        bendWeght=(random.random()*2-1)*MaxBendWeight # 0-1 #[0-0.3]
        bendWeght=bendWeght*0.5*new_length # [ -0.1*l , 0.1*l ]
        # bendWeght=MaxBendWeight*0.5*new_length
        # startLen=0.01#random.random()*0.01
        bendf=BendFun() #用于实现线段弯曲
        for i in range(h):
            for j in range(w):

                # 1.1判断像素点是否在线段的长度范围内
                # 1.2计算像素点到线段起点的向量
                pixel_to_start_x = j - x
                pixel_to_start_y = i - y
                # 1.3计算像素点到线段终点的向量
                pixel_to_end_x = j - end_x
                pixel_to_end_y = i - end_y
                if (pixel_to_start_x * direction_x + pixel_to_start_y * direction_y >= 0 and
                    pixel_to_end_x * direction_x + pixel_to_end_y * direction_y <= 0):

                    ####################开始插入实现弯曲的代码######################
                    be = 0 # 拉伸距离
                    if True: #if not bendWeght==0:
                        def normal(x0,y0):
                            return (x0**2+y0**2)**0.5
                        
                        k = normal(pixel_to_start_x,pixel_to_start_y)/(new_length0) #在直线内的时候：k的数值应该是0-1
                        # k = k+startLen

                        if k>0 and k<1:
                            be = bendf.get(k) * bendWeght
                    center_x2 = center_x + be * normal_x
                    center_y2 = center_y + be * normal_y
                    ####################完成插入实现弯曲的代码######################
                    
                    # 2.1 计算像素点到线段中心点的向量
                    pixel_vector_x = j - center_x2
                    pixel_vector_y = i - center_y2
                    # 2.2 计算像素点到线段的投影距离
                    projection_length = abs(pixel_vector_x * normal_x + pixel_vector_y * normal_y)
                    # 2.3 判断像素点是否在线段的宽度范围内
                    if projection_length <= width / 2:    

                        
                        # print(i00,j00)
                        self.image[i, j] = 255 # self.image[i, j] = 255  # 设置像素值为255（白色）
                        
                        distance = projection_length
                        radius   = width/2
                        thickness = 2*(radius**2-distance**2)**0.5
                        self.thickness[i,j] = max(self.thickness[i,j],thickness) # self.thickness[i,j] = max(self.thickness[i,j],thickness)#255.#

    def is_pattern_connected(self):
        # 将图像转换为二值图像（0为背景，1为图案）
        binary_image = (self.image == 255).astype(np.uint8)
            
        # 定义连通性结构（8连通）
        structure = generate_binary_structure(2, 1)
            
        # 标记连通区域
        labeled_array, num_features = label(binary_image, structure=structure)
        print('num_features:',num_features)
            
        # 如果只有一个连通区域（num_features == 1），则说明所有像素都聚集在一起
        return num_features == 1

#######################   结束创建一个numpy对象的图片   #########################



class LSystem_vessel():
    def __init__(self, axiom, rules, rules_2, rules_3, theta=0, width=5, dtheta_1=40, dtheta_2=30, start=(-350,0), length=80,iteration=3,width_1=0.79, width_2 = 0.5):
        self.sentence = axiom   # 基本规则 # axiom的本义是原理、公理
        self.rules = rules      # 规则1
        self.relus_2 = rules_2  # 规则2
        self.rules_3 = rules_3  # 规则3
        self.iteration = iteration # 迭代次数
        print("self.sentence",  self.sentence)
        print("self.rules",     self.rules)
        print("self.relus_2",   self.relus_2)
        print("self.rules_3",   self.rules_3)
        print("self.iteration", self.iteration)
        self.width = width  # 初始宽度
        self.theta = theta  # 初始角度
        self.dtheta_1 = dtheta_1 # 偏转角度？
        self.dtheta_2 = dtheta_2 # 偏转角度？
        self.length = length #长度
        self.positions = []
        self.start = start
        self.lamda_1 = width_1 #
        self.lamda_2 = width_2

        self.x, self.y = start # 初始位置
        self.img=Img() #lzc

    def __str__(self): # 感觉这个函数应该是用于print语句
        return self.sentence

    def generate(self): #通过迭代算法来更新规则语句
        '''
        初始语句有2种情况，迭代规则有3种情况，因此：
         迭代次数为1时的结果种类数量=2*3
         迭代次数为2时的结果种类数量=2*3*3
         总共的种类数量=2*3+2*3*3=24种
        '''
        self.x, self.y = self.start # 初始位置
        # print('self.iteration',self.iteration)
        for iter in range(self.iteration): #迭代次数
            newStr = ""
            # print('iter:',iter,'self.sentence:',self.sentence)
            '''
                self.sentence: 
                    
                    F
                    F-F[+F-F][-F+F]
                    
                    F
                    F[+F]+F[+F]+F[+F]
                    
                    F
                    F-F-F[+F-F][-F-F]F+F
                    
                    [+F-F][-F]
                    [+F-F[+F-F][-F+F]-F-F[+F-F][-F+F]][-F-F[+F-F][-F+F]]
                    
                    [+F-F][-F]
                    [+F[+F]+F[+F]+F[+F]-F[+F]+F[+F]+F[+F]][-F-F-F[+F-F][-F-F]F+F]
            '''
            for char in self.sentence:
                mapped = char
                try: # 如果字符为F就替换为rule字符串，否则保持不变
                    p = random.random()
                    if p>0.4:
                        mapped = self.rules[char]
                    elif 0.8>p>=0.4:
                        mapped = self.rules_3[char]
                    else:
                        mapped = self.relus_2[char]
                except:
                    pass
                newStr += mapped # 更新这个规则
            self.sentence = newStr
        print('self.sentence:',self.sentence)
        return self.sentence

    def draw(self):#根据规则语句来绘制图像
        flag = False
        for char in self.sentence: #规则语句由5种符号组成，分别是'F、+、-、[、]'。
            if char == 'F' or char == 'G': #根据行进轨迹 绘制线段
                if flag == True:
                    new_length = self.length*self.lamda_1#分段后长度改变
                else:
                    new_length = self.length*self.lamda_2

                self.img.draw(new_length, self.width, self.x, self.y, self.theta)
                self.x=self.img.end_x
                self.y=self.img.end_y
                # print('x1:',self.x,self.img.end_x, 'y1:',self.y,self.img.end_y)
            elif char == '+': #调整行进方向
                # dtheta = np.random.randint(low=1, high=5) #
                dtheta = np.random.randint(low=10, high=40) #【根据论文】
                self.theta += dtheta
                self.width = self.width * self.lamda_1#分叉后宽度改变
            elif char == '-': #反方向调整行进方向
                self.width = self.width * self.lamda_2
                # dtheta = np.random.randint(low=1, high=5)  #
                dtheta = np.random.randint(low=10, high=40) #【根据论文】
                self.theta -= dtheta
            elif char == '[': #记录迭代位置 #使用数据栈结构
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length})
                flag = True
            elif char == ']': #返回迭代位置
                position = self.positions.pop()
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
            if self.width<2:#宽度不能低于两像素
                self.width = 2 #self.width = 1.5

# 所谓的规则语句，就是将字符F替换为由F组成的字符串
rules = {"F":"F-F[+F-F][-F+F]"} # 这应该是类似json格式的对象
rules_2 = {"F":"F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"} # 规则是通过字符串来定义的
rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}

path = "[+F-F][-F]" #path变量应该是目前使用的规则
print('path:',path)
# 前期F多后期分支多

Num_image = 1621 # 150 # 8 # 25 #生成图片的数量
Start_theta = (-0,0) # 初始方向为0 # 0为水平向右
Start_theta = (20,120) # 【根据论文】初始角度方向为20~120
Start_theta_2 = (0,0)

# Start_position_x = (-350, -150) # 初始位置的横坐标的范围
# Start_position_y = (-100, -100) # 初始位置的纵坐标坐标为-100
Start_position_x = (0, 0) # 初始位置的横坐标的范围
Start_position_y = (0, 0) # 初始位置的纵坐标坐标为-100

Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1) #宽度的分支衰减率
Ratio_LW = (0.7,0.8) #宽度的分支衰减率【LZC:我的优化】

Dtheta = (20,120) # 每次分岔时，角度的变化范围
Width = (2,12) #initi 宽度的变化范围
# Width = (50,50) #【LZC:我的优化】
Width = (20,20) #【LZC:我的优化】
Width = (20,30) #【LZC:我的优化】
# Length_range = (90, 150) # init range 长度的变化范围
Length_range = (90, 90) #【LZC:我的优化】
Length_range = (90, 100) #【LZC:我的优化】
# Length_range = (90, 150) # init range 长度的变化范围
# 实际上最大宽度为14像素左右
i=0
while i<Num_image: 
# for i  in range(Num_image): #生成Num_image张图片
    p2 = random.random()
    if p2>0.5: #两种基本形状
        path = "[+F-F][-F]" #分叉
    else:
        path = "F" #直线
    # path = "[-F][+F-F]"#失败
    # path = "[+F-F][-F]"#失败
    # path = "F"#失败
    p_vessel = random.random()
    if p_vessel>0.5:
        r1= rules
        r2= rules_2
        r3 = rules_3
    else:
        r1 = rules_4
        r2 = rules_3
        r3 = rules
    print("image_num", i)
    dtheta_1 = np.random.uniform(Dtheta[0],Dtheta[1])
    dtheta_2 = np.random.uniform(Dtheta[0],Dtheta[1])
    init_width = np.random.randint(Width[0],Width[1]+1)
    init_length = np.random.randint(Length_range[0],Length_range[1]+1)
    iteration = np.random.randint(1,3) # 迭代次数为1或2次
    #iteration = 1

    Ratio_lw_1 = np.random.uniform(Ratio_LW[0], Ratio_LW[1])
    Ratio_lw_2 = np.random.uniform(Ratio_LW[0], Ratio_LW[1])

    p = random.random()
    if p>0.5:
        init_theta = np.random.uniform(Start_theta[0], Start_theta[1])
        x_position = np.random.randint(Start_position_x[0],Start_position_x[1]+1)
    else:
        init_theta = np.random.uniform(Start_theta_2[0], Start_theta_2[1])
        x_position = np.random.randint(Start_position_x2[0],Start_position_x2[1]+1)
    y_position = np.random.randint(Start_position_y[0],Start_position_y[1]+1)

    system = LSystem_vessel(path, r1, r2, r3, theta=init_theta, width=init_width, dtheta_1=dtheta_1, dtheta_2=dtheta_2, start=(x_position, y_position),length=init_length, iteration=iteration, width_1=Ratio_lw_1, width_2=Ratio_lw_2)
    sentence=system.generate()
    if len(sentence)<20:
        print("sentence is too short (len="+str(len(sentence))+")",sentence)
        continue
    system.draw()
    # saved=system.img.save(
    #     "./fake_very_smalltheta_label_1621/"+str(i)+'.png',
    #     "./fake_very_smalltheta_1621/"+str(i)+'.png'
    #     )
    saved=system.img.save(
        "./fake_very_smalltheta/"+str(i)+'_label.png',
        "./fake_very_smalltheta/"+str(i)+'.png'
        )
    if saved:
        i=i+1
    # exit(0)