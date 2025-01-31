#from turtle import *
import turtle
import numpy as np
import random
# random.seed(42)#感觉这个随机数种子没有起作用
from PIL import Image


#######################   开始创建一个numpy对象的图片   #########################
class Img():
    # def __init__(self, width=1024, height = 1024):
    def __init__(self, width=800, height = 800):
        self.image = np.zeros((width, height), dtype=np.uint8)
        self.end_x=None
        self.end_y=None

    def find_center_of_pattern(self):
        """
        找到图案的中心点。
        
        参数:
            image: numpy数组,表示图像。
        
        返回:
            center_x, center_y: 图案的中心点坐标。
        """
        # 找到所有非零像素点的坐标
        y_coords, x_coords = np.nonzero(self.image)
        
        # 计算中心点
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return center_x, center_y
    def extract_rectangle(self, center_x, center_y, size=512):
        """
        从图像中提取以中心点为中心的矩形框。
        
        参数:
            image: numpy数组，表示图像。
            center_x, center_y: 中心点坐标。
            size: 矩形框的大小（默认为100）。
        
        返回:
            rectangle: 提取的矩形框。
        """
        # 计算矩形框的边界
        half_size = size // 2
        x_start = max(0, center_x - half_size)
        y_start = max(0, center_y - half_size)
        x_end = min(self.image.shape[1], center_x + half_size)
        y_end = min(self.image.shape[0], center_y + half_size)
        
        # 提取矩形框
        rectangle = self.image[y_start:y_end, x_start:x_end]
        
        # 如果矩形框大小不足100x100，进行填充
        if rectangle.shape[0] < size or rectangle.shape[1] < size:
            padded_rectangle = np.zeros((size, size), dtype=rectangle.dtype)
            padded_rectangle[:rectangle.shape[0], :rectangle.shape[1]] = rectangle
            rectangle = padded_rectangle
        
        return rectangle
    
    def save(self, output_path = "output_image.png"): # 保存图像
        center_x, center_y = self.find_center_of_pattern()
        self.image = self.extract_rectangle( center_x, center_y)

        pic = Image.fromarray(self.image, mode='L')  # 'L' 表示灰度图像
        pic.save(output_path)
        print(f"图像已保存到 {output_path}")

    def draw(self,new_length,width,x,y,theta):
        h, w = self.image.shape
        # print(new_length,width,x,y,theta)
        # 77.18 2 -152 -100 81.31 
        # theta=np.radians(45)
        theta=np.radians(theta)#将角度制转换为弧度值
        x=h/2+x
        y=w/2+y
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
        self.end_x=end_x-h/2
        self.end_y=end_y-w/2
        
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
        for i in range(h):
            for j in range(w):
                # 计算像素点到线段中心点的向量
                pixel_vector_x = j - center_x
                pixel_vector_y = i - center_y
                
                # 计算像素点到线段的投影距离
                projection_length = abs(pixel_vector_x * normal_x + pixel_vector_y * normal_y)
                
                # 判断像素点是否在线段的宽度范围内
                if projection_length <= width / 2:
                    # 计算像素点到线段起点的向量
                    pixel_to_start_x = j - x
                    pixel_to_start_y = i - y
                    
                    # 计算像素点到线段终点的向量
                    pixel_to_end_x = j - end_x
                    pixel_to_end_y = i - end_y
                    
                    # 判断像素点是否在线段的长度范围内
                    if (pixel_to_start_x * direction_x + pixel_to_start_y * direction_y >= 0 and
                        pixel_to_end_x * direction_x + pixel_to_end_y * direction_y <= 0):
                        self.image[i, j] = 255  # 设置像素值为255（白色）

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

    def draw(self, turtle):#根据规则语句来绘制图像
        # turtle的本义是乌龟，感觉这里应该是表示画笔
        turtle.pu()
        turtle.hideturtle()
        turtle.speed(0)
        turtle.goto((self.x, self.y))
        turtle.setheading(self.theta)
        flag = False
        for char in self.sentence: #规则语句由5种符号组成，分别是'F、+、-、[、]'。
            turtle.pd()
            if char == 'F' or char == 'G': #根据行进轨迹 绘制线段
                turtle.pu()
                turtle.setheading(self.theta) # 设定方向
                if flag == True:
                    new_length = self.length*self.lamda_1#分段后长度改变
                else:
                    new_length = self.length*self.lamda_2
                turtle.pensize(self.width) # 设定宽度
                # x, y = turtle.position() #起点
                # theta = self.theta
                # width = self.width
                # makecolor(turtle, new_length, width, x, y, theta) # 长度、宽度、起点、方向
                # self.x, self.y = turtle.position() #更新位置

                self.img.draw(new_length, self.width, self.x, self.y, self.theta)
                self.x=self.img.end_x
                self.y=self.img.end_y
                # print('x1:',self.x,self.img.end_x, 'y1:',self.y,self.img.end_y)
            elif char == '+': #调整行进方向
                # dtheta = np.random.randint(low=1, high=5) #
                dtheta = np.random.randint(low=10, high=40) #【根据论文】
                self.theta += dtheta
                self.width = self.width * self.lamda_1#分叉后宽度改变
                turtle.right(self.theta)
            elif char == '-': #反方向调整行进方向
                self.width = self.width * self.lamda_2
                # dtheta = np.random.randint(low=1, high=5)  #
                dtheta = np.random.randint(low=10, high=40) #【根据论文】
                self.theta -= dtheta
                turtle.left(self.theta)
            elif char == '[': #记录迭代位置 #使用数据栈结构
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length})
                flag = True
            elif char == ']': #返回迭代位置
                turtle.pu()
                position = self.positions.pop()
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
                turtle.goto((self.x, self.y))
                turtle.setheading(self.theta)
            if self.width<2:#宽度不能低于两像素
                self.width = 1.5

# 所谓的规则语句，就是将字符F替换为由F组成的字符串
rules = {"F":"F-F[+F-F][-F+F]"} # 这应该是类似json格式的对象
rules_2 = {"F":"F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"} # 规则是通过字符串来定义的
rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}

path = "[+F-F][-F]" #path变量应该是目前使用的规则
print('path:',path)
# 前期F多后期分支多

Num_image = 3 # 25 #生成图片的数量
Start_theta = (-0,0) # 初始方向为0 # 0为水平向右
Start_theta = (20,120) # 【根据论文】初始角度方向为20~120
Start_theta_2 = (0,0)

# Start_position_x = (-350, -150) # 初始位置的横坐标的范围
# Start_position_y = (-100, -100) # 初始位置的纵坐标坐标为-100
Start_position_x = (0, 0) # 初始位置的横坐标的范围
Start_position_y = (0, 0) # 初始位置的纵坐标坐标为-100

Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1) #宽度的分支衰减率
Ratio_LW = (0.7,0.7) #宽度的分支衰减率【LZC:我的优化】

Dtheta = (20,120) # 每次分岔时，角度的变化范围
Width = (2,12) #initi 宽度的变化范围
# Width = (50,50) #【LZC:我的优化】
Width = (20,20) #【LZC:我的优化】
# Length_range = (90, 150) # init range 长度的变化范围
Length_range = (90, 90) #【LZC:我的优化】
# Length_range = (90, 150) # init range 长度的变化范围
for i  in range(Num_image): #生成Num_image张图片
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
    system.generate()
    turtle.speed(0)
    turtle.delay(0)
    turtle.tracer(False)
    turtle_screen = turtle.Screen()  # create graphics window
    turtle_screen.colormode(255)
    #ratip=1.36
    turtle.setup(800,800)
    turtle_screen.screensize(800, 800)
    turtle.bgcolor(0,0,0)
    system.draw(turtle)
    #file_name = "./fake_lrange_vessel/"+str(i)+'.png'
    # file_name = "./fake_very_smalltheta/"+str(i)+'2D.png'
    # tsimg = turtle.getscreen()
    turtle.reset()
    system.img.save("./fake_very_smalltheta/007-"+str(i)+'-3D.png')
    # exit(0)