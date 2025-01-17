#from turtle import *
import turtle
import numpy as np
import random
import math
from PIL import Image
#tempNumpyArray=np.load("DRIVE_instensity.npy")
#score = tempNumpyArray.tolist()
from PIL import EpsImagePlugin

EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.04.0\bin\gswin64c'

def draw_background(a_turtle):
    """ Draw a background rectangle. """
    ts = a_turtle.getscreen()
    canvas = ts.getcanvas()
    height = ts.getcanvas()._canvas.winfo_height()
    width = ts.getcanvas()._canvas.winfo_width()

    turtleheading = a_turtle.heading()
    turtlespeed = a_turtle.speed()
    penposn = a_turtle.position()
    penstate = a_turtle.pen()

    a_turtle.penup()
    a_turtle.speed(0)  # fastest
    a_turtle.goto(-width/2-2, -height/2+3)
    print("turtle.Screen().bgcolor()",turtle.Screen().bgcolor())
    a_turtle.fillcolor(255,255,255)
    a_turtle.begin_fill()
    a_turtle.setheading(0)
    a_turtle.forward(width)
    a_turtle.setheading(90)
    a_turtle.forward(height)
    a_turtle.setheading(180)
    a_turtle.forward(width)
    a_turtle.setheading(270)
    a_turtle.forward(height)
    a_turtle.end_fill()

    a_turtle.penup()
    a_turtle.setposition(*penposn)
    a_turtle.pen(penstate)
    a_turtle.setheading(turtleheading)
    a_turtle.speed(turtlespeed)

def makecolor(turtle,new_length,width,x,y,theta):
    #Tile one pixel or two pixel
    turtle_screen.colormode(255) #背景设置为白色
    # system.draw(turtle)
    #Need more set of different colorset#Get random colorset
    random_R = np.random.randint(0, 255) #随机取一个颜色
    random_G = np.random.randint(0, 255)
    random_B = np.random.randint(0, 255)
    R_insity = random_R
    G_insity = random_G
    B_insity = random_B

    Ar1 = np.random.uniform(-3,3) # 正负三个颜色单位的偏移
    Ag1 = np.random.uniform(-3,3)
    Ab1 = np.random.uniform(-3,3)
    turtle.pu() #抬起画笔，这样移动的时候不会留下痕迹 # turtle.pu()和turtle.pd()都有它们的别名turtle.up()和turtle.down()，它们的功能是完全相同的。
    turtle.goto((x, y))
    turtle.setheading(theta)
    width = width
    if width<2:#宽度不能低于两像素
        width = 2
    turtle.pensize(width)
    new_instenisty_r = int(np.clip(R_insity + Ar1,0,255)) # rand(0, 255)+/-3
    new_instenisty_g = int(np.clip(G_insity + Ag1,0,255))
    new_instenisty_b = int(np.clip(B_insity + Ab1,0,255))
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    turtle.pd() #放下画笔
    # print('1a:', x, y, theta,width,new_instenisty_r, new_instenisty_g, new_instenisty_b,new_length)
    turtle.forward(new_length) #向前移动

    # 后面是绘制了第2遍，是冗余的代码
    # turtle.pu() #抬起画笔
    # turtle.goto((x, y)) #设置位置
    # turtle.setheading(theta) #设置方向
    # turtle.pd()
    # turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    # turtle.forward(new_length)
    # print('2b:', x, y, theta, width, new_instenisty_r, new_instenisty_g, new_instenisty_b, new_length)

class LSystem_vessel():
    def __init__(self, axiom, rules, rules_2, rules_3, theta=0, width=5, dtheta_1=40, dtheta_2=30, start=(-350,0), length=80,iteration=3,width_1=0.79, width_2 = 0.5):
        self.sentence = axiom   # 基本规则 # axiom的本义是原理、公理
        self.rules = rules      # 规则1
        self.relus_2 = rules_2  # 规则2
        self.rules_3 = rules_3  # 规则3
        self.iteration = iteration # 迭代次数
        self.width = width  # 初始宽度？
        self.theta = theta  # 初始角度？
        self.dtheta_1 = dtheta_1 # 偏转角度？
        self.dtheta_2 = dtheta_2 # 偏转角度？
        self.length = length #长度
        self.positions = []
        self.start = start
        self.lamda_1 = width_1
        self.lamda_2 = width_2

        self.x, self.y = start # 初始位置

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
                    new_length = self.length*self.lamda_1
                else:
                    new_length = self.length*self.lamda_2
                turtle.pensize(self.width) # 设定宽度
                x, y = turtle.position() #起点
                theta = self.theta
                width = self.width
                makecolor(turtle, new_length, width, x, y, theta) # 长度、宽度、起点、方向
                self.x, self.y = turtle.position() #更新位置
            elif char == '+': #调整行进方向
                dtheta = np.random.randint(low=1, high=5) #
                self.theta += dtheta
                self.width = self.width * self.lamda_1
                turtle.right(self.theta)
            elif char == '-': #反方向调整行进方向
                self.width = self.width * self.lamda_2
                dtheta = np.random.randint(low=1, high=5)  #
                self.theta -= dtheta
                turtle.left(self.theta)
            elif char == '[': #记录迭代位置 #使用数据栈结构
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length,"new_width_0": self.width * 0.79})
                flag = True
            elif char == ']': #返回迭代位置
                turtle.pu()
                position = self.positions.pop()
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
                turtle.goto((self.x, self.y))
                turtle.setheading(self.theta)

# 所谓的规则语句，就是将字符F替换为由F组成的字符串
rules = {"F":"F-F[+F-F][-F+F]"} # 这应该是类似json格式的对象
rules_2 = {"F":"F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"} # 规则是通过字符串来定义的
rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}

path = "[+F-F][-F]" #path变量应该是目前使用的规则
print('path:',path)

Num_image = 25 #生成图片的数量
Start_theta = (-0,0) # 初始方向为0
Start_theta_2 = (0,0)

Start_position_x = (-350, -150) # 初始位置的横坐标的范围
Start_position_y = (-100, -100) # 初始位置的纵坐标坐标为-100
Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1)

Dtheta = (20,120) # 每次分岔时，角度的变化范围
Width = (2,12) #initi 宽度的变化范围
Length_range = (90, 150) # init range 长度的变化范围

for i  in range(Num_image): #生成Num_image张图片
    p2 = random.random()
    if p2>0.5:
        path = "[+F-F][-F]"
    else:
        path = "F"
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

    Ratio_lw_1 = np.random.uniform(Ratio_LW[0],Ratio_LW[1])
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
    draw_background(turtle)
    system.draw(turtle)
    #file_name = "./fake_lrange_vessel/"+str(i)+'.png'
    file_name = "./fake_very_smalltheta/"+str(i)+'.png'
    tsimg = turtle.getscreen()
    tsimg.getcanvas().postscript(file="work_vessel.eps")
    im = Image.open("work_vessel.eps")
    out = im.resize((512,512))
    #out = im
    im_array = np.array(out)
    print("im_array",im_array.shape)
    out.save(file_name)
    turtle.reset()