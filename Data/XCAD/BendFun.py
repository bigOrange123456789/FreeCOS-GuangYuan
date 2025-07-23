import numpy as np
import random

class BendFun():
    def __init__(self):
        self.c={}

        def bend_error(k0): #f(0)=f(1)=0
            return (2*k0-1)**2

        def bend(k0,c): #f(0)=f(1)=0 ,最好取值范围是[-1,1]
            return 0 #无任何弯曲
        
        self.c["exponent"]=random.randint(1, 3)            
        
        # self.c["x3"]=10*(random.random()-0.5)
        num=random.randint(0, 2)
        self.c["x_list"]=[]
        for i in range(num):
            self.c["x_list"].append(
                1.5*(random.random()-0.5)
            )
        # self.c["x_list"]=[ 0.5 ]
        def bend0(x,c): #f(0)=f(1)=0 ,最好取值范围是[-1,1]
            # f = k0*(k0-1)*(k0-c["x3"])
            y = 2 * x*(x-1)
            for x0 in c["x_list"]:
                y = 2 * y*(x-x0)
            y = y ** c["exponent"]
            return y    
        
        self.c["sin_cycle"]=random.randint(1, 3) #正弦周期的数量
        def bend1(x,c):
            y = np.sin(c["sin_cycle"]*np.pi*x)
            y = y ** c["exponent"]
            return y
        
        self.bendList=[bend0,bend1]
        # self.bendList=[bend1]
        self.bendListIndex=random.randint(0, len(self.bendList)-1)
        
        # bend=bendList[0]
    def get(self,k):
        f=self.bendList[self.bendListIndex]
        return f(k,self.c)
class BendFunCathe():
    def __init__(self):
        self.c={}

        
        self.c["exponent"]=random.randint(1, 2)            
        
        # self.c["x3"]=10*(random.random()-0.5)
        num=random.randint(0, 2)
        self.c["x_list"]=[]
        for i in range(num):
            self.c["x_list"].append(
                1.5*(random.random()-0.5)
            )
        # self.c["x_list"]=[ 0.5 ]
        def bend0(x,c): #f(0)=f(1)=0 ,最好取值范围是[-1,1]
            # f = k0*(k0-1)*(k0-c["x3"])
            y = 2 * x*(x-1)
            for x0 in c["x_list"]:
                y = 2 * y*(x-x0)
            y = y ** c["exponent"]
            return y    
        
        self.bendList=[bend0]
        # self.bendList=[bend1]
        self.bendListIndex=random.randint(0, len(self.bendList)-1)
        
        # bend=bendList[0]
    def get(self,k):
        f=self.bendList[self.bendListIndex]
        return f(k,self.c)
import matplotlib.pyplot as plt
class Test():
    def __init__(self):
        bend = BendFun()
        # 定义x的范围（从0到2π，共100个点）
        x = np.linspace(0, 1, 100)
        
        # 计算y值（余弦函数）
        y = np.cos(x)
        y = bend.get(x)
        
        # 创建图形
        plt.figure(figsize=(8, 4))  # 设置图形大小
        plt.plot(x, y, label="cos(x)", color="blue", linewidth=2)  # 绘制余弦曲线
        
        # 添加标题和标签
        plt.title("Cosine Curve", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("cos(x)", fontsize=12)
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 显示图形
        plt.show()

if __name__ == '__main__':
    Test()

