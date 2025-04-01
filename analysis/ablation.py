info_config = {
    "indicatorFlag":"f1",
    "indicatorFlagMax":"f1",
    "indicatorFlag-old":[
        "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"
    ],
    "nrows":1000,
    "fileList":[
        ["FreeCOS-GuangYuan46","#4F81BD"],
        ["FreeCOS-GuangYuan58","#70AD47"],
        ["FreeCOS-GuangYuan59","m"],
        ["FreeCOS-GuangYuan60","b"],
        ["FreeCOS-GuangYuan57","#F79646"]
        
    ]
}
info =info_config["fileList"]

indicatorFlag=info_config["indicatorFlag"]#'f1'
nrows = info_config["nrows"]
# indicatorFlagList = ['Dice', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
# assert indicatorFlag in indicatorFlagList

import pandas as pd
import matplotlib.pyplot as plt
import re

# 定义一个函数，用于从字符串中提取数值部分并转换为浮点数
def extract_float_from_tensor_string2(tensor_string):
    if not isinstance(tensor_string, str):
        return tensor_string
    # 使用正则表达式提取数值部分
    match1 = re.search(r"tensor\(([^,]+)", tensor_string)
    match2 = re.search(r"tensor\(([-+]?\d*\.\d+|\d+)", tensor_string)
    # print("match1",match1)
    # print("match2",match2)
    if match1:
        # 提取匹配到的数值部分并转换为浮点数
        return float(match1.group(1))
    elif match2:
        # 提取匹配到的数值部分并转换为浮点数
        return float(match2.group(1))
    else:
        return float(tensor_string)#raise ValueError(f"无法从字符串中提取数值: {tensor_string}")

###############################################   333   ###############################################

config={}
def progressFile05(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv',nrows=nrows)
    # print("name:",name)

    indicatorFlagMax=info_config["indicatorFlagMax"]
    max_indicator_epoch = df.loc[df[indicatorFlagMax].idxmax(), 'epoch']
    # print(indicatorFlagMax+"最大时epoch为:",max_indicator_epoch)
    result = df[df['epoch'] == max_indicator_epoch]
    
    config0={}
    
    

    # for i in ['JC', 'f1',"Dice", 'recall', 'pr','Acc', 'Sp', 'AUC' ]:
    for i in ['JC', 'f1',"Dice", 'pr','recall', 'Acc', 'Sp', 'AUC' ]:
        if i=="Dice":
            if "Dice2" in result:
                value = result["Dice2"]
                config0["Dice"]=1-float(value)
            elif "Dice" in result:
                value = result["Dice"]
                config0["Dice"]=1-float(value)
            elif name=="FreeCOS-GuangYuan01": config0['Dice']= 0.6611683742394523 # 0.661
            else:
                config0["Dice"]=0
        else:
            value = result[i]
            if i in ['total_loss','Acc', 'Sp']:
                value = str(result[i])
                value = extract_float_from_tensor_string2(str(value))
            else:
                value = float(value)
            config0[i]=value

    for i in ['Acc', 'Sp', 'f1', 'pr', 'recall', 'JC']:
        config0[i]=config0[i]*0.01


    # print(config0,"\n\n\n")
    if True:
        name=name.split("FreeCOS-GuangYuan")[1]
        # name="test"+name
        if   name=="46":name="our"
        elif name=="58":name="our w/o Consis"
        elif name=="59":name="our w/o Syn-Bg"
        elif name=="57":name="our w/o 3D-Vessel"
        elif name=="60":name="our w/o adv"
    
    config0={#这里还决定了排列顺序
        'JC':config0['JC'], 
        'F1':config0['f1'],
        "Dice":config0['Dice'],
        'Pr':config0['pr'],
        'Sn':config0['recall'], 
        'AUC':config0['AUC'], 
        'Acc':config0['Acc'], 
        'Sp':config0['Sp'] ,
    }
    config[name]={
        "data":config0,
        "color":color
    }

def draw3(data):
    # 提取指标名称
    metrics = list(next(iter(data.values()))['data'].keys())

    # 提取每次测试的数据
    test_results = [list(test['data'].values()) for test in data.values()]
    test_colors = [test['color'] for test in data.values()]
    test_names = list(data.keys())

    # 设置图形大小
    plt.figure(figsize=(12, 6))

    # 设置柱子的宽度和位置
    bar_width = 0.23
    index = np.arange(len(metrics))

    # 绘制每个测试的柱状图
    for i, result in enumerate(test_results):
        plt.bar(index + (i+0.1) * bar_width, result, bar_width*0.9, color=test_colors[i], label=test_names[i])

    # 在每个柱形上方显示数值
    if True:
        # 设置字体样式
        font = {
            # 'family': 'serif',  # 字体族，如 'serif', 'sans-serif', 'cursive' 等
            # 'color': 'darkred',  # 字体颜色
            # 'weight': 'normal',  # 字体粗细，如 'normal', 'bold', 'light' 等
            'size': 5  # 字体大小
        }
        for i, result in enumerate(test_results):
            for j, value in enumerate(result):
                plt.text(index[j] + i * bar_width, value + 0.0, f'{value:.4f}', ha='center', va='bottom', fontdict=font)

    # 设置图形标签和标题
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Comparison of Metrics across Four Tests')
    plt.xticks(index + bar_width * (len(test_results) - 1) / 2, metrics)
    plt.ylim(0.49, max(max(test_results)) + 0.1)  # 设置纵坐标起点为0.7
    plt.legend()

    # 显示图形
    plt.show()
import numpy as np
if True:
    for info0 in info:
        progressFile05(info0[0],info0[1])
    # print(config)
    # 提取指标名称和测试名称

    draw3(config)


    