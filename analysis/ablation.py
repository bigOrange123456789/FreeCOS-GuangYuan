info_config = {
    "indicatorFlag":"f1",
    "indicatorFlagMax":"f1",
    "indicatorFlag-old":[
        "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"
    ],
    "nrows":294,
    "fileList":[
        ["FreeCOS-GuangYuan46","#4F81BD"],
        ["FreeCOS-GuangYuan58","#70AD47"],
        ["FreeCOS-GuangYuan59","m"],
        ["FreeCOS-GuangYuan57","#F79646"]
    ]
}
info =info_config["fileList"]

indicatorFlag=info_config["indicatorFlag"]#'f1'
nrows = info_config["nrows"]
indicatorFlagList = ['Dice', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
assert indicatorFlag in indicatorFlagList

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
    for i in ['JC', 'f1',"Dice", 'recall', 'pr','Acc', 'Sp' ]:
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
    
    config0={
        'JC':config0['JC'], 
        'F1':config0['f1'],
        "Dice":config0['Dice'],
        'Sn':config0['recall'], 
        'Pr':config0['pr'],
        'Acc':config0['Acc'], 
        'Sp':config0['Sp'] 
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
    bar_width = 0.22
    index = np.arange(len(metrics))

    # 绘制每个测试的柱状图
    for i, result in enumerate(test_results):
        plt.bar(index + (i+0.1) * bar_width, result, bar_width*0.9, color=test_colors[i], label=test_names[i])

    # 在每个柱形上方显示数值
    if True:
        for i, result in enumerate(test_results):
            for j, value in enumerate(result):
                plt.text(index[j] + i * bar_width, value + 0.0, f'{value:.3f}', ha='center', va='bottom')

    # 设置图形标签和标题
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Comparison of Metrics across Four Tests')
    plt.xticks(index + bar_width * (len(test_results) - 1) / 2, metrics)
    plt.ylim(0.45, max(max(test_results)) + 0.1)  # 设置纵坐标起点为0.7
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
    exit(0)

    # 获取所有指标名称
    metrics = list(next(iter(config.values()))['data'].keys())

    # 创建 2×4 的子图布局
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))  # 调整整体布局大小

    # 遍历每个指标并绘制柱形图
    for i, metric in enumerate(metrics):
        row = i // 4  # 计算行索引
        col = i % 4   # 计算列索引
        ax = axs[row, col]  # 获取当前子图的轴对象

        bars = []
        labels = []
        colors = []

        for experiment, values in config.items():
            bars.append(values['data'][metric])
            labels.append(experiment)
            colors.append(values['color'])

        # ax.bar(labels, bars, color=colors)
        # 绘制柱形图
        bar_container = ax.bar(labels, bars, color=colors)
        # 在每个柱形上方显示数值
        for bar in bar_container:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(metric)
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(bars) * 1.1)  # 设置 y 轴范围，确保柱子不会被截断
        ax.tick_params(axis='x', rotation=45)  # 旋转 x 轴标签，避免重叠

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示整个图形
    plt.show()

    