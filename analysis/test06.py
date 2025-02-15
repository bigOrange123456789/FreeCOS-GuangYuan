[
    ["FreeCOS-GuangYuan01","r"],
    ["FreeCOS-GuangYuan17","g"],
    ["FreeCOS-GuangYuan18","b"],
    ["FreeCOS-GuangYuan22","m"],
    ["FreeCOS-GuangYuan23","c"]
]
info=[
    ["FreeCOS-GuangYuan01","r"],
    ["FreeCOS-GuangYuan02","g"],
    ["FreeCOS-GuangYuan03","b"],
    ["FreeCOS-GuangYuan17","c"],
    ["FreeCOS-GuangYuan17","m"],
    ["FreeCOS-GuangYuan23","y"]
]
import json
with open('test.json', 'r', encoding='utf-8') as file:
    info = json.load(file)

'''
    progressFile("FreeCOS-GuangYuan01",'r')
    progressFile("FreeCOS-GuangYuan02",'g')
    progressFile("FreeCOS-GuangYuan03",'b')
    # progressFile("FreeCOS-GuangYuan04",'c')
    # progressFile("FreeCOS-GuangYuan05",'m')
    # progressFile("FreeCOS-GuangYuan06",'y') # 错误
    # progressFile("FreeCOS-GuangYuan07",'k')
    # progressFile("FreeCOS-GuangYuan09",'g')
    # progressFile("FreeCOS-GuangYuan10",'b')
    # progressFile("FreeCOS-GuangYuan11",'g')
    # progressFile("FreeCOS-GuangYuan01",'r')
    # progressFile("FreeCOS-GuangYuan21",'g')
    # progressFile("FreeCOS-GuangYuan22",'b')
    # progressFile("FreeCOS-GuangYuan17",'r')
    progressFile("FreeCOS-GuangYuan17",'c')
'''

indicatorFlag='total_loss'
indicatorFlagList = ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
assert indicatorFlag in indicatorFlagList
# pr : precision
# sp:  specificity
#   特异性是衡量模型在负样本（Negative Class）上表现的指标，它反映了模型正确识别负样本的能力。
'''
    precision = tp / (tp + fp + epsilon) # 预测为T的这些样本中，实际为T的比例 (预测T的正确率)        伪主的正确率:查准率precision
    recall = tp / (tp + fn + epsilon)    # 实际为T的这些样本中，预测为T的比例 (实际T的召回率)        主体的正确率:查全率recall
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    Acc = (tp+tn)/(tp+fn+tn+fp)         # accuracy：准确率                  (全预测正确率)         全部的正确率:准确率accuracy
    Sp = tn/(tn+fp+ epsilon)            # 特异性： 预测为F的这些样本中，实际为F的比例 (预测F的正确率) 背景的正确率:特异性specificity
    jc_score = jc(pred,gt) #Jaccard系数: 重叠度
    AUC 在曲线下方的面积。一般默认指的是ROC曲线。所以完整的缩写是ROC_AUC
        ROC曲线: 随着伪主区域的逐渐增加,
            TPR:在所有实际为阳性的样本中,被正确地判断为阳性之比率。
            FPR:在所有实际为阴性的样本中,被错误地判断为阳性之比率。
    AUC是阈值无关,而A、P、R三者都是阈值相关的。
'''

import pandas as pd
import matplotlib.pyplot as plt
import re

# 定义一个函数，用于从字符串中提取数值部分并转换为浮点数
def extract_float_from_tensor_string(tensor_string):
    if not isinstance(tensor_string, str):
        return tensor_string
    # 使用正则表达式提取数值部分
    match = re.search(r"tensor\(([^,]+)", tensor_string)
    if match:
        # 提取匹配到的数值部分并转换为浮点数
        return float(match.group(1))
    else:
        raise ValueError(f"无法从字符串中提取数值: {tensor_string}")
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
###############################################   111   ###############################################
def getMaxF1(df):
    # df = df.apply(extract_float_from_tensor_string2)
    max_indicator_epoch = df.loc[df["f1"].idxmax(), 'epoch']
    print("f1最大时epoch为:",max_indicator_epoch)
    result = df[df['epoch'] == max_indicator_epoch]
    # result = result.apply(extract_float_from_tensor_string2)
    # print(result,"\n")
    # total_loss=result['total_loss']
    # total_loss=extract_float_from_tensor_string2(str(total_loss))
    # print(total_loss)
    # 
    config={}
    for i in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']:
    #     # print(dir(result[i]))
        # print(i)
        # print(result[i])
        value = result[i]
        if i in ['total_loss','Acc', 'Sp']:
            value = str(result[i])
            value = extract_float_from_tensor_string2(str(value))
        else:
            value = float(value)
        config[i]=value
    if "Dice" in result:
        value = result["Dice"]
        config["Dice"]=float(value)

        # print(value,"\n\n\n")
    #     # j=float(result[i])
    #     print(i,type(result[i]),isinstance(value, str))
    #     if isinstance(value, str):
    #         match = re.search(r"tensor\(([^,]+)", value)
    #         value=float(match.group(1))
    #     print(float(value))
    # exit(0)
    print(config,"\n\n\n")
    # exit(0)


def progressFile03(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    print("name:",name)
    getMaxF1(df)
    
    '''# 打印数据框的列名
    print(df.columns)
    [5 rows x 9 columns]
    Index(['epoch', 'total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC'], dtype='object')
    '''
    
    df[indicatorFlag] = df[indicatorFlag].apply(extract_float_from_tensor_string2)

    # 绘制折线图
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
    plt.plot(df['epoch'], df[indicatorFlag], linestyle='-', color=color,label=name)  # 绘制折线图

    # # 隐藏纵坐标轴的刻度
    # plt.yticks([])

    if indicatorFlag=='total_loss':
        return

    # 绘制竖直直线
    max_indicator_epoch1 = df.loc[df[indicatorFlag].idxmax(), 'epoch']
    plt.axvline(x=max_indicator_epoch1, color=color, linestyle='--', alpha=0.6)

    # 绘制水平直线
    max_indicator_value1 = df[indicatorFlag].max()
    plt.axhline(y=max_indicator_value1, color=color, linestyle='--', alpha=0.6)

    # 计算最优结果之后的分数的均值和标准差
    max_indicator_epoch = df.loc[df[indicatorFlag].idxmax(), 'epoch']
    remaining_epochs = df[df['epoch'] > max_indicator_epoch]
    mean_indicator = remaining_epochs[indicatorFlag].mean()
    std_indicator = remaining_epochs[indicatorFlag].std()

    # 在图中添加均值和标准差的注释
    plt.text(max_indicator_epoch + 1, mean_indicator, 
            f'Max: {max_indicator_value1:.4f}\nMean: {mean_indicator:.4f}\nStd: {std_indicator:.4f}', 
            verticalalignment='center', horizontalalignment='left', 
            color=color, fontsize=10, bbox=dict(facecolor='white', alpha=1))

if True:
    for info0 in info:
        progressFile03(info0[0],info0[1])
    
    # progressFile("FreeCOS-GuangYuan17",'b')
    plt.title('y:'+indicatorFlag+' x:epoch')  # 设置标题
    plt.xlabel('Epoch')  # 设置横坐标标签
    plt.ylabel(indicatorFlag+' Score')  # 设置纵坐标标签
    # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # plt.grid(True)  # 添加网格线
    # plt.xticks(df['epoch'])  # 设置横坐标刻度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形
###############################################   222   ###############################################

# 创建2行4列的子图布局
fig, axs = plt.subplots(2, 4, figsize=(12, 6))  # figsize调整整体大小
axsList = [ axs[0,0],axs[0,1],axs[0,2],axs[0,3], axs[1,0],axs[1,1],axs[1,2],axs[1,3] ]

# pr : precision
# sp:  specificity
#   特异性是衡量模型在负样本（Negative Class）上表现的指标，它反映了模型正确识别负样本的能力。
# 定义一个函数，用于从字符串中提取数值部分并转换为浮点数
def progressFile04(id,name,color):
    indicatorFlag=indicatorFlagList[id]
    axs0=axsList[id]
    # assert indicatorFlag in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    
    '''# 打印数据框的列名
    print(df.columns)
    [5 rows x 9 columns]
    Index(['epoch', 'total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC'], dtype='object')
    '''
    
    df[indicatorFlag] = df[indicatorFlag].apply(extract_float_from_tensor_string2)

    # 绘制折线图
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
    axs0.plot(df['epoch'], df[indicatorFlag], linestyle='-', color=color,label=name)  # 绘制折线图

    # # 隐藏纵坐标轴的刻度
    # plt.yticks([])

    # 绘制竖直直线
    max_indicator_epoch1 = df.loc[df[indicatorFlag].idxmax(), 'epoch']
    axs0.axvline(x=max_indicator_epoch1, color=color, linestyle='--', alpha=0.6)

    # 绘制水平直线
    max_indicator_value1 = df[indicatorFlag].max()
    axs0.axhline(y=max_indicator_value1, color=color, linestyle='--', alpha=0.6)

    # 计算最优结果之后的分数的均值和标准差
    max_indicator_epoch = df.loc[df[indicatorFlag].idxmax(), 'epoch']
    remaining_epochs = df[df['epoch'] > max_indicator_epoch]
    mean_indicator = remaining_epochs[indicatorFlag].mean()
    std_indicator = remaining_epochs[indicatorFlag].std()

    # 在图中添加均值和标准差的注释
    axs0.text(max_indicator_epoch + 1, mean_indicator, 
            f'Max: {max_indicator_value1:.4f}\nMean: {mean_indicator:.4f}\nStd: {std_indicator:.4f}', 
            verticalalignment='center', horizontalalignment='left', 
            color=color, fontsize=10, bbox=dict(facecolor='white', alpha=1))
    
    axs0.legend()
if True:
    for id in range(len(axsList)):
        indicatorFlag=indicatorFlagList[id]
        axs0=axsList[id]
        for info0 in info:
            progressFile04(id,info0[0],info0[1])

        axs0.set_title('y:'+indicatorFlag+' x:epoch')

    # plt.title('y:indicator x:epoch')  # 设置标题    
    # plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形

###############################################   333   ###############################################

config={}
def progressFile05(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    print("name:",name)

    max_indicator_epoch = df.loc[df["f1"].idxmax(), 'epoch']
    print("f1最大时epoch为:",max_indicator_epoch)
    result = df[df['epoch'] == max_indicator_epoch]
    
    config0={}
    # if name=="FreeCOS-GuangYuan26" or name=="FreeCOS-GuangYuan27":
    #     value = result["Dice"]
    #     config0["Dice"]=1-float(value)
    # elif "Dice2" in result:
    #     value = result["Dice2"]
    #     config0["Dice"]=1-float(value)
    
    if "Dice2" in result:
        value = result["Dice2"]
        config0["Dice"]=1-float(value)
    elif "Dice" in result:
        value = result["Dice"]
        config0["Dice"]=1-float(value)
    elif name=="FreeCOS-GuangYuan01": config0['Dice']= 0.6611683742394523 # 0.661
    else:
        config0["Dice"]=0
    # for i in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']:
    # for i in ['total_loss', 'Acc', 'Sp', 'AUC', 'f1', 'pr', 'recall', 'JC']:
    for i in ['Acc', 'Sp', 'AUC', 'f1', 'pr', 'recall', 'JC']:
    #     # print(dir(result[i]))
        # print(i)
        # print(result[i])
        value = result[i]
        if i in ['total_loss','Acc', 'Sp']:
            value = str(result[i])
            value = extract_float_from_tensor_string2(str(value))
        else:
            value = float(value)
        config0[i]=value
    if "AUC2" in result:
        value = result["AUC2"]
        config0["AUC"]=float(value)

    print(config0,"\n\n\n")
    if True:
        name=name.split("FreeCOS-GuangYuan")[1]
        name="Test"+name
    
    config[name]={
        "data":config0,
        "color":color
    }
    return config0


if True:
    for info0 in info:
        progressFile05(info0[0],info0[1])

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
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
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

    