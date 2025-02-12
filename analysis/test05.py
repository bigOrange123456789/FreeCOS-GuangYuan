import pandas as pd
import matplotlib.pyplot as plt
import re

indicatorFlag='f1'
assert indicatorFlag in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
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
'''
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

config={}
def progressFile(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    print("name:",name)

    max_indicator_epoch = df.loc[df["f1"].idxmax(), 'epoch']
    print("f1最大时epoch为:",max_indicator_epoch)
    result = df[df['epoch'] == max_indicator_epoch]
    
    config0={}
    # for i in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']:
    for i in ['total_loss', 'Acc', 'Sp', 'AUC', 'f1', 'pr', 'recall', 'JC']:
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
    
    # progressFile("FreeCOS-GuangYuan17",'c')
    # progressFile("FreeCOS-GuangYuan21",'m')
    progressFile("FreeCOS-GuangYuan22",'y')

    # print(config)
    # plt.title('y:'+indicatorFlag+' x:epoch')  # 设置标题
    # plt.xlabel('Epoch')  # 设置横坐标标签
    # plt.ylabel(indicatorFlag+' Score')  # 设置纵坐标标签
    # # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    # plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # # plt.grid(True)  # 添加网格线
    # # plt.xticks(df['epoch'])  # 设置横坐标刻度
    # plt.tight_layout()  # 自动调整布局
    # plt.show()  # 显示图形
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

        ax.bar(labels, bars, color=colors)
        ax.set_title(metric)
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(bars) * 1.1)  # 设置 y 轴范围，确保柱子不会被截断
        ax.tick_params(axis='x', rotation=45)  # 旋转 x 轴标签，避免重叠

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示整个图形
    plt.show()

    