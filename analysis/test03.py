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


def progressFile(name,color):
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
    
    df[indicatorFlag] = df[indicatorFlag].apply(extract_float_from_tensor_string)

    # 绘制折线图
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
    plt.plot(df['epoch'], df[indicatorFlag], linestyle='-', color=color,label=name)  # 绘制折线图

    # # 隐藏纵坐标轴的刻度
    # plt.yticks([])

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
    # progressFile("FreeCOS-GuangYuan01",'r')
    # progressFile("FreeCOS-GuangYuan02",'g')
    # progressFile("FreeCOS-GuangYuan03",'b')
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
    progressFile("FreeCOS-GuangYuan17",'r')
    progressFile("FreeCOS-GuangYuan21",'g')
    progressFile("FreeCOS-GuangYuan22",'b')
    plt.title('y:'+indicatorFlag+' x:epoch')  # 设置标题
    plt.xlabel('Epoch')  # 设置横坐标标签
    plt.ylabel(indicatorFlag+' Score')  # 设置纵坐标标签
    # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # plt.grid(True)  # 添加网格线
    # plt.xticks(df['epoch'])  # 设置横坐标刻度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形

    