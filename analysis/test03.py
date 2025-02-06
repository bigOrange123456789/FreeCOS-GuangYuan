import pandas as pd
import matplotlib.pyplot as plt
import re

indicatorFlag='f1'
assert indicatorFlag in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
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
def progressFile(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    
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
    progressFile("FreeCOS-GuangYuan01",'r')
    # progressFile("FreeCOS-GuangYuan02",'g')
    # progressFile("FreeCOS-GuangYuan03",'b')
    # progressFile("FreeCOS-GuangYuan04",'c')
    # progressFile("FreeCOS-GuangYuan05",'m')
    # progressFile("FreeCOS-GuangYuan06",'y') # 错误
    # progressFile("FreeCOS-GuangYuan07",'k')
    # progressFile("FreeCOS-GuangYuan09",'g')
    # progressFile("FreeCOS-GuangYuan10",'b')
    progressFile("FreeCOS-GuangYuan14",'g')
    progressFile("FreeCOS-GuangYuan15",'b')
    plt.title('y:'+indicatorFlag+' x:epoch')  # 设置标题
    plt.xlabel('Epoch')  # 设置横坐标标签
    plt.ylabel(indicatorFlag+' Score')  # 设置纵坐标标签
    # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # plt.grid(True)  # 添加网格线
    # plt.xticks(df['epoch'])  # 设置横坐标刻度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形

    