import pandas as pd
import matplotlib.pyplot as plt


def progressFile(name,color):
    path="../logs/"+name
    # 读取CSV文件
    df = pd.read_csv(path+'/val_train_f1.csv')
    
    '''# 打印数据框的列名
    print(df.columns)
    [5 rows x 9 columns]
    Index(['epoch', 'total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC'], dtype='object')
    '''
    
    # 绘制折线图
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
    plt.plot(df['epoch'], df['f1'], linestyle='-', color=color,label=name)  # 绘制折线图

    # 绘制竖直直线
    max_f1_epoch1 = df.loc[df['f1'].idxmax(), 'epoch']
    plt.axvline(x=max_f1_epoch1, color=color, linestyle='--', alpha=0.6)

    # 绘制水平直线
    max_f1_value1 = df['f1'].max()
    plt.axhline(y=max_f1_value1, color=color, linestyle='--', alpha=0.6)

if True:
    progressFile("FreeCOS-GuangYuan01",'r')
    # progressFile("FreeCOS-GuangYuan02",'g')
    # progressFile("FreeCOS-GuangYuan03",'b')
    # progressFile("FreeCOS-GuangYuan04",'c')
    # progressFile("FreeCOS-GuangYuan05",'m')
    # progressFile("FreeCOS-GuangYuan06",'y') # 错误
    # progressFile("FreeCOS-GuangYuan07",'y')
    progressFile("FreeCOS-GuangYuan08",'y')
    plt.title('y:F1 x:epoch')  # 设置标题
    plt.xlabel('Epoch')  # 设置横坐标标签
    plt.ylabel('F1 Score')  # 设置纵坐标标签
    # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # plt.grid(True)  # 添加网格线
    # plt.xticks(df['epoch'])  # 设置横坐标刻度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形

    