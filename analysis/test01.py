import pandas as pd
path="../logs/FreeCOS-GuangYuan01/"
# 读取CSV文件
df = pd.read_csv(path+'val_train_f1.csv')

# 打印整个数据框
print(df)

# 打印数据框的前几行
print(df.head())

# 打印数据框的列名
print(df.columns)
'''
[5 rows x 9 columns]
Index(['epoch', 'total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC'], dtype='object')
'''

# 访问特定列
print(df['f1'])  # 替换为实际的列名
# F1分数是精确率和召回率的调和平均数

f1=df['f1']
print(type(f1))

import matplotlib.pyplot as plt
# 绘制折线图
# plt.figure(figsize=(10, 6))  # 设置图形大小
# plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
plt.plot(df['epoch'], df['f1'], linestyle='-', color='b')  # 绘制折线图
plt.title('y:F1 x:epoch')  # 设置标题
plt.xlabel('Epoch')  # 设置横坐标标签
plt.ylabel('F1指标')  # 设置纵坐标标签
# plt.grid(True)  # 添加网格线
plt.xticks(df['epoch'])  # 设置横坐标刻度
plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形