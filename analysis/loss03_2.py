import json
with open('test_config.json', 'r', encoding='utf-8') as file:
    info = json.load(file)["fileList"]

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
# 创建2行4列的子图布局
N1=3
N2=5
# fig, axs = plt.subplots(N1, N2, figsize=(12, 6))  # figsize调整整体大小
indicatorFlagList = ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
['epoch', 'loss_D_tar', 'loss_D_src', 'loss_adv', 'loss_ce', 'loss_dice',
       'loss_seg_w', 'loss_adv_w', 'loss_contrast_w', 'loss_pseudo_w',
       'loss_conn_w', 'loss_adv.1', 'loss_contrast', 'loss_pseudo',
       'loss_conn']
indicatorFlagList = [
       'loss_D_tar', 'loss_D_src', 'loss_adv', 'loss_ce', 'loss_dice',
       'loss_seg_w', 'loss_adv_w', 'loss_contrast_w', 'loss_pseudo_w','loss_conn_w', 
       'loss_contrast', 'loss_pseudo', 'loss_conn'] #'loss_adv.1', 'loss_contrast', 'loss_pseudo', 'loss_conn']

config0={
    'loss_seg_w':"r", 
    'loss_adv_w':"g", 
    'loss_contrast_w':"b", 
    'loss_pseudo_w':"m",
    'loss_conn_w':"c" 
}
# axsList = [ 
#     axs[0,0],axs[0,1],axs[0,2],axs[0,3],axs[0,4],  
#     axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[1,4], 
#     axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[1,4], 
#     ]
axsList = []
for i in range(N1):
    for j in range(N2):
        axsList.append(plt)#axsList.append(axs[i,j])
# indicatorFlagList = [ 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
# axsList = [ axs[0,0],axs[0,1],axs[0,2],axs[0,3], axs[1,0],axs[1,1],axs[1,2] ]

# pr : precision
# sp:  specificity
#   特异性是衡量模型在负样本（Negative Class）上表现的指标，它反映了模型正确识别负样本的能力。

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



fileName="FreeCOS-GuangYuan32"
path="../logs/"+ fileName
# 读取CSV文件
if not os.path.exists(path+'/train_loss.csv'):
    print("文件不存在:",path+'/train_loss.csv')
    exit(0)
# 读取CSV文件
df = pd.read_csv(path+'/train_loss.csv')
    
for indicatorFlag in config0:
    color = config0[indicatorFlag]
    df[indicatorFlag] = df[indicatorFlag].apply(extract_float_from_tensor_string)
    plt.plot(df['epoch'], df[indicatorFlag], linestyle='-', color=color,label=indicatorFlag)  # 绘制折线图
    plt.legend()


df_all=0
for indicatorFlag in config0:
    df_all = df_all + df[indicatorFlag]
plt.plot(df['epoch'], df_all, linestyle='-', color="y",label="loss_total")  # 绘制折线图
plt.legend()

# plt.title('y:indicator x:epoch')  # 设置标题    
# plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形

    