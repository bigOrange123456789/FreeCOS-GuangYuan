import json
with open('test_config.json', 'r', encoding='utf-8') as file:
    info_config = json.load(file)
    info =info_config["fileList"]
nrows = info_config["nrows"]
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
# 创建2行4列的子图布局
N1=3
N2=5
fig, axs = plt.subplots(N1, N2, figsize=(12, 6))  # figsize调整整体大小
indicatorFlagList = ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
['epoch', 'loss_D_tar', 'loss_D_src', 'loss_adv', 'loss_ce', 'loss_dice',
       'loss_seg_w', 'loss_adv_w', 'loss_contrast_w', 'loss_pseudo_w',
       'loss_conn_w', 'loss_adv.1', 'loss_contrast', 'loss_pseudo',
       'loss_conn']
indicatorFlagList = [
       'loss_D_tar', 'loss_D_src', 'loss_adv', 'loss_ce', 'loss_dice',
       'loss_seg_w', 'loss_adv_w', 'loss_contrast_w', 'loss_pseudo_w','loss_conn_w', "loss_cons_w",
       'loss_contrast', 'loss_pseudo', 'loss_conn',"loss_cons"] #'loss_adv.1', 'loss_contrast', 'loss_pseudo', 'loss_conn']
# axsList = [ 
#     axs[0,0],axs[0,1],axs[0,2],axs[0,3],axs[0,4],  
#     axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[1,4], 
#     axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[1,4], 
#     ]
axsList = []
for i in range(N1):
    for j in range(N2):
        axsList.append(axs[i,j])
# indicatorFlagList = [ 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
# axsList = [ axs[0,0],axs[0,1],axs[0,2],axs[0,3], axs[1,0],axs[1,1],axs[1,2] ]

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
def progressFile(id,name,color):
    indicatorFlag=indicatorFlagList[id]
    axs0=axsList[id]
    # assert indicatorFlag in ['total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC']
    path="../logs/"+name
    # 读取CSV文件
    if not os.path.exists(path+'/train_loss.csv'):
        print("文件不存在:",path+'/train_loss.csv')
        return
    # 读取CSV文件
    df = pd.read_csv(path+'/train_loss.csv',nrows=nrows)
    # print(df[indicatorFlag])
    '''# 打印数据框的列名
    print(df.columns)
    [5 rows x 9 columns]
    Index(['epoch', 'total_loss', 'f1', 'AUC', 'pr', 'recall', 'Acc', 'Sp', 'JC'], dtype='object')
    '''
    
    if indicatorFlag in df:
        df[indicatorFlag] = df[indicatorFlag].apply(extract_float_from_tensor_string)
    else:
        z=pd.Series(0, index=df["epoch"].index)
        df[indicatorFlag] = z

    # 绘制折线图
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.plot(df['epoch'], df['f1'], marker='o', linestyle='-', color='b')  # 绘制折线图
    axs0.plot(df['epoch'], df[indicatorFlag], linestyle='-', color=color,label=name)  # 绘制折线图

    # # 隐藏纵坐标轴的刻度
    # plt.yticks([])

    '''
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
    '''

    axs0.legend()




# if True:
for id in range(len(indicatorFlagList)):
    indicatorFlag=indicatorFlagList[id]
    print("indicatorFlag:",indicatorFlag)
    axs0=axsList[id]
    for info0 in info:
        progressFile(id,info0[0],info0[1])
    # progressFile(id,"FreeCOS-GuangYuan01",'r')
    # progressFile(id,"FreeCOS-GuangYuan02",'g')
    # progressFile(id,"FreeCOS-GuangYuan03",'b')
    # progressFile(id,"FreeCOS-GuangYuan04",'c')
    # progressFile(id,"FreeCOS-GuangYuan05",'m')
    # progressFile(id,"FreeCOS-GuangYuan06",'y') # 错误
    # progressFile(id,"FreeCOS-GuangYuan07",'k')
    # progressFile(id,"FreeCOS-GuangYuan09",'g')
    # progressFile(id,"FreeCOS-GuangYuan10",'b')
    # progressFile(id,"FreeCOS-GuangYuan11",'g')
    # progressFile(id,"FreeCOS-GuangYuan17",'c')
    # progressFile(id,"FreeCOS-GuangYuan21",'g')
    # progressFile(id,"FreeCOS-GuangYuan22",'b')
    # progressFile("FreeCOS-GuangYuan17",'c')
    axs0.set_title('y:'+indicatorFlag+' x:epoch')
    '''
    plt.xlabel('Epoch')  # 设置横坐标标签
    plt.ylabel(indicatorFlag+' Score')  # 设置纵坐标标签
    # plt.legend(fontsize=10, title='Curve Names')  # 添加图例，并设置字体大小和标题
    plt.legend(fontsize=10)  # 添加图例，并设置字体大小和标题
    # plt.grid(True)  # 添加网格线
    # plt.xticks(df['epoch'])  # 设置横坐标刻度
    '''
# plt.title('y:indicator x:epoch')  # 设置标题    
# plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形

    