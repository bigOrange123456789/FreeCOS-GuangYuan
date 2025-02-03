import os

def save_filenames_to_txt(folder_path, output_file_path):
    # 获取指定文件夹中的所有文件名
    filenames = os.listdir(folder_path)
    
    # 打开输出文件，准备写入
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # 遍历文件名列表并写入到输出文件中
        for filename in filenames:
            output_file.write(filename + '\n')

# 指定文件夹路径
folder_path = 'fake_very_smalltheta'
# 指定输出文件路径
output_file_path = 'list.txt'

# 调用函数
save_filenames_to_txt(folder_path, output_file_path)