import os

def rename_files(source_folder, target_folder):
    """
    使用source_folder中的文件名来重命名target_folder中的文件。
    假设两个文件夹中的文件数量相同。
    
    :param source_folder: 包含原始文件名的文件夹路径
    :param target_folder: 需要重命名文件的文件夹路径
    """
    # 获取两个文件夹中的文件列表
    source_files = sorted(os.listdir(source_folder), reverse=True)
    target_files = sorted(os.listdir(target_folder), reverse=True)
    
    # 检查文件数量是否相同
    if len(source_files) != len(target_files):
        print("两个文件夹中的文件数量不匹配！")
        return
    
    # 遍历文件并重命名
    for source_name, target_name in zip(source_files, target_files):
        if not source_name == target_name:
            # 构造完整的文件路径
            # source_path = os.path.join(source_folder, source_name)
            target_path = os.path.join(target_folder, target_name)
            new_target_path = os.path.join(target_folder, source_name)
            
            # 重命名文件
            os.rename(target_path, new_target_path)
            print(f"重命名文件：{target_path} -> {new_target_path}")


source_folder = "./img"  # 替换为源文件夹路径
target_folder = "./bg_lzc"  # 替换为目标文件夹路径

rename_files(source_folder, target_folder)