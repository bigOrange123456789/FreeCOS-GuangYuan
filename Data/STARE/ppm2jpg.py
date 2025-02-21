from PIL import Image
# import numpy as np
import os

# 示例：读取PPM文件并保存为PNG
ppm_file_path = 'img/im0001.ppm'  # 替换为你的PPM文件路径
png_file_path = 'im0001.ppm'  # 输出PNG文件路径

def convert_ppm_to_jpg(ppm_path, jpg_path):
    """
    将PPM格式的图片转换为JPG格式。
    :param ppm_path: PPM文件路径
    :param jpg_path: 输出JPG文件路径
    """
    # 打开PPM文件
    with Image.open(ppm_path) as img:
        # 将图片保存为JPG格式
        img.save(jpg_path, format='JPEG')
        # print(f"图片已成功转换为JPG格式并保存为：{jpg_path}")

# 示例：将PPM图片转换为JPG
# ppm_file_path = 'img_ppm/im0001.ppm'  # 替换为你的PPM文件路径
# jpg_file_path = 'im0001.jpg'   # 输出JPG文件路径

# 调用函数进行转换
# convert_ppm_to_jpg(ppm_file_path, jpg_file_path)

# 假设 ppm_images 是包含 PPM 文件路径的列表
ppm_images = os.listdir("img_ppm")#["image1.ppm", "image2.ppm", "image3.ppm"]
# print("ppm_images",ppm_images)
for i in range(len(ppm_images)):
    name = ppm_images[i]
    print(i,"/",len(ppm_images))
    convert_ppm_to_jpg("img_ppm/"+name, "img_jpg/"+name+".jpg")
    ppm_path="img_ppm/"+name
    jpg_path1="img_jpg/"+name+".jpg"
    jpg_path2="img_gray/"+name+".jpg"
    with Image.open(ppm_path) as img:
        # 将图片保存为JPG格式
        img.save(jpg_path1, format='JPEG')
        # 转换为灰度图
        gray_img = img.convert("L")
        gray_img.save(jpg_path2, format='JPEG')
# for ppm_path in ppm_images:
# Gray