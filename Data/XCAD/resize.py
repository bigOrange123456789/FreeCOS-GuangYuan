
from PIL import Image
import os

def resize(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)# 创建输出文件夹（如果不存在）
    
    for filename in os.listdir(folder_path):# 遍历文件夹中的所有图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): # if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img_resized = img.resize((700, 605))  # 调整分辨率
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

    print("所有图片已调整完成！")

if __name__ == '__main__':
    folder_path   = "../../../DataSet-images/STARE_sim/train/fake_grayvessel_bend_old"
    output_folder = "../../../DataSet-images/STARE_sim/train/fake_grayvessel_bend"  # 如果没有，可以创建一个
    resize(folder_path, output_folder)

    folder_path   = "../../../DataSet-images/STARE_sim/train/fake_gtvessel_bend_old"
    output_folder = "../../../DataSet-images/STARE_sim/train/fake_gtvessel_bend"  # 如果没有，可以创建一个
    resize(folder_path, output_folder)
