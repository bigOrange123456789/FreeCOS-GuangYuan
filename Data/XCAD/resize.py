
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


            # 创建一个新的空白图片，分辨率为 200x100，背景为白色
            new_img = Image.new(mode="RGB", size=(200, 100), color="white")    
            paste_x = (new_img.width - resized_img.width) // 2 # 计算填充后的图片应该放置的位置（居中）
            paste_y = (new_img.height - resized_img.height) // 2

            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

    print("所有图片已调整完成！")


from PIL import Image

# 输入图片路径
input_image_path = "path/to/your/image.png"
# 输出图片路径
output_image_path = "path/to/output/image.png"

# 打开图片
with Image.open(input_image_path) as img:
    # 调整图片分辨率到 100x50
    resized_img = img.resize((100, 50))

    # 创建一个新的空白图片，分辨率为 200x100，背景为白色
    new_img = Image.new(mode="RGB", size=(200, 100), color="white")

    # 计算填充后的图片应该放置的位置（居中）
    paste_x = (new_img.width - resized_img.width) // 2
    paste_y = (new_img.height - resized_img.height) // 2

    # 将调整后的图片粘贴到新图片的指定位置
    new_img.paste(resized_img, (paste_x, paste_y))

    # 保存最终结果
    new_img.save(output_image_path)

print(f"处理完成，结果保存到：{output_image_path}")



if __name__ == '__main__':
    folder_path   = "../../../DataSet-images/STARE_sim/train/fake_grayvessel_bend_old"
    output_folder = "../../../DataSet-images/STARE_sim/train/fake_grayvessel_bend"  # 如果没有，可以创建一个
    resize(folder_path, output_folder)

    folder_path   = "../../../DataSet-images/STARE_sim/train/fake_gtvessel_bend_old"
    output_folder = "../../../DataSet-images/STARE_sim/train/fake_gtvessel_bend"  # 如果没有，可以创建一个
    resize(folder_path, output_folder)
