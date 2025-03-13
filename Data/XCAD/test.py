
import cv2
import numpy as np

light = cv2.imread('fake_very_smalltheta/2.png', cv2.IMREAD_GRAYSCALE)
light = light.astype(np.float32)/255
print(light.shape,type(light),light[0,0])



def test(name):


    # 1. 读取灰度图片
    input_image_path = 'fake_very_smalltheta/train/bg/'+name+'.jpg'  # 替换为你的图片路径
    output_image_path = 'fake_very_smalltheta/train/'+name+'_bg.jpg'       # 替换为你希望保存的路径

    # 使用 OpenCV 读取图片并转换为灰度图
    gray_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 2. 保存图片到另一个位置
    cv2.imwrite(output_image_path, gray_image)


    image_new = (gray_image.astype(np.float32)/255)*light*255
    cv2.imwrite(
        'fake_very_smalltheta/train/'+name+'_2.jpg', 
        image_new)


def test2():
    for i in range(30):
        test(str(i+1))
test2()