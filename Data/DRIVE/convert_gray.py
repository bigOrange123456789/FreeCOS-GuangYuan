import numpy as np
import random
import math
from PIL import Image
import os
#Img_dir = "./DRIVE_test/1st_manual/"
#Orgin_dir = "./DRIVE_test/images/"
#Img_dir = "./fake_rgbvessel/"
# Img_dir = "./train/fake_rgbvessel_more/"
# Save_dir  ="./train/fake_vessel_more/"
Img_dir = "./train/fake_onlythin_vessel/"
Save_dir  ="./train/fake_onlythin_vessel_gray/"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)
files = os.listdir(Img_dir)
i = 0
list_RGB= []
idx = 300
for image_dir in files:
    print("image_dir",image_dir)
    image =  Image.open(Img_dir+image_dir).convert("L")
    image_array = np.asarray(image)
    print("image_array",image_array.shape)
    # save_path = Save_dir + image_dir
    save_path = Save_dir + str(idx) + '.png'
    idx += 1
    Image.fromarray((image_array).astype('uint8')).convert('L').save(save_path)