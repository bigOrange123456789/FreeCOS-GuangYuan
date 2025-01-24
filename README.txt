export PATH="~/anaconda3/bin:$PATH"
source activate FreeCOS
# python Data/XCAD/make_fakevessel.py #make_fakevessel.py
cd ./Data/XCAD
python make_fakevessel.py
cd ../../
python train_DA_contrast_liot_finalversion.py

#### 3. Evaluation scripts

```bash

CUDA_VISIBLE_DEVICES=0 python test_DA_thresh.py

```

#### 4. Trained models
Trained models can be downloaded from here. [[Google Drive](https://drive.google.com/drive/folders/1nLgsTQYKXHP3QlHg9RQmPuPNM3UKcCKY?usp=share_link)] [[Baidu Drive](https://pan.baidu.com/s/1hyj-3rlQ8X_Fj9sTygpacA) (download code: 3w1a)].   
Put the weights in the "logs/" directory.  

#### 5. Trained Data
Trained Data can be down from here. [[Google Drive](https://drive.google.com/drive/folders/1nLgsTQYKXHP3QlHg9RQmPuPNM3UKcCKY?usp=share_link)] [[Baidu Drive](https://pan.baidu.com/s/1hyj-3rlQ8X_Fj9sTygpacA) (download code: 3w1a)] (you can generate different curvilinear data by our method or use the same generated curvilinear data as our experiment. Different generated curvilinear data will effect the performance)

## Future Work

-FreeCOS will be continuously updated.

-Thanks for the parts of LIOT codes from "Local Intensity Order Transformation for Robust Curvilinear Object Segmentation" (https://github.com/TY-Shi/LIOT), we changes the LIOT to a online way in FreeCOS codes.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- shitianyihust@hust.edu.cn
- shitianyihust@126.com
########################################################
idea1:(1)改为3D血管合成【已完成】；(2)如何合成弯曲血管；
idea2:(1)添加EMA，并使用一致性学习；(2)使用双解码器；
idea3:感觉这里对抗学习在逻辑上有问题，解决掉这种不适感
idea4:对比学习：强正样本、弱正样本
idea5:(1)使用傅里叶变换进行域迁移。(2)找出主方向。频率的主方向、是否对应树形结构的上下。
idea6:不平衡学习



