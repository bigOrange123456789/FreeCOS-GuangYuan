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
