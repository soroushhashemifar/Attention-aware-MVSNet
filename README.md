# Attention-Aware Multi-View Stereo
The code of the paper **Attention-Aware Multi-View Stereo**.

The original paper could be found [here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Attention-Aware_Multi-View_Stereo_CVPR_2020_paper.pdf) (CVPR2020).

---


## How to use

**Our experiment use the dataset of DTU benchmark**
### 0. Dataset

- DTU train part: gdown --id 1_Nuud3lRGaN_DOkeTNOvzwxYa2z2YRbX

### 1. Clone the source code

`git clone https://github.com/soroushhashemifar/Attention-aware-MVSNet.git`

### 2. Download testing dataset

- DTU test part: gdown --id 1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D

### 3. Train the model

`python train.py`

### 4. Generate depth map using our pre-trained model

`python eval.py`

When finished, you can find depth maps in `results` folder.

