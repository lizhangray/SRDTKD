# DTKD

## [KBS 2025] 《Reliable Image Super-Resolution Using Dual-Teacher Knowledge Distillation》

------

> **Abstract:** *Reliable image super-resolution (SR) requires recovery of accurate textures and trustworthy details for real-world applications. It is challenging for SR methods to achieve both high reconstruction fidelity and desirable perceptual quality, even at the cost of excessive computational complexity. As a model-compression technique, knowledge distillation (KD) provides a solution for training an efficient lightweight student model with the guidance of a high-performance teacher network. However, untrustworthy teachers that generate false textures will increase generalization errors in SR KD, resulting in a performance detriment of their student. To address these issues, we present a theoretical analysis of errors that arise in SR KD and discuss various conditions for distinguishing \``good’’ teachers that generate reliable and learnable textures. Based on our theoretical criteria, we propose a dual-teacher KD (DTKD) framework that incorporates both fidelity and perceptual teachers to train lightweight and balanced SR student models. To reduce errors in KD training, we design plug-in modules of image entropy routing and two attention loss functions. To construct a ``good'' perceptual teacher for KD training, we further design an edge-guided SR network, called EdgeSRN, which replaces generative adversarial networks by incorporating edge-enhanced learning to reduce artifacts. Extensive evaluations regarding both the reconstruction accuracy and perceptual quality verify that student models trained using the proposed DTKD outperform other state-of-the-art SR methods with fewer network parameters and lower computation costs.*

## Datasets

| Crowd Dataset          | Link                                                         |
| ---------------------- | ------------------------------------------------------------ |
| Hazy-JHU               | [[**Google**](https://drive.google.com/file/d/1rLQ_oXHFAUqaYktk-3OFpHHk7uohEcNt/view?usp=sharing)] \| [[**BaiduNetdisk**](https://pan.baidu.com/s/1YZuWGhxZGyFmwVRntamCvA?pwd=xhcm)] |
| Hazy-ShanghaiTech      | [[**Google**](https://drive.google.com/file/d/1ibvFlZ-sdd_A6xEI1cFuXk4_hHf409Mt/view?usp=sharing)] \| [[**BaiduNetdisk**](https://pan.baidu.com/s/197CyDnxarjCL3O66yIfNwQ?pwd=jky9)] |
| Hazy-ShanghaiTechRGBD  | [[**Google**](https://drive.google.com/file/d/1rJD9IBuKA1Nhm-Ek3yDe-8V11CLKZnaG/view?usp=drive_link)] |
| Rainy-ShanghaiTechRGBD | [[**Google**](https://drive.google.com/file/d/1uCeHtVO1_Mnc3KnOKzLd0JyOUhzzKKNo/view?usp=sharing)] |

## Example

<table>
  <tr>
    <td align="center"> <img src = "./assets/results/Set5_head_LR.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Set5_head_DTKD-RFDN.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Set5_head_DTKD-LBNet.png" width="240" height="240" > </td>
  </tr>
  <tr>
    <td align="center"> <img src = "./assets/results/Set14_baboon_LR.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Set14_baboon_DTKD-RFDN.png"width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Set14_baboon_DTKD-LBNet.png" width="240" height="240" > </td>
  </tr>
  <tr>
    <td align="center"> <img src = "./assets/results/BSD100_37073_LR.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/BSD100_37073_DTKD-RFDN.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/BSD100_37073_DTKD-LBNet.png" width="240" height="240" > </td>
  </tr>
  <tr>
    <td align="center"> <img src = "./assets/results/Urban100_img_067_LR.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Urban100_img_067_DTKD-RFDN.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Urban100_img_067_DTKD-LBNet.png" width="240" height="240"  > </td>
  </tr>
    <tr>
    <td align="center"> <img src = "./assets/results/Manga109_YumeiroCooking_LR.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Manga109_YumeiroCooking_DTKD-RFDN.png" width="240" height="240" > </td>
    <td align="center"> <img src = "./assets/results/Manga109_YumeiroCooking_DTKD-LBNet.png" width="240" height="240"  > </td>
  </tr>
  <tr>
    <td align="center"><p><b>Input</b></p></td>
    <td align="center"><p><b>DTKD-RFDN</b></p></td>
    <td align="center"><p><b>DTKD-LBNet</b></p></td>
  </tr>
</table>


| Image Name                          | Predict | Ground-truth |
| ----------------------------------- | ------- | ------------ |
| Rainy-ShanghaiTechRGBD/IMG_0895.jpg | 15      | 14           |
| Hazy-ShanghaiTechRGBD/IMG_3.jpg     | 91      | 85           |
| Hazy-ShanghaiTech/PartA/IMG_160.jpg | 117     | 121          |
| Hazy-JHU/IMG_0895.jpg               | 1200    | 945          |

</div>

## Usage

#### Pre-trained Models

- **Hazy-JHU** → [Hazy_JHU_best.pth](https://drive.google.com/file/d/18saECAlz6mc7_neo8_uLeBrc7xs5UKVf/view?usp=sharing)
- **Hazy-ShanghaiTech PartA** → [DH_SHTA_best.pth](https://drive.google.com/file/d/1DrVEb2exzgO17ZbtoaJZctgTiqRaiuMo/view?usp=sharing)
- **Hazy-ShanghaiTech PartB** → [DH_SHTB_best.pth](https://drive.google.com/file/d/1Tu9VH0FmWyMTTwe8rqQt3gq_U2mUZGY3/view?usp=share_link)
- **Hazy-ShanghaiTechRGBD** → [Hazy_SHTRGBD_best.pth](https://drive.google.com/file/d/1jQv0Kj8aT_PGUi4LzWppiGPpXQtq15uG/view?usp=sharing)
- **Rainy-ShanghaiTechRGBD** → [Rainy_SHTRGBD_best.pth](https://drive.google.com/file/d/1Fqr7RqSJk-fSUw9YMpCg2MZnsBhynFWJ/view?usp=sharing)
- [vgg16_bn-6c64b313.pth](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth)

#### Environment

```bash
torch
torchvision
tensorboardX
easydict
pandas
numpy
scipy
matplotlib
Pillow
opencv-python
```

#### Install

```bash
git clone https://github.com/lizhangray/Dehaze-P2PNet.git
pip install -r requirements.txt
```

#### Download Datasets and Pre-trained Models to Prepare Your Directory Structure

```bash
Dehaze-P2PNet
    |- assets
    |- crowd_datasets
    |- datasets
        |- Hazy_JHU
            |- test_data
            |- train_data
            |- val
        |- Hazy_ShanghaiTech
            |- PartA
                |- test_data
                |- train_data
                |- val
                ....
        |- Hazy_ShanghaiTechRGBD
        ....
    |- models
    |- util
    |- weights
        |- DH_SHTA_best.pth
        |- DH_SHTB_best.pth
        ....
        |- vgg16_bn-6c64b313.pth
    |- engine.py
    |- run_test.py
```

#### How To Test

```bash
python run_test.py --dataset_file NAME_OF_DATASET --weight_path CHECKPOINT_PATH

# e.g., Hazy-JHU
python run_test.py --dataset_file Hazy_JHU --weight_path weights/Hazy_JHU_best.pth

# e.g., Hazy-ShanghaiTech PartA
python run_test.py --dataset_file Hazy_SHTA --weight_path weights/DH_SHTA_best.pth

# e.g., Hazy-ShanghaiTech PartB
python run_test.py --dataset_file Hazy_SHTB --weight_path weights/DH_SHTB_best.pth

# e.g., Hazy-ShanghaiTechRGBD
python run_test.py --dataset_file Hazy_SHARGBD --weight_path weights/Hazy_SHTRGBD_best.pth

# e.g., Rainy-ShanghaiTechRGBD
python run_test.py --dataset_file Rainy_SHARGBD --weight_path weights/Rainy_SHTRGBD_best.pth
```

There are two parameters that must be provided:

`'--dataset_file', help='(Hazy_JHU | Hazy_SHARGBD | Hazy_SHTA | Hazy_SHTB | Rainy_SHARGBD)'`

`'--weight_path', help='load pretrained weight from checkpoint', such as 'weights/Hazy_JHU_best.pth'`

## Citation

Please cite this paper in your publications if it is helpful for your tasks.

```tex
@InProceedings{yuan2024crowd,
    author    = {Yuan, Weijun and Li, Zhan and Li, Xiaohan and Fang, Liangda and Zhang, Qingfeng and Qiu, Zhixiang},
    title     = {Crowd Counting and Localization in Haze and Rain},
    booktitle = {2024 IEEE International Conference on Multimedia and Expo (ICME)},
    year      = {2024}
}

```
