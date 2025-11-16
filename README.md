# SRDTKD

## [KBS 2025] 《Reliable Image Super-Resolution Using Dual-Teacher Knowledge Distillation》

------

> **Abstract:** *Reliable image super-resolution (SR) requires recovery of accurate textures and trustworthy details for real-world applications. It is challenging for SR methods to achieve both high reconstruction fidelity and desirable perceptual quality, even at the cost of excessive computational complexity. As a model-compression technique, knowledge distillation (KD) provides a solution for training an efficient lightweight student model with the guidance of a high-performance teacher network. However, untrustworthy teachers that generate false textures will increase generalization errors in SR KD, resulting in a performance detriment of their student. To address these issues, we present a theoretical analysis of errors that arise in SR KD and discuss various conditions for distinguishing \``good’’ teachers that generate reliable and learnable textures. Based on our theoretical criteria, we propose a dual-teacher KD (DTKD) framework that incorporates both fidelity and perceptual teachers to train lightweight and balanced SR student models. To reduce errors in KD training, we design plug-in modules of image entropy routing and two attention loss functions. To construct a ``good'' perceptual teacher for KD training, we further design an edge-guided SR network, called EdgeSRN, which replaces generative adversarial networks by incorporating edge-enhanced learning to reduce artifacts. Extensive evaluations regarding both the reconstruction accuracy and perceptual quality verify that student models trained using the proposed DTKD outperform other state-of-the-art SR methods with fewer network parameters and lower computation costs.*



#### Single Image SR Reconstruction: See clearer, See More
< img src="./assets/figures/fig1.png" width="720" height="320">

#### Error Analysis of Knowledge Distillation in Super Resolution
< img src="./assets/figures/fig2.png" width="720" height="200">

#### Applications
< img src="./assets/figures/fig3.png" width="720" height="450">



## Datasets

| Dataset          | Link                                                         |
| :----------------------: | :------------------------------------------------------------: |
| DIV2K | [[**Website**](https://data.vision.ee.ethz.ch/cvl/DIV2K/)] |
| Set5+Set14+BSD100+Urban100+Manga109+RealSRSet | [[**Google**](https://drive.google.com/drive/folders/1XsTVrkrTYga0-E_jfoavi2x07qWw2opa?usp=sharing)] |

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


## Usage

#### Pre-trained Models

- **DTKD-LBNet** → [DTKD-LBNet.pth](https://pan.quark.cn/s/7074834cfec9)
- **DTKD-RFDN** → [DTKD-RFDN.pth](https://pan.quark.cn/s/7074834cfec9)
- **DTKD-LBNet-perceptual** → [DTKD-LBNet-perceptual.pth](https://pan.quark.cn/s/7074834cfec9)
- **DTKD-RFDN-perceptual** → [DTKD-RFDN-perceptual.pth](https://pan.quark.cn/s/7074834cfec9)
- **EdgeSRN** → [EdgeSRN_x4.pth](https://pan.quark.cn/s/7074834cfec9)
- **SwinIR-S** → [lightweightSR_SwinIRx4.pth](https://pan.quark.cn/s/7074834cfec9)

#### Environment

```bash
torch
torchvision
easydict
pandas
numpy
scipy
matplotlib
Pillow
opencv-python
scikit-image
```

#### Install

```bash
git clone https://github.com/lizhangray/SRDTKD.git
pip install -r requirements.txt
```

#### Download Datasets and Pre-trained Models to Prepare Your Directory Structure

```bash
 DTKD
    |- assets
    |- Checkpoints
        |- EdgeSRN
            |- EdgeSRN_x4.pth
        |- LBNet
            |- DTKD-LBNet.pth
            |- DTKD-LBNet-perceptual.pth
        |- RFDN
            |- DTKD-RFDN.pth
            |- DTKD-RFDN-perceptual.pth
        |- SwinIR
            |- lightweightSR_SwinIRx4.pth
    |- Datasets
    |- Datasets2023
        |- GT
            |- BSD100
            |- Manga109
                ....
        |- GTmod12
            |- BSD100_GTmod12
            |- Manga109_GTmod12
                ....
        |- GTmod12_LRx4
            |- BSD100_LRbicx4
            |- Manga109_LRbicx4
                ....
        |- RealSRSet
        ....
    |- Model
    |- Utils
    |- demo.sh
    |- main.py
    |- requirements.txt
    |- Trainer.py
```

#### How To Infer

```bash
python main_for_infer.py --Train False ----model_name NAME_OF_MODEL --checkpoint CHECKPOINT_PATH --test_folder TESTSET_PATH

# e.g., infer DTKD-RFDN in Set5
python main_for_infer.py --Train False --model_name RFDN --checkpoint DTKD-RFDN.pth --test_folder Datasets2023/GTmod12_LRx4/Set5_LRbicx4

# e.g., infer DTKD-LBNet in Set5
python main_for_infer.py --Train False --model_name LBNet --checkpoint DTKD-LBNet.pth --test_folder Datasets2023/GTmod12_LRx4/Set5_LRbicx4

# e.g., infer DTKD-RFDN in Urban100
python main_for_infer.py --Train False --model_name RFDN --checkpoint DTKD-RFDN.pth --test_folder Datasets2023/GTmod12_LRx4/Urban100_LRbicx4

# e.g., infer DTKD-LBNet in Urban100
python main_for_infer.py --Train False --model_name LBNet --checkpoint DTKD-LBNet.pth --test_folder Datasets2023/GTmod12_LRx4/Urban100_LRbicx4
```

There are four parameters that must be provided:

`'--Train', defalt=False`

`'--model_name', help='(RFDN | LBNet | EdgeSRN | SwinIR)'`

`'--checkpoint', help='load pretrained weight from checkpoint', such as 'DTKD-RFDN.pth'`

`'--test_folder', help='load testset from folder', such as 'Datasets2023/GTmod12_LRx4/Set5_LRbicx4'`

#### Visual Results

- **DTKD-LBNet** → [DTKD-LBNet_x4](https://pan.quark.cn/s/8e9c1e2a8ce5)
- **DTKD-RFDN** → [DTKD-RFDN_x4](https://pan.quark.cn/s/8e9c1e2a8ce5)
- **DTKD-LBNet-perceptual** → [DTKD-LBNet-perceptual_x4](https://pan.quark.cn/s/8e9c1e2a8ce5)
- **DTKD-RFDN-perceptual** → [DTKD-RFDN-perceptual_x4](https://pan.quark.cn/s/8e9c1e2a8ce5)
- **EdgeSRN** → [EdgeSRN_x4](https://pan.quark.cn/s/8e9c1e2a8ce5)

## Citation

Please cite this paper in your publications if it is helpful for your tasks.

```tex
@article{li2025reliable,
  title={Reliable Image Super-Resolution Using Dual-Teacher Knowledge Distillation},
  author={Li, Zhan and Yuan, Weijun and Yao, Boyang and Chen, Yihang and Bhanu, Bir and Zhang, Kehuan},
  journal={Knowledge-Based Systems},
  pages={114843},
  year={2025},
  publisher={Elsevier}
}

```
