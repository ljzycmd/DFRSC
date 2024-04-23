## Rolling Shutter Correction with Intermediate Distortion Flow Estimation

This repo is under construction.

[Mingdeng Cao](https://github.com/ljzycmd),
[Sidi Yang](https://ieeexplore.ieee.org/author/37088955345),
[Yujiu Yang](https://scholar.google.com/citations?user=4gH3sxsAAAAJ),
[Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ) <br>

[Paper](https://arxiv.org/abs/2404.06350)&ensp;|&ensp;[Checkpoints](https://github.com/ljzycmd/DFRSC/releases/tag/ckpts)&ensp;|&ensp;[Results on BS-RSC](https://github.com/ljzycmd/DFRSC/releases/download/results/DFRSC_3F_BSRSC.zip)&ensp;|&ensp;[Results on Fastec-RS](https://github.com/ljzycmd/DFRSC/releases/download/results/DFRSC_3F_FastecRS.zip)&ensp;|&ensp;[Results on Carla-RS](https://github.com/ljzycmd/DFRSC/releases/download/results/DFRSC_3F_CarlaRS.zip)

> We propose to rectify Rolling Shutter (RS) distorted images by directly estimating the intermediate distortion flow from the underlying Global Shutter (GS) image to the RS image. This method differs from previous methods that calculate undistortion flow and apply forward warping to obtain the GS image. Instead, the proposed method directly estimates the non-linear distortion flow and employs backward warping to acquire the corrected image. More specifially, the frame-wise RS features are firstly obtained by a multi-scale encoder. After that, a global correlation-based attention mechanism is proposed to to jointly obtain the initial distortion flow and GS features. Then, the coarse-to-fine decoder refines and upscales the resolu-tion of the flow and GS features simultaneously. The final GS image is obtained by a multi-flow predicting strategy.

## Quick Start

### Dependencies

Clone the repo and install corresponding packages:

```bash
git clone https://github.com/ljzycmd/DFRSC.git
cd DFRSC
pip install -r requirements.txt
```

### Evaluation

1. Download pretrained checkpoints

    | Model | Dataset | #Num Frames | Link |
    | -------  | --------- |:------:|:------:|
    | DFRSC-3F | BS-RSC  | 3  | [Github](https://github.com/ljzycmd/DFRSC/releases/download/ckpts/dfrsc_3f_bsrsc.pth)  |
    | DFRSC-3F | Fastec-RS | 3  | [Github](https://github.com/ljzycmd/DFRSC/releases/download/ckpts/dfrsc_3f_fastecrs.pth) |
    | DFRSC-3F | Carla-RS  |  3 | [Github](https://github.com/ljzycmd/DFRSC/releases/download/ckpts/dfrsc_3f_fastecrs.pth) |

2. Prepare data

    You can download the following datasets to calculate metrics:

    * [BS-RSC](https://github.com/ljzycmd/BSRSC)
    * [Fastec-RS](https://github.com/ethliup/DeepUnrollNet)
    * [Carla-RS](https://github.com/ethliup/DeepUnrollNet)

3. Usage

    You should modify the pretrained path in the corresponding config file.

    ```bash
    python test.py
    ```

### Training

1. Prepare dataset

    Download the BS-RSC, Fastec-RS and Carla-RS and specify the dataset root in the training configs (`.yaml` file).

2. Start training

    ```bash
    bash train.sh
    ```

    You my modify the config file path in the `train.sh` to train the model with different datasets.

## Acknowlements

The code is implemented upon the open-soured image restoration framework [BasicSR](https://github.com/XPixelGroup/BasicSR), we thank the authors for relasing such an awesome framework.

## Citation

If the proposed model is useful for your research, please consider citing our paper

```bibtex
@InProceedings{Cao_2024_CVPR,
    author    = {Cao, Mingdeng and Yang, Sidi and Yang, Yujiu and Zheng, Yinqiang},
    title     = {Rolling Shutter Correction with Intermediate Distortion Flow Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

## Contact

If you have any questions about our project, please feel free to contact me at `mingdengcao [AT] gmail.com`.
