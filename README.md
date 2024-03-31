## Rolling Shutter Correction with Distortion Flow Estimation

This repo is under construction.

[Mingdeng Cao](https://github.com/ljzycmd),
[Sidi Yang](https://ieeexplore.ieee.org/author/37088955345),
[Yujiu Yang](https://scholar.google.com/citations?user=4gH3sxsAAAAJ),
[Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ) <br>

[Paper]() | [Checkpoints]() | [Visual Results]()

> We propose to correct the rolling shutter distorted images by directly estimating the intermediate **distortion flow** from the underlying global shutter image to the rolling shutter image. Unlike previous methods calculate undistortion flow and apply forward warping to obtain the GS image, the proposed method directly estimates the non-linear distortion flow and utilizes **backward warping** to obtain the corrected image. We propose a global correlation-based attention machnism to obtain the intially distortion flow and GS features jointly. Then, the coarse-to-fine decoder refines and upscales the resolution of the flow and GS features simultaneously. The final GS image is obtained by a multi-flow decoding strategy.

## Quick Start

### Dependencies

Clone the repo and install corresponding packages:

```bash
git clone https://github.com/ljzycmd/DFRSC.git
cd DFRSC
pip install -r requirements.txt
```

### Testing

1. Download pretrained checkpoints

    | Model | Dataset | Link |
    | -------  | --------- | ------ |
    | DFRSC-3F | BS-RSC  |     |
    | DFRSC-3F | Fastec-RS |      |
    | DFRSC-3F | Carla-RS  |      |

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

1. Train the model with BS-RSC dataset

    ```bash
    python train.py
    ```

2. Train the model with Fastec-RS dataset

    ```bash
    python train.py
    ```

3. Train the model with Carla-RS dataset

    ```bash
    python train.py
    ```

### Acknowlements

The code is implemented with the open-soured image restoration framework [BasicSR](https://github.com/XPixelGroup/BasicSR), we thank the developers for relasing such an awesome framework.

### Citation

If the proposed model is useful for your research, please consider citing our paper

```bibtex
@InProceedings{Cao_2024_CVPR,
    author    = {Cao, Mingdeng and Yang, Sidi and Yang, Yujiu and Zheng, Yinqiang},
    title     = {Rolling Shutter Correction with Intermediate Distortion Flow Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

### Contact

If you have any questions about our project, please feel free to contact me at `mingdengcao [AT] gmail.com`.
