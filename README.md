# Art2Real
This repository contains the reference code for the paper _[Art2Real: Unfolding the Reality of Artworks via Semantically-Aware Image-to-Image Translation](https://arxiv.org/pdf/1811.10666)_ (CVPR 2019).

Please cite with the following BibTeX:

```
@inproceedings{tomei2019art2real,
  title={{Art2Real: Unfolding the Reality of Artworks via Semantically-Aware Image-to-Image Translation}},
  author={Tomei, Matteo and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

<p align="center">
<img src="images/samples01.gif" alt="Art2Real" />
</p>

## Requirements

This code is built on top of the [Cycle-GAN source code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

The required Python packages are:
* torch>=0.4.1
* torchvision>=0.2.1
* dominate>=2.3.1
* visdom>=0.1.8.3

## Pre-trained Models

* **Monet2Photo** [[Checkpoints]](https://drive.google.com/drive/folders/1XciFP86aKuYoWUWKXgGaBVrZrWx6vjxe?usp=sharing) [[Dataset]](https://drive.google.com/file/d/14fZ-Qu-AwQ3Yy8xJZZxzrnumirkjt4qC/view?usp=sharing)
* **Landscape2Photo** [[Checkpoints]](https://drive.google.com/drive/folders/1rmYKPYFu3FGwfrkAAGaqn1G7R-fODj41?usp=sharing) [[Dataset]](https://drive.google.com/file/d/17UMyQQkUrDk3CwmelaYW_Tj6E8vkBoUf/view?usp=sharing)
* **Portrait2Photo** [[Checkpoints]](https://drive.google.com/drive/folders/12Vr6oceBzi4NWRZsyF7eUg-3WTZdh8Lv?usp=sharing) [[Dataset]](https://drive.google.com/file/d/1Q1VkesUNZXPBafUgZ2FIaIKoFTSG7YN1/view?usp=sharing)

Download pre-trained models and place them under the checkpoint folder. For example, when downloading the monet2photo checkpoints, place them under the folder `./checkpoints/monet2photo/`.

## Test

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--dataroot` | Dataset root folder containing the `testA` directory |
| `--name ` | `monet2photo`, `landscape2photo`, `portrait2photo` |
| `--num_test ` | Number of test samples |

For example, to reproduce the results of our model for the first 100 test samples of the landscape2photo setting, use:
```
python test.py --dataroot ./datasets/landscape2photo --name landscape2photo --num_test 100
```


## Training

The training code will be available soon.

<p align="center">
<img src="images/samples02.gif" alt="Art2Real" />
<img src="images/samples03.gif" alt="Art2Real" />
</p>

