# FreeCG: Free the Design Space of Clebsch–Gordan Transform for Machine Learning Force Fields

## Overview

FreeCG free the design space of Clebsch-Gordan transform, enabling high-expressive and efficient geometric neural network. This codebase is based on ViSNet (https://github.com/microsoft/AI2BMD/tree/ViSNet). 


## Environments

- Clone this repository

- Install the dependencies

```shell
conda create -y -n freecg python=3.9
conda activate freecg
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg==2.1.0 -c pyg
pip install pytorch-lightning==1.8.0
pip install ase ase[test] ogb
pip install e3nn
```

## Getting started

To train FreeCG, please run:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf configs/default_configs.yml --dataset-root /path/to/data --log-dir /path/to/log
```

One can modify the ```dataset-arg``` to train the desired molecules or properties. The default setting is for training FreeCG on aspirin from rMD17.

## Inference

Once FreeCG is trained, to use a pretrained checkpoint for inference, simply run:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --conf configs/default_configs.yml --dataset-arg dataset-arg --dataset-root /path/to/data --log-dir /path/to/log --task inference --load-model /path/to/ckpt
```

## Contact

Please contact Shihao Shao (shaoshihao@pku.edu.cn) if you have any question.

## License

This project is licensed under the terms of the MIT license. 
