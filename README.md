## Code for GaitSTR: Gait Recognition with Sequential Two-stream Refinement

This folder contains the code and pretrained model for GaitSTR: Gait Recognition with Sequential Two-stream Refinement ([[Paper](https://arxiv.org/pdf/2404.02345)]). We provide the Python script with the processed data, as well as instructions on how to prepare the dataset on your own.

### Environment Setup

 We have tested our code and model on a single NVIDIA 3090 gpu with Centos 8 as well as A40 gpu on Ubuntu 18.04, with python 3.8.0, CUDA 11.6, pytorch 1.13.1.

 A suggestion for this is to use conda and create an environment as follows

```
conda create -n gaitstr python=3.8.0
conda activate gaitstr
```

 After you create a python 3.8.0, please use the following command for installing required PyTorch

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Data Preparation and Pretrained Models

 Since silhouettes and skeletons for CASIA-B are publicly available online ([[Silhouettes Link](http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip)], [[Skeleton Link](https://github.com/tteepe/GaitGraph/releases/download/v0.1/data.zip)]), for simplicity, we directly provide the processed data we used along with the pretrained models in the following link [[Processed Dataset and Pretrained Model](https://drive.google.com/drive/folders/165eooQqa-7lzzvb1scE2xwuOoIQLVoKR?usp=sharing)]. Please download both of the files and place them in the current directory. Unzip them with the commands below

```
tar -xzf CASIA-B-mix.tar.gz
tar -xzf pretrained.tar.gz
```

 For other datasets, please contact the dataset owner for downloading the silhouettes and skeletons. You could generate the orientation files with following script.

```
python scripts/gen_orientation.py
```

### Reproduce the results on CASIA-B dataset

 To produce the numbers for GaitSTR, please use the following command and replace the GPU id with the id you want (>6 GB memory available and please only use ONE gpu for the default config)

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/gaitstr.yaml --phase test
```

 For training, please use the following

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/gaitstr.yaml --phase train
```