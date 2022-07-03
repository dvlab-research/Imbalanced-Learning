#!/bin/bash
#SBATCH --job-name=CIFAR10V2
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=CIFAR100V2_imf0.02_res32x1_beta0.994.log
#SBATCH --gres=gpu:1
#SBATCH -c 2

## Below is the commands to run , for this example,
## Create a sample helloworld.py and Run the sample python file 
## Result are stored at your defined --output location

python cifarTrain_reslt_cifar10.py \
  -mark CIFAR10V2_imf0.1_res32x1_beta0.9600 \
  --arch ResLTResNet32 \
  --scale 1 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  -dataset CIFAR10V2 \
  --imb_factor 0.1 \
  -num_classes 10 \
  -b 128 \
  --epochs 200 \
  --beta 0.9600
