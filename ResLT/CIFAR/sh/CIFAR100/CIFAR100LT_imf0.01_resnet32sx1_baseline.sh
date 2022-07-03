#!/bin/bash
#SBATCH --job-name=CIFAR10V2
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/proj56/jqcui_new/Ablation_CIFAR_LongTail/ImageNetLT_beta0.80_tmp
#SBATCH --gres=gpu:1
#SBATCH -c 2

## Below is the commands to run , for this example,
## Create a sample helloworld.py and Run the sample python file 
## Result are stored at your defined --output location

python cifarTrain_baseline.py -mark CIFAR100V2_imf0.01_baselinex1_5e-4 --arch Res32_3Level_baseline --lr 0.1 --weight-decay 5e-4 -dataset CIFAR100V2 --imb_factor 0.01 -num_classes 100 -b 128 -seed 0 --epochs 200 --scale 1 
