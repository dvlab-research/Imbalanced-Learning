#!/bin/bash
#SBATCH --job-name=reslt_bt128_t2
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=Inat.log
#SBATCH --gres=gpu:4
#SBATCH -c 24 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 


source activate py3.6pt1.5

python ImageNetTrain_reslt_ride.py \
  --arch ResNeXt50Model \
  --mark ResNeXt50Model_reslt_ride \
  -dataset ImageNet \
  --data_path /research/dept6/jqcui/Data/ImageNet \
  -b 128 \
  --epochs 180 \
  --num_works 40 \
  --lr 0.05 \
  --weight-decay 5e-4 \
  --beta 0.99 \
  --gamma 0.7 \
  --after_1x1conv \
  --num_classes 1000 
