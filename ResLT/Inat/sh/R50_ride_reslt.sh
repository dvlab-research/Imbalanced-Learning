#!/bin/bash
#SBATCH --job-name=Inat
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=Inat.log
#SBATCH --gres=gpu:4
#SBATCH -c 24 
#SBATCH --constraint=ubuntu18
#SBATCH -p batch_72h 

source activate py3.6pt1.5

python iNaturalTrain_reslt_ride.py \
  --arch ResNet50Model \
  --mark ResNet50Model \
  -dataset iNaturalist2018 \
  --data_path /research/dept6/jqcui/Data/iNaturalist2018/ \
  -b 256 \
  --epochs 200 \
  --num_works 40 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --beta 0.85 \
  --gamma 0.3 \
  --after_1x1conv \
  --num_classes 8142 
