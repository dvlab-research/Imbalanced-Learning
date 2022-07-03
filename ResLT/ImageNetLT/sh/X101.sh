
source activate py3.6pt1.5

python ImageNetTrain_reslt.py \
  --arch resnext101_32x4d_reslt \
  --mark resnext101_reslt \
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
