source activate py3.6pt1.5

python ImageNetTrain_reslt.py \
  --arch resnet10_reslt \
  --mark resnet10_reslt_bt256 \
  -dataset ImageNet \
  --data_path /research/dept6/jqcui/Data/ImageNet \
  -b 256 \
  --epochs 180 \
  --num_works 40 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --beta 0.99 \
  --gamma 0.3 \
  --num_classes 1000 
