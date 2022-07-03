# ResLT: Residual Learning for Long-tailed Recognition (TPAMI 2022)
This repository contains the implementation code for paper:  
**Residual Learning for Long-tailed Recognition** https://arxiv.org/abs/2101.10633    
  
If you find this code or idea useful, please consider citing our work:
```
@article{cui2021reslt,
  title={ResLT: Residual Learning for Long-tailed Recognition},
  author={Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya},
  journal={arXiv preprint arXiv:2101.10633},
  year={2021}
}
```  


# Updates
We further verifty the proposed ResLT is complementary to ensemble-based methods. Equipped with RIDEResNeXt, our model achieves better results. All experiments are conducted without knowledge distillation for fair comparison. For RIDE, we use their public code and train 180 epochs.

### ImageNet-LT
Model |Top-1 Acc | Download | log 
---- | --- | --- | ---
RIDEResNeXt(3 experts)        | 55.1 | - | [log](https://drive.google.com/file/d/1Xv06BOrpFoj7eArwTBnHU2nVVYM016CB/view?usp=sharing)
RIDEResNeXt-ResLT(3 experts)  | 57.6 | [model](https://drive.google.com/file/d/1IgC9N4LbjRDqM2N1dndKIAwFB-5giUgK/view?usp=sharing) | [log](https://drive.google.com/file/d/15yeaaOvY596lT5cv0vXUyVEVaVJJBP_b/view?usp=sharing)

### Inaturalist 2018
Model |Top-1 Acc | Download | log 
---- | --- | --- | ---
RIDEResNeXt(3 experts)        | 70.8 | - | [log](https://drive.google.com/file/d/1ssoFupuQJ0k37TVmMAWSGlB8aXt1xqCZ/view?usp=sharing)
RIDEResNeXt-ResLT(3 experts)  | 72.9 | [model](https://drive.google.com/file/d/11xE2KUvb5M8caR1MEWOYp1BnUO7Ny-vO/view?usp=sharing) | [log](https://drive.google.com/file/d/1Z_IDcuf5nuP0eISQOOyaLXB4vV2DuujN/view?usp=sharing)




# Overview
In this paper, we proposed a residual learning method to address long-tailed recognition, which contains a **Residual Fusion Module** and a **Parameter Specialization Mechanism**.
With extensive ablation studies, we demonstrate the effectiveness of our method.  

![image](https://github.com/FPNAS/ResLT/blob/main/assets/reslt.jpg)

# Get Started
## ResLT Training
For CIFAR, due to the small data size, different experimental environment can have a big difference. To achieve the reported results, you may need to slightly tune the $\alpha$.
```
bash sh/CIFAR100/CIFAR100LT_imf0.01_resnet32sx1_beta0.9950.sh
```
For ImageNet-LT,

```
bash sh/X50.sh
```

For iNaturalist 2018,

```
bash sh/R50.sh
```

## Results and Models
### CIFAR
Model | Download
---- | ---
CIFAR-10-imb0.01 | -
CIFAR-10-imb0.02 | -
CIFAR-10-imb0.1  | -
CIFAR-100-imb0.01 | -
CIFAR-100-imb0.02 | -
CIFAR-100-imb0.1  | -

### ImageNet-LT
Model | Download | log 
---- | --- | ---
ResNet-10   | [model](https://drive.google.com/file/d/1n1s68gQkty1bl0hNL9rd-56PcBg21tkM/view?usp=sharing) | [log](https://drive.google.com/file/d/1NVyS-bihpkYOuJ4-KHezfVK5Sw0-gQVB/view?usp=sharing)
ResNeXt-50  | [model](https://drive.google.com/file/d/1W7uv5O18LHyX6jUWtLF2s9LtwAEtvgG_/view?usp=sharing) | [log](https://drive.google.com/file/d/1pYCHSaYAJ7g75lMbHvOgJUlMDiGohGSv/view?usp=sharing)
ResNeXt-101 | [model](https://drive.google.com/file/d/1qyIShPI2UIt1e9IIouScHy0UC146ZTuL/view?usp=sharing) | [log](https://drive.google.com/file/d/1xM5wJECYpHHYE4erJhLs8BSwrpr6OzmI/view?usp=sharing)



### iNatualist 2018
Model | Download | log
---- | ---- | ----
ResNet-50 | [model](https://drive.google.com/file/d/1xxUYtJ0hJHUwb-JaZ5kdovF_lowyEZSD/view?usp=sharing) | [log](https://drive.google.com/file/d/1VlZryf03ujUPhhQ8eTYYM23BsD9oOTg1/view?usp=sharing)

### Places-LT
Model | Download | log
---- | ---- | ----
ResNet-152 | - | -

# Acknowledgements
This code is partly based on the open-source implementations from offical PyTorch examples and [LDAM-DRW](https://github.com/kaidic/LDAM-DRW).

# Contact
If you have any questions, feel free to contact us through email (jiequancui@link.cuhk.edu.hk) or Github issues. Enjoy!











