# Imbalanced-Learning

Imbalanced learning for imbalanced recognition and segmentation, including MiSLAS, PaCo, ResLT, RR, and ResCom developed by CUHK, [Deep Vision Lab](https://www.dvlab.ai).

**News**

- 2022/07 The code of [ResCom](https://arxiv.org/abs/2203.11506) has been released!

- 2022/06 The paper of [ResLT](https://arxiv.org/abs/2101.10633) (ResLT: Residual Learning for Long-Tailed Recognition) is accepted by TPAMI 2022.

- 2022/04 The paper of [RR](https://arxiv.org/abs/2204.01969) (Region Rebalance for Long-Tailed Semantic Segmentation) is available on arXiv.

- 2022/03 The paper of [ResCom](https://arxiv.org/abs/2203.11506) (Rebalanced Siamese Contrastive Mining for Long-Tailed Recognition) is available on arXiv.

- 2021/07 The paper of [PaCo](https://arxiv.org/abs/2107.12028) (Paramateric Contrastive Learning) is accepted by ICCV 2021.

- 2021/03 The paper of [MiSLAS](https://arxiv.org/abs/2104.00466) (Improving Calibration for Long-Tailed Recognition) is accepted by CVPR 2021.


---



## Imbalanced Recognition

### ResCom

The repo [./ResCom](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/ResCom) provides ResCom's **trained models**, **trained log**, and **code** for PyTorch.
### ResLT
The repo [./ResLT](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/ResLT) provides ResLT's **trained models**, **trained log**, and **code** for PyTorch.

### PaCo

The repo [./PaCo](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/PaCo) provides PaCo's **trained models**, **trained log**, and **code** for PyTorch.


### MiSLAS
The repo [./MiSLAS](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/MiSLAS) provides MiSLAS's **trained models**, and **code** for PyTorch.



## Imbalanced Segmentation

### RR
todo



## Citation

Please consider citing our papers in your publications if they help your research. 

If you have any questions, feel free to contact us through email or Github issues. Thanks!



```
@article{cui2022reslt,
  title={ResLT: Residual Learning for Long-Tailed Recognition},
  author={Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022},
}

@article{cui2022rr,
  title={Region Rebalance for Long-Tailed Semantic Segmentation},
  author={Cui, Jiequan and Yuan, Yuhui and Zhong, Zhisheng and Tian, Zhuotao and Hu, Han and Lin, Stephen and Jia, Jiaya},
  journal={arXiv preprint arXiv:2204.01969},
  year={2022}
}

@article{zhong2022rescom,
  title={Rebalanced Siamese Contrastive Mining for Long-Tailed Recognition},
  author={Zhong, Zhisheng and Cui, Jiequan and Lo, Eric and Li, Zeming and Sun, Jian and Jia, Jiaya},
  journal={arXiv preprint arXiv:2203.11506},
  year={2022}
}

@inproceedings{cui2021parametric,
    title={Parametric Contrastive Learning}, 
    author={Jiequan Cui and Zhisheng Zhong and Shu Liu and Bei Yu and Jiaya Jia},
    booktitle={IEEEinternational Conference on Computer Vision(ICCV)},
    year={2021},
}

@inproceedings{zhong2021mislas,
    title={Improving Calibration for Long-Tailed Recognition},
    author={Zhisheng Zhong, Jiequan Cui, Shu Liu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021},
}
```

