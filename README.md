# Imbalanced-Learning

Imbalanced learning for imbalanced recognition and segmentation, including **MiSLAS**, **PaCo**, **ResLT**, **RR**, **ResCom**, **CeCo**, and **GPaCo** developed by CUHK, [Deep Vision Lab](https://www.dvlab.ai).

**News**
- 2023/05 The paper of [GPaCo](https://arxiv.org/pdf/2209.12400.pdf) (Generalized Parametric Contrastive Learning) is accepted by TPAMI 2023.

- 2023/02 The paper of [CeCo](https://arxiv.org/pdf/2301.01100.pdf) (Understanding Imbalanced Semantic Segmentation Through Neural Collaps) is accepted by CVPR 2023.

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

### PaCo/GPaCo/RR

The repo [./Parametric Contrastive Learning](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/PaCo) provides code and models for **PaCo**, **GPaCo**, and **RR**.


### MiSLAS
The repo [./MiSLAS](https://github.com/dvlab-research/Imbalanced-Learning/tree/main/MiSLAS) provides MiSLAS's **trained models**, and **code** for PyTorch.



## Imbalanced Segmentation

### RR && CeCo
todo



## Citation

Please consider citing our papers in your publications if they help your research. 

If you have any questions, feel free to contact us through email or Github issues. Thanks!



```
@ARTICLE{10130611,
  author={Cui, Jiequan and Zhong, Zhisheng and Tian, Zhuotao and Liu, Shu and Yu, Bei and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Generalized Parametric Contrastive Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3278694}}


@ARTICLE{9774921,
  author={Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ResLT: Residual Learning for Long-Tailed Recognition}, 
  year={2023},
  volume={45},
  number={3},
  pages={3695-3706},
  doi={10.1109/TPAMI.2022.3174892}
  }


@article{cui2022region,
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
  title={Parametric contrastive learning},
  author={Cui, Jiequan and Zhong, Zhisheng and Liu, Shu and Yu, Bei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={715--724},
  year={2021}
}

@inproceedings{zhong2021improving,
  title={Improving calibration for long-tailed recognition},
  author={Zhong, Zhisheng and Cui, Jiequan and Liu, Shu and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16489--16498},
  year={2021}
}

```

