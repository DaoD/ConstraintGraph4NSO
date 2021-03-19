# Neural Sentence Ordering Based on Constraint Graphs

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- We upload data and source code for the second phase!

## Abstract
This repository contains the source code and datasets for the AAAI 2021 paper Neural Sentence Ordering Based on Constraint Graphs by Zhu and Zhou et al. <br>

Sentence ordering is a subtask of text coherence modeling, aiming at arranging a list of sentences in the correct order. Based on the observation that sentence order at different distances may rely on different types of information, we devise a new approach based on multi-granular orders between sentences. These orders from multiple constraint graphs, which are then encoded by GINs and fused into sentence representations. Finally, sentence order is determined using the order-enhanced sentence representations. Our experiments on five benchmark datasets show that our method outperforms all the existing baselines significantly, achieving new state-of-the-art performance. The results confirm the advantage of considering multiple types of order information and using graph neural networks to integrate sentence content and order information for the task.

Authors: Yutao Zhu, Kun Zhou, Jian-Yun Nie, Shengchao Liu, Zhicheng Dou

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.5 <br>
- Pytorch 1.3.1 (with GPU support)<br>

## Usage
Second phase
- Download the [data](https://drive.google.com/file/d/13f6PEZZbn_KnFk53au9s2CbAVFQdTRnC/view?usp=sharing), and unzip it to "data" directory.
- Train the model
```
python3 run.py --task nips
```

## Citations
If you use the code and datasets, please cite the following paper:  
```
@article{DBLP:journals/corr/abs-2101-11178,
  author    = {Yutao Zhu and
               Kun Zhou and
               Jian{-}Yun Nie and
               Shengchao Liu and
               Zhicheng Dou},
  title     = {Neural Sentence Ordering Based on Constraint Graphs},
  booktitle = {Proceedings of the Thirty-Fifth {AAAI} Conference on Artificial Intelligence,
               February 2-9, 2021, Virtual Conference},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```
