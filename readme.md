# Neural Sentence Ordering Based on Constraint Graphs

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- 2021.3.19: We upload data and source code!

## Abstract
This repository contains the source code and datasets for the AAAI 2021 paper [Neural Sentence Ordering Based on Constraint Graphs](https://arxiv.org/pdf/2101.11178.pdf) by Zhu et al. <br>

Sentence ordering is a subtask of text coherence modeling, aiming at arranging a list of sentences in the correct order. Based on the observation that sentence order at different distances may rely on different types of information, we devise a new approach based on multi-granular orders between sentences. These orders from multiple constraint graphs, which are then encoded by GINs and fused into sentence representations. Finally, sentence order is determined using the order-enhanced sentence representations. Our experiments on five benchmark datasets show that our method outperforms all the existing baselines significantly, achieving new state-of-the-art performance. The results confirm the advantage of considering multiple types of order information and using graph neural networks to integrate sentence content and order information for the task.

Authors: Yutao Zhu, Kun Zhou, Jian-Yun Nie, Shengchao Liu, Zhicheng Dou

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.5 <br>
- Pytorch 1.3.1 (with GPU support)<br>

## Usage
### First phase
- Download the data files, and uzip them to "data" directory
   - For NeurIPS, AAN and NSF dataset, please refer to the paper "Sentence Ordering and Coherence Modeling using Recurrent Neural Networks" and require the dataset from the author.
   - For Sind dataset, please refer to https://visionandlanguage.net/VIST/dataset.html and [download](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz) it.
   - For ROCStory dataset, please refer to https://www.cs.rochester.edu/nlp/rocstories/.
- Prepare the data
```
python prepare_data.py --data_dir ./data/nips/ --out_dir ./data/nips_data/ --task_name nips
python prepare_data.py --data_dir ./data/aan/ --out_dir ./data/aan_data/ --task_name aan
python prepare_data.py --data_dir ./data/nsf/ --out_dir ./data/nsf_data/ --task_name nsf
python prepare_data.py --data_dir ./data/sind/ --out_dir ./data/sind_data/ --task_name sind
python prepare_data.py --data_dir ./data/roc/ --out_dir ./data/roc_data/ --task_name roc
```
- Train the model (using the nips data as an example)
```
python model.py --data_dir ./data/nips_data/ --output_dir ./trained_models/nips_bert/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 16 --window_size 5 --overwrite_output_dir
```
- Do the inference (using the nips data as an example)
```
python model.py --data_dir ./data/nips_data/ --output_dir ./trained_models/nips_bert/checkpoint-X/ --do_test --per_gpu_eval_batch_size 64
```
Note: (checkpoint-X) should be replaced by the last checkpoint obtained in training.

### Second phase
- Download the [data](https://drive.google.com/file/d/13f6PEZZbn_KnFk53au9s2CbAVFQdTRnC/view?usp=sharing), and unzip it to "data" directory.
Note: preparing input file from the results obtained in the first phase
```
python3 prepare_data.py
```
- Train the model
```
python3 run.py --task nips
```

## Citations
If you use the code and datasets, please cite the following paper:  
```
@arinproceedings{ZhuZNLD21,
  author    = {Yutao Zhu and
               Kun Zhou and
               Jian{-}Yun Nie and
               Shengchao Liu and
               Zhicheng Dou},
  title     = {Neural Sentence Ordering Based on Constraint Graphs},
  booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2021, Thirty-Third Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
               2021},
  pages     = {14656--14664},
  publisher = {{AAAI} Press},
  year      = {2021},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/17722}
}
```
