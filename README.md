<h1 align="center">Evolutionary Generalized Zero-Shot Learning</h1>

<p align="center">
    <a href="https://www.ijcai.org/proceedings/2024/70"><img src="https://img.shields.io/badge/IJCAI.2024-10.24963-blue" alt="Paper"></a>
    <a href="https://arxiv.org/pdf/2211.13174"><img src="https://img.shields.io/badge/arXiv-2211.13174-b31b1b.svg" alt="arXiv"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="#License: Apache2.0"></a>
</p>


This repository hosts the PyTorch implementation of Evolutionary Generalized Zero-Shot Learning (EGZSL), designed for unsupervised continual evolution during deployment.

## Get Started

### Dependencies

This project requires the following:
- Python >= 3.7
- PyTorch >= 1.0.1
- NumPy (version as per compatibility)

### Prepare Dataset

Download the dataset from [this link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) and place it in the `data` folder. Ensure the following structure:
```
checkpoints
data
├── AWA2
├── CUB
├── APY
scripts
egzsl.py
model.py
util.py
```

### Train and Test
Train EGZSL with a single GPU, *e.g.*:
```
scripts/AWA2.sh
```

## Bibtex
Please consider citing our paper if it is helpful for your research:
```BibTeX
  @inproceedings{ijcai2024p70,
  title     = {Evolutionary Generalized Zero-Shot Learning},
  author    = {Chen, Dubing and Jiang, Chenyi and Zhang, Haofeng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {632--640},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/70},
  url       = {https://doi.org/10.24963/ijcai.2024/70},
}
```

