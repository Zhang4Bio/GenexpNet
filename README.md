# scAnnoNet: an attention-based deep-learning model for robust cell-type annotation across complex and cross-dataset single-cell RNA sequencing data

# Introduction

scAnnoNet is a novel two-stage ensemble model that combines statistical cell type-specific gene selection and attention-based deep learning. By introducing discriminative and collinearity feature losses, GenexpNet effectively captures informative and nonredundant signals, demonstrating superior robustness and generalizability across complex, cross-platform, and cross-batch scenarios.

## Requirement

```python
Python 3.9.19
torch 1.13.0
numpy 1.26.4
pandas 2.2.2
scipy 1.13.1
sklearn 1.5.1
Scanpy 1.10.2
anndata 0.10.8
json
random
numba 0.60.0
```

## Usage

step 1. Generate a cell type-specific gene matrix

python  SP\_gene\_matrix.py&#x20;

step 2. Splitting data for training, validation, and testing

python  datasplit.py

step 3. Training deep learning models and testing

python  main\_GenexpNet.py

## Set parameters

A summary of the parameters in the deep learning model (main\_GenexpNet.py) that change according to the dataset is provided.

| Datasets     | batch\_size | lr\_cla | w1    | oversampling |
| :----------- | :---------- | :------ | :---- | :----------- |
| AMB          | 256         | 0.01    | 1000  | True         |
| Baron Human  | 256         | 0.0005  | 1000  | True         |
| Segerstolpe  | 256         | 0.0001  | 1000  | False        |
| TM           | 256         | 0.01    | 1000  | True         |
| Zheng 68K    | 256         | 0.0003  | 10000 | True         |
| Zheng sorted | 256         | 0.0003  | 10000 | False        |
| 10Xv2        | 256         | 0.0001  | 1000  | True         |
| 10Xv3        | 256         | 0.00008 | 1000  | True         |
| Drop-Seq     | 256         | 0.00005 | 1000  | True         |
| inDrop       | 256         | 0.0001  | 1000  | True         |
| Seq-Well     | 256         | 0.0002  | 1000  | True         |
| Dendritic    | 128         | 0.0001  | 1000  | True         |
| Retina(5)    | 128         | 0.0003  | 1000  | True         |
| Retina(19)   | 128         | 0.0003  | 1000  | True         |

## processsed Data

The filtered datasets analyzed during the current study can be downloaded from Zenodo ([https://zenodo.org/records/17548897]()).











