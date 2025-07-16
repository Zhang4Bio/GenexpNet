# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:02:16 2025

@author: Jialin Zhang
"""


import os

import random
import numpy as np

import colorcet as cc

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import umap as umap


def get_keys_from_value(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]

def type_to_label_dict(types):
    # input -- types
    # output -- type_to_label dictionary
    dicts = {}
    all_type = list(set(types))
    for i in range(len(all_type)):
        dicts[all_type[i]] = i
    return dicts

# Convert types to labels
def convert_type_to_label(types, type_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    type_list = list(types)
    labels = list()
    for t in type_list:
        labels.append(type_label_dict[t])
    return labels

def convert_label_to_type(types, type_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    type_list = list(types)
    labels = list()
    for t in type_list:
        keys = get_keys_from_value(type_label_dict, t)
        labels.append(keys[0])
    return labels


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def Weight_loss(attention,fisher,att_loss):
    loss = att_loss(attention.to(torch.float32),fisher.to(torch.float32))
    return loss

def MF_loss(x):
    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
    n_row, n_col = x.shape
      
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-5)
   
    cos_sim = (torch.mm(x, x.T)).pow(2)*(torch.ones(n_row,n_row).to(device)-torch.eye(n_row).to(device))

    return 0.5*torch.sum(cos_sim) * (1/(n_row**2-len(torch.diag(cos_sim))))

def MF_dec_loss(W):
    device = torch.device("cuda") if W.is_cuda else torch.device("cpu")

    n_row, n_col = W.shape

    WTW = torch.mm(W.T, W)
    
    loss = torch.abs(torch.mm(torch.ones(n_col,n_col).to(device), WTW).trace() - WTW.trace()) * (1/(n_row**2-len(torch.diag(WTW))))
    
    return loss

def sequence_scale(x):
  scaler = MinMaxScaler(feature_range=(0, 1))
  score= scaler.fit_transform(x)
  return score



def plot_datapoint_umap(X, y, name):
    
    X = X
    y = y
    sns.set(font_scale=1.5)
    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(X)
    
    PCA_data = {
        'projected1': projected[:, 0], 
        'projected2': projected[:, 1],  
        'labels': y
        }
    df = pd.DataFrame(PCA_data)
    
    
    plt.figure(figsize=(10, 8))
    sns.set()
    sns.set_style('ticks')
    pal = sns.color_palette(cc.glasbey,len(np.unique(y)))

    sns.scatterplot(x="projected1", y="projected2",hue="labels",legend='auto', data=df, palette=pal)

    plt.xlabel("")
    plt.ylabel("")
    plt.legend().set_visible(False)

    plt.tight_layout()
    plt.savefig('E:/My_project/GenexpNet/Experiments Figure/UMAP_input/' + name + '.jpg',dpi=300)
    plt.show()
    
def plot_datapoint_umap_with_label(X, y, name):
    
    X = X
    y = y
    sns.set(font_scale=1.5)
    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(X)
    #projected = pca.fit_transform(X)
    
    PCA_data = {
        'projected1': projected[:, 0], 
        'projected2': projected[:, 1],  
        'labels': y
        }
    df = pd.DataFrame(PCA_data)
    
    
    plt.figure(figsize=(20, 16))
    sns.set()
    sns.set_style('ticks')
    pal = sns.color_palette(cc.glasbey,len(np.unique(y)))
    sns.scatterplot(x="projected1", y="projected2",hue="labels",legend='auto', data=df, palette=pal)
    #plt.title(name)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), markerscale=2, fontsize=20)
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig('your path' + name + '.jpg',dpi=300)
    plt.show()
    
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
