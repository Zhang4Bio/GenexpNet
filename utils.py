# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:02:16 2025

@author: gaga6
"""


import os

import random
import numpy as np

import colorcet as cc

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import umap as umap


def datascaler(input):
    sc = MinMaxScaler(feature_range=(0, 10))     
    data = input
    datarow, datacol = data.shape
    gene_name = data.columns.values.tolist()[:datacol-1]
    sample_name = data.index.values.tolist()
    X = data.iloc[:,range(datacol)].values
    X[:,0:datacol-1] = sc.fit_transform(X[:,0:datacol-1])
    n_samples,_ = X.shape    # number of samples and number of features
    return X,n_samples,gene_name,sample_name

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

def data_split(data,test_size,seed):
  data,n_samples,gene_name,sample_name = datascaler(data)
  data_dim = data.shape[1]
  values = data[:,0:data_dim-1]
  labels = data[:,-1]
  labels = encode_onehot(labels)
  labels = np.argmax(labels,-1)
  X_train, X_test, y_train, y_test = train_test_split(values, labels, test_size=test_size, random_state=0)
  tensor_TrainValues = torch.FloatTensor(X_train).float()
  tensor_TestValues = torch.FloatTensor(X_test).float()
  tensor_TrainLabels = torch.FloatTensor(y_train).float()
  tensor_TestLabels = torch.FloatTensor(y_test).float()
  return tensor_TrainValues,tensor_TestValues,tensor_TrainLabels,tensor_TestLabels,gene_name

def train_val_test_split(data,test_size):
  test_size = test_size
  validation_size = test_size  
  
  data,n_samples,gene_name,sample_name = datascaler(data)
  data_dim = data.shape[1]
  values = data[:,0:data_dim-1].astype(np.float32)
  labels = data[:,-1]
  labels = encode_onehot(labels)
  labels = np.argmax(labels,-1).astype(np.int16)
  
  X_train_val, X_test, y_train_val, y_test = train_test_split(values, labels, test_size=test_size, random_state=0) 
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size/(1 - test_size), random_state=0)
  
  tensor_TrainValues = torch.FloatTensor(X_train).float()
  tensor_ValValues = torch.FloatTensor(X_val).float()
  tensor_TestValues = torch.FloatTensor(X_test).float()
  tensor_TrainLabels = torch.FloatTensor(y_train).float()
  tensor_ValLabels = torch.FloatTensor(y_val).float()
  tensor_TestLabels = torch.FloatTensor(y_test).float()
  
  return tensor_TrainValues,tensor_ValValues,tensor_TestValues,tensor_TrainLabels,tensor_ValLabels,tensor_TestLabels,gene_name

def train_val_test_split_with_orilabel(data, test_size, dicts):
  test_size = test_size
  validation_size = test_size  
  
  data,n_samples,gene_name,sample_name = datascaler(data)
  data_dim = data.shape[1]
  values = data[:,0:data_dim-1].astype(np.float32)
  labels = data[:,-1]
  
  labels = convert_type_to_label(labels, dicts)
  
  X_train_val, X_test, y_train_val, y_test = train_test_split(values, labels, test_size=test_size, random_state=0) 
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size/(1 - test_size), random_state=0)
  
  tensor_TrainValues = torch.FloatTensor(X_train).float()
  tensor_ValValues = torch.FloatTensor(X_val).float()
  tensor_TestValues = torch.FloatTensor(X_test).float()
  tensor_TrainLabels = torch.FloatTensor(y_train).float()
  tensor_ValLabels = torch.FloatTensor(y_val).float()
  tensor_TestLabels = torch.FloatTensor(y_test).float()
  
  return tensor_TrainValues,tensor_ValValues,tensor_TestValues,tensor_TrainLabels,tensor_ValLabels,tensor_TestLabels,gene_name

def data_direct(data):
  data,n_samples,gene_name,sample_name = datascaler(data)
  np.random.shuffle(data)
  data_dim = data.shape[1]
  values = data[:,0:data_dim-1].astype(np.float32)
  labels = data[:,-1]
  labels = encode_onehot(labels)
  labels = np.argmax(labels,-1).astype(np.float32)
  tensor_TrainValues = torch.FloatTensor(values).float()
  tensor_TrainLabels = torch.FloatTensor(labels).float()
  return tensor_TrainValues,tensor_TrainLabels,gene_name

def data_direct_with_orilabel(data):
  data,n_samples,gene_name,sample_name = datascaler(data)
  np.random.shuffle(data)
  data_dim = data.shape[1]
  values = data[:,0:data_dim-1].astype(np.float32)
  labels = data[:,-1]
  
  dicts = type_to_label_dict(labels)
  value_label = convert_type_to_label(labels, dicts)
  
  tensor_TrainValues = torch.FloatTensor(values).float()
  tensor_TrainLabels = torch.FloatTensor(value_label).float()
  return tensor_TrainValues,tensor_TrainLabels,gene_name,dicts


def dis_loss(attention,fisher,att_loss):
    loss = att_loss(attention.to(torch.float32),fisher.to(torch.float32))
    return loss

def cfc_loss(W):
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
    
    X = X.values
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
    
    
    plt.figure(figsize=(16, 10))
    sns.set()
    sns.set_style('ticks')
    pal = sns.color_palette(cc.glasbey,len(np.unique(y)))
    sns.scatterplot(x="projected1", y="projected2",hue="labels",legend='full', data=df, palette=pal)
    #plt.title(name)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0), markerscale=2, fontsize=20)
    plt.xlabel('UMAP 1', fontsize=20)
    plt.ylabel('UMAP 2', fontsize=20)
    #sns.set_palette("husl")
    plt.tight_layout()
    plt.savefig('E:/My_project/GenexpNet/Experiments Figure/UMAP_input/' + name + '.jpg',dpi=300)
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
