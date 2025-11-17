# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:59:41 2025

@author: Jialin Zhang
"""

import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils import type_to_label_dict, convert_type_to_label, convert_label_to_type, set_seed

def train_val_test_split(mat,labels,test_size):
  test_size = test_size # The proportions of the validation and test sets.
  validation_size = test_size
  
  values = mat
  labels = labels
  
  X_train_val, X_test, y_train_val, y_test = train_test_split(values, labels, test_size=test_size, random_state=0, shuffle= True) 
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size/(1 - test_size), random_state=0, shuffle= True)
  
  return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_split_intra(name):
    adata_read_path = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/','processed_',name,'.h5ad'])
    adata = sc.read_h5ad(adata_read_path)
     
    row_num, col_num = adata.shape 
    mat = pd.DataFrame(adata.X,columns=adata.var.gene.values)
    label = pd.DataFrame({'label': adata.obs.celltype.values})
    label.columns = ['celltype']
    label_dict = type_to_label_dict(label['celltype'])  
    label_num = np.array(convert_type_to_label(label['celltype'], label_dict))
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(mat.values, label_num, 0.1)
    
    label_train = pd.DataFrame(y_train)
    label_train.columns = ['label']
    var_train = pd.DataFrame(adata.var.gene.values)
    var_train.columns = ['gene']
    
    label_val = pd.DataFrame(y_val)
    label_val.columns = ['label']
    var_val = pd.DataFrame(adata.var.gene.values)
    var_val.columns = ['gene']
    
    label_test = pd.DataFrame(y_test)
    label_test.columns = ['label']
    var_test = pd.DataFrame(adata.var.gene.values)
    var_test.columns = ['gene']
    
    
    train_data_save = ad.AnnData(X=X_train, obs=label_train, var=var_train)
    val_data_save = ad.AnnData(X=X_val, obs=label_val, var=var_val)
    test_data_save = ad.AnnData(X=X_test, obs=label_test, var=var_test)
    
    save_path_train = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/splited/', 'splited_',name,'/','Train_',name,'.h5ad'])
    save_path_val = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/splited/', 'splited_',name,'/','Val_',name,'.h5ad'])
    save_path_test = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/splited/', 'splited_',name,'/','Test_',name,'.h5ad'])
    save_path_dict = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/splited/', 'splited_',name,'/','dict_',name,'.json'])
    
    train_data_save.write(save_path_train,compression="gzip")
    val_data_save.write(save_path_val,compression="gzip")
    test_data_save.write(save_path_test,compression="gzip")
    
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)
    
def get_data_split_inter(name, idx):
    adata_read_path = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/','processed_',name,'.h5ad'])
    adata = sc.read_h5ad(adata_read_path)
    
    row_num, col_num = adata.shape 
    mat = pd.DataFrame(adata.X,columns=adata.var.gene.values)
    label = pd.DataFrame({'label': adata.obs.celltype.values})
    label.columns = ['celltype']
    label_dict = type_to_label_dict(label['celltype'])  
    label_num = np.array(convert_type_to_label(label['celltype'], label_dict))
    
    X_test, y_test = mat.values[0:idx,], label_num[0:idx,]
    X_train, y_train = mat.values[idx:-1,], label_num[idx:-1,]
    
    train_data_all = pd.DataFrame(X_train)
    train_data_all['label'] = y_train
    
    test_data_all = pd.DataFrame(X_test)
    test_data_all['label'] = y_test
    
    train_data_all = train_data_all.sample(frac=1).reset_index(drop=True)
    test_data_all = test_data_all.sample(frac=1).reset_index(drop=True)
    
    _,all_col_num = train_data_all.shape
    
    X_train = train_data_all.iloc[:,0:all_col_num-1].values
    y_train = train_data_all.iloc[:,-1].values
    
    X_test = test_data_all.iloc[:,0:all_col_num-1].values
    y_test = test_data_all.iloc[:,-1].values
    
    
    label_train = pd.DataFrame(y_train)
    label_train.columns = ['label']
    var_train = pd.DataFrame(adata.var.gene.values)
    var_train.columns = ['gene']
    
    label_test = pd.DataFrame(y_test)
    label_test.columns = ['label']
    var_test = pd.DataFrame(adata.var.gene.values)
    var_test.columns = ['gene']
    
    train_data_save = ad.AnnData(X=X_train, obs=label_train, var=var_train)
    test_data_save = ad.AnnData(X=X_test, obs=label_test, var=var_test)
    
    save_path_train = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/splited/', 'splited_',name,'/','Train_',name,'.h5ad'])
    save_path_test = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/splited/', 'splited_',name,'/','Test_',name,'.h5ad'])
    save_path_dict = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/splited/', 'splited_',name,'/','dict_',name,'.json'])
    
    train_data_save.write(save_path_train,compression="gzip")
    test_data_save.write(save_path_test,compression="gzip")
    
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)
    
def get_data_split_crobatch(name):
    adata_read_path = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/','processed_',name,'.h5ad'])
    adata = sc.read_h5ad(adata_read_path)

    row_num, col_num = adata.shape 
    mat = pd.DataFrame(adata.X,columns=adata.var.gene.values,index=adata.obs_names)
    label = pd.DataFrame({'label': adata.obs.celltype.values},index=adata.obs_names)
    label.columns = ['celltype']
    
    label_dict = type_to_label_dict(label['celltype'])  
    
    X_train, y_train = mat[mat.index=='data1'], label[label.index=='data1']
    
    X_test, y_test = mat[mat.index=='data2'], label[label.index=='data2']
    
    y_train = np.array(convert_type_to_label(y_train['celltype'], label_dict))

    y_test = np.array(convert_type_to_label(y_test['celltype'], label_dict))
    
    
    train_data_all = pd.DataFrame(X_train)
    train_data_all['label'] = y_train
    
    test_data_all = pd.DataFrame(X_test)
    test_data_all['label'] = y_test
    
    train_data_all = train_data_all.sample(frac=1).reset_index(drop=True)
    test_data_all = test_data_all.sample(frac=1).reset_index(drop=True)
    
    _,all_col_num = train_data_all.shape
    
    X_train = train_data_all.iloc[:,0:all_col_num-1].values
    y_train = train_data_all.iloc[:,-1].values
    
    X_test = test_data_all.iloc[:,0:all_col_num-1].values
    y_test = test_data_all.iloc[:,-1].values
    
    
    label_train = pd.DataFrame(y_train)
    label_train.columns = ['label']
    var_train = pd.DataFrame(adata.var.gene.values)
    var_train.columns = ['gene']
    
    label_test = pd.DataFrame(y_test)
    label_test.columns = ['label']
    var_test = pd.DataFrame(adata.var.gene.values)
    var_test.columns = ['gene']
    
    train_data_save = ad.AnnData(X=X_train, obs=label_train, var=var_train)
    test_data_save = ad.AnnData(X=X_test, obs=label_test, var=var_test)
    
    save_path_train = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/splited/', 'splited_',name,'/','Train_',name,'.h5ad'])
    save_path_test = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/splited/', 'splited_',name,'/','Test_',name,'.h5ad'])
    save_path_dict = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/splited/', 'splited_',name,'/','dict_',name,'.json'])
    
    train_data_save.write(save_path_train,compression="gzip")
    test_data_save.write(save_path_test,compression="gzip")
    
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)


if __name__ == "__main__":
    set_seed(2024)
    
    dataset_intra = ['AMB', 'Baron Human','Segerstolpe', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    dataset_crobat = ['Dendritic', 'Retina(5)', 'Retina(19)']
    
    idx_list = [6444, 3222, 3222, 3222, 3176]
    
    for i, name in enumerate(dataset_intra):
        print('{}'.format(i))
        get_data_split_intra(name)    
    
    for i, name in enumerate(dataset_inter):
        print('{}'.format(i))
        get_data_split_inter(name, idx_list[i])    
        
    for i, name in enumerate(dataset_crobat):
        print('{}'.format(i))
        get_data_split_crobatch(name)       
            
        
        
        
        
        
        

        
