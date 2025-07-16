# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:06:41 2025

@author: Jialin Zhang
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from utils import set_seed


def preprocessing_sc_intra(data_name, copy=True, highly_genes = 2000, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    
    read_path_mat = ''.join(['Your intra data matrix path',data_name,'/',data_name,'.csv'])
    read_path_lab = ''.join(['Your inter data label path',data_name,'/','Labels.csv'])
    X = pd.read_csv(read_path_mat, delimiter = ',',index_col=0).reset_index(drop=True) 
    label = pd.read_csv(read_path_lab) 
    label.columns = ['celltype']
    gene = X.columns
    adata = ad.AnnData(X, obs=label, dtype='float64')
    adata.var['gene'] = list(gene)
    
    if copy:
        adata = adata.copy()
        
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    
    assert 'n_counts' not in adata.obs, norm_error
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1000)
        sc.pp.filter_cells(adata, min_counts=3)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
        
    return adata

def write_text_matrix_sc_intra(data_name):
    
    adata = preprocessing_sc_intra(data_name) 
    
    matrix = pd.DataFrame(adata.X,columns=adata.var.gene)
    label = adata.obs.celltype

    matrix['label'] = label.values   
    
    save_path = ''.join(['Your processed intra data save path','processed_',data_name,'.csv'])
    
    matrix.to_csv(save_path, sep=',',
                  index=None,header=True)
    

def preprocessing_sc_inter(data_name, copy=True, highly_genes = 2000, size_factors=True, normalize_input=True, logtrans_input=True):
    
    read_path_mat = ''.join(['Your inter data matrix path',data_name,'/',data_name,'.csv'])
    read_path_lab = ''.join(['Your inter data label path',data_name,'/','Labels.csv'])
    X = pd.read_csv(read_path_mat, delimiter = ',',index_col=0).reset_index(drop=True) 
    label = pd.read_csv(read_path_lab) 
    label.columns = ['celltype']
    gene = X.columns
    adata = ad.AnnData(X, obs=label, dtype='float64')
    adata.var['gene'] = list(gene)
    if copy:
        adata = adata.copy()

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
        
    return adata

def write_text_matrix_sc_inter(data_name):
    

    adata = preprocessing_sc_inter(data_name) 
 
    matrix = pd.DataFrame(adata.X,columns=None)
    label = adata.obs.celltype
   
    matrix['label'] = label.values   
    
    save_path = ''.join(['Your processed inter data save path','processed_',data_name,'.csv'])
    
    matrix.to_csv(save_path, sep=',',
                  index=None,header=None)    
    
if __name__ == "__main__":
    
    set_seed(2025)
    
    dataset_sc_intra = ['AMB', 'Baron Human', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_sc_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
  
    for i in dataset_sc_intra:
        print(i)
        write_text_matrix_sc_intra(i)   
    for i in dataset_sc_inter:
        print(i)
        write_text_matrix_sc_inter(i)
    
    
    