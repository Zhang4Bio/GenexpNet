# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:42:02 2025

@author: gaga6
"""



import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import t
from numba import njit, prange
from utils import set_seed

def compute_log2fc_matrix(avg_expr, pseudocount=1e-6):
    """
    Input: avg_expr is a c × m DataFrame (one cell type per row, one gene per column)
    Output: c × m log2FC matrix, log2 fold change for each cell type vs. all other types
    """
    
    cell_types = avg_expr.index
    log2fc_matrix = pd.DataFrame(index=cell_types, columns=avg_expr.columns)

    for cell_type in cell_types:
        expr_target = avg_expr.loc[cell_type] + pseudocount
        expr_others = avg_expr.drop(index=cell_type).mean(axis=0) + pseudocount
        log2fc = np.log2(expr_target / expr_others)
        log2fc_matrix.loc[cell_type] = log2fc

    return log2fc_matrix.astype(float)    

@njit(parallel=True)
def compute_mean_std(A):
    n, p = A.shape
    mean = np.empty(p)
    std = np.empty(p)
    for i in prange(p):
        col = A[:, i]
        m = 0.0
        for j in range(n):
            m += col[j]
        m /= n
        mean[i] = m

        s = 0.0
        for j in range(n):
            s += (col[j] - m) ** 2
        std[i] = np.sqrt(s / (n - 1))
    return mean, std

@njit(parallel=True)
def fast_pearsonr_only(A, B):
    n, p1 = A.shape
    _, p2 = B.shape

    A_mean, A_std = compute_mean_std(A)
    B_mean, B_std = compute_mean_std(B)

    corr = np.empty((p1, p2))

    for i in prange(p1):
        for j in prange(p2):
            num = 0.0
            for k in range(n):
                num += (A[k, i] - A_mean[i]) * (B[k, j] - B_mean[j])
            denom = (n - 1) * A_std[i] * B_std[j]
            corr[i, j] = num / denom
    return corr


def compute_p_values(corr, n):
    df = n - 2
    t_stat = corr * np.sqrt(df / (1 - corr ** 2))
    pval = 2 * t.sf(np.abs(t_stat), df)
    return pval


def prepare_data(data_name,datatype, copy=True, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, highly_genes = 4000, FCthreshold = 1, p_thresh=0.05, top_n=20):
    read_path_mat = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/',data_name,'.csv'])
    read_path_lab = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/','Labels.csv'])
    X = pd.read_csv(read_path_mat, delimiter = ',',index_col=0).reset_index(drop=True) 
    label = pd.read_csv(read_path_lab) 
    label.columns = ['celltype']
    gene = X.columns
    adata = ad.AnnData(X, obs=label, dtype='float64')
    adata.var['gene'] = list(gene)
    
    # -----------Data preprocessing------------------------
    if copy:
        adata = adata.copy()   
    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= 10].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()
    
        
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
    # ------------------Calculate log2FC values and identify cell type–specific genes------------------------     
    matrix = pd.DataFrame(adata.X,columns=adata.var.gene.values)
    matrix['cell_type'] = adata.obs.celltype.values
    
    mean_matrix = matrix.groupby('cell_type').mean() # Compute the mean expression values of samples grouped by cell type.
    
    FC_mat = compute_log2fc_matrix(mean_matrix, pseudocount=1e-6)  #Compute the log2FC matrix using the mean expression matrix.
    
    top_markers = {}
    for cell_type in FC_mat.index:
        
        filtered_genes = FC_mat.loc[cell_type][FC_mat.loc[cell_type] > FCthreshold]
        
        top_genes = filtered_genes.sort_values(ascending=False).head(top_n)
           
        top_markers[cell_type] = top_genes
        
    all_gene = list(adata.var.gene.values)
    
    temp_df = []
    for key, value in top_markers.items():
        for v in value.index:
            temp_df.append([key, v])
            
    all_marker = pd.DataFrame(temp_df, columns=['celltype', 'marker_gene'])
    
    all_marker.drop_duplicates(subset='marker_gene')
    
    all_marker = list(all_marker['marker_gene'].values)
    
    all_other_gene = [gene for gene in all_gene if gene not in all_marker]
    
    all_gene = np.array(all_gene)
    
    if normalize_input:

        sc.pp.scale(adata)
    
    pre_X = adata.X
    if isinstance(pre_X, np.ndarray):  
        pre_X = pd.DataFrame(pre_X, index=adata.obs.celltype.values, columns=adata.var.gene.values)     
        
    selected_genes_by_type = {}
    
    results = []
    
    for cell_type, marker_genes in top_markers.items():
        print(cell_type)
        marker_genes = marker_genes.index
        selected_genes = []
        
        X_marker = pre_X.loc[:, marker_genes].values + 1e-5
        X_other = pre_X.loc[:, all_other_gene].values + 1e-5
        
        n_cells = X_marker.shape[0]
        
        corr_mat = fast_pearsonr_only(X_other, X_marker)
        pval_mat = compute_p_values(corr_mat, n_cells)
        
        # Traversing and filtering high co-expression pairs
        for i in range(corr_mat.shape[0]):
            
            for j in range(corr_mat.shape[1]):

                p = pval_mat[i, j]
                
                if p < p_thresh:
                    results.append({
                        'celltype': cell_type,
                        'marker_gene': marker_genes[j],
                        'coexpressed_gene': all_other_gene[i],
                        'pval': p
                    })
                    selected_genes.append(all_other_gene[i])
        selected_genes_by_type[cell_type] = list(set(selected_genes))
         
    co_gene = list(set(sum(selected_genes_by_type.values(), [])))
    sp_co_df = pd.DataFrame(results)
    
    feature_list = co_gene + all_marker
    
    feature_list = list(set(feature_list))
    
    filt_adata = adata[:,adata.var.gene.isin(feature_list)]
    
    feature_list = pd.DataFrame(feature_list)
    
    return sp_co_df, feature_list, filt_adata

def prepare_data_crobatch(data_name, datatype, copy=True, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, highly_genes = 4000, FCthreshold = 1, p_thresh=0.05, top_n=20):
    read_path_mat1 = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/','batch1.csv'])
    read_path_lab1 = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/','batch1label.csv'])
    
    read_path_mat2 = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/','batch2.csv'])
    read_path_lab2 = ''.join(['E:/My_project/GenexpNet/datasets/',datatype,'/',data_name,'/','batch2label.csv'])
    
    X1 = pd.read_csv(read_path_mat1, delimiter = ',',index_col=0).reset_index(drop=True) 
    X2 = pd.read_csv(read_path_mat2, delimiter = ',',index_col=0).reset_index(drop=True) 
    
    X1.index = ['data1'] * len(X1)
    X2.index = ['data2'] * len(X2)
    
    common_columns = X1.columns.intersection(X2.columns)
    
    X1 = X1.loc[:,common_columns]
    X2 = X2.loc[:,common_columns]
    
    merged_X = pd.concat([X1, X2], axis=0, ignore_index=False)
    
    gene = merged_X.columns
    
    label1 = pd.read_csv(read_path_lab1, index_col=0) 
    label1.columns = ['celltype']
    label1.index = ['data1'] * len(label1)
      
    label2 = pd.read_csv(read_path_lab2, index_col=0) 
    label2.columns = ['celltype']
    label2.index = ['data2'] * len(label2)
    
    merged_label = pd.concat([label1, label2], axis=0, ignore_index=False)
    
    adata = ad.AnnData(merged_X, obs=merged_label, dtype='float64')
    
    adata.var['gene'] = list(gene)
    
    if copy:
        adata = adata.copy()
        
    celltype_counts = adata.obs['celltype'].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= 100].index
    adata = adata[adata.obs['celltype'].isin(valid_celltypes)].copy()
        
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
        
    matrix = pd.DataFrame(adata.X,columns=adata.var.gene.values)
    matrix['cell_type'] = adata.obs.celltype.values
    
    mean_matrix = matrix.groupby('cell_type').mean()
    
    FC_mat = compute_log2fc_matrix(mean_matrix, pseudocount=1e-6)
    
    top_markers = {}
    for cell_type in FC_mat.index:
        
        filtered_genes = FC_mat.loc[cell_type][FC_mat.loc[cell_type] > FCthreshold]
        
        top_genes = filtered_genes.sort_values(ascending=False).head(top_n)
           
        top_markers[cell_type] = top_genes
        
        
    all_gene = list(adata.var.gene.values)
    
    temp_df = []
    for key, value in top_markers.items():
        for v in value.index:
            temp_df.append([key, v])
            
    all_marker = pd.DataFrame(temp_df, columns=['celltype', 'marker_gene'])
    
    all_marker.drop_duplicates(subset='marker_gene')
    
    all_marker = list(all_marker['marker_gene'].values)
    
    all_other_gene = [gene for gene in all_gene if gene not in all_marker]
    
    all_gene = np.array(all_gene)
    
    if normalize_input:
        sc.pp.scale(adata)
         
    pre_X = adata.X
    if isinstance(pre_X, np.ndarray):  
        pre_X = pd.DataFrame(pre_X, index=adata.obs.celltype.values, columns=adata.var.gene.values)     
        

    selected_genes_by_type = {}
    
    results = []
    
    for cell_type, marker_genes in top_markers.items():
        print(cell_type)
        marker_genes = marker_genes.index
        selected_genes = []
        

        X_marker = pre_X.loc[:, marker_genes].values + 1e-5
        X_other = pre_X.loc[:, all_other_gene].values + 1e-5
        
        n_cells = X_marker.shape[0]
        
        corr_mat = fast_pearsonr_only(X_other, X_marker)
        pval_mat = compute_p_values(corr_mat, n_cells)
        
        #Traversing and filtering high co-expression pairs
        for i in range(corr_mat.shape[0]):
            
            for j in range(corr_mat.shape[1]):
                
                p = pval_mat[i, j]
                
                if p < p_thresh:
                    results.append({
                        'celltype': cell_type,
                        'marker_gene': marker_genes[j],
                        'coexpressed_gene': all_other_gene[i],
                        'pval': p
                    })
                    selected_genes.append(all_other_gene[i])
        selected_genes_by_type[cell_type] = list(set(selected_genes))
         
    co_gene = list(set(sum(selected_genes_by_type.values(), [])))
    sp_co_df = pd.DataFrame(results)
    
    feature_list = co_gene + all_marker
    
    feature_list = list(set(feature_list))
    
    filt_adata = adata[:,adata.var.gene.isin(feature_list)]
    
    feature_list = pd.DataFrame(feature_list)
    
    return sp_co_df, feature_list, filt_adata


def write_preprocessed_matrix_sc_intra(data_name):
    
    gene_df, feature_list, adata = prepare_data(data_name,'intra') 
    
    df_save_path = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/','gene_list_',data_name,'.csv'])
    feature_save_path = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/','feature_list_',data_name,'.csv'])
    adata_save_path = ''.join(['E:/My_project/GenexpNet/datasets/intra/preprocessed/','processed_',data_name,'.h5ad'])
    gene_df.to_csv(df_save_path, sep=',',
                  index=None,header=True)   
    feature_list.to_csv(feature_save_path, sep=',',
                  index=None,header=True)    
    adata.write(adata_save_path)
    
    
    
def write_preprocessed_matrix_sc_inter(data_name):
    
    gene_df, feature_list, adata = prepare_data(data_name,'inter') 
    
    df_save_path = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/','gene_list_',data_name,'.csv'])
    feature_save_path = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/','feature_list_',data_name,'.csv'])
    adata_save_path = ''.join(['E:/My_project/GenexpNet/datasets/inter/preprocessed/','processed_',data_name,'.h5ad'])
    gene_df.to_csv(df_save_path, sep=',',
                  index=None,header=True)    
    feature_list.to_csv(feature_save_path, sep=',',
                  index=None,header=True)    
    adata.write(adata_save_path)
    
    
def write_preprocessed_matrix_sc_crobatch(data_name):
    
    gene_df, feature_list, adata = prepare_data_crobatch(data_name,'cross_batch') 
    
    df_save_path = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/','gene_list_',data_name,'.csv'])
    feature_save_path = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/','feature_list_',data_name,'.csv'])
    adata_save_path = ''.join(['E:/My_project/GenexpNet/datasets/cross_batch/preprocessed/','processed_',data_name,'.h5ad'])
    gene_df.to_csv(df_save_path, sep=',',
                  index=None,header=True)    
    feature_list.to_csv(feature_save_path, sep=',',
                  index=None,header=True)    
    adata.write(adata_save_path)
    
if __name__ == "__main__":
    
    set_seed(2025)
    
    dataset_sc_intra = ['AMB', 'Baron Human','Segerstolpe', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_sc_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    dataset_sc_crosbatch = ['Dendritic', 'Retina(5)', 'Retina(19)']
    
    
    for i in dataset_sc_intra:
        print(i)
        write_preprocessed_matrix_sc_intra(i)   
        
    for i in dataset_sc_inter:
        print(i)
        write_preprocessed_matrix_sc_inter(i)
       
    for i in dataset_sc_crosbatch:
        print(i)
        write_preprocessed_matrix_sc_crobatch(i)
        
        
    