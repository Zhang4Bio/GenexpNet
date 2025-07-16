# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:59:41 2025

@author: Jialin Zhang
"""


import pandas as pd

import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils import type_to_label_dict, convert_type_to_label, convert_label_to_type, set_seed


def train_val_test_split(mat,labels,test_size):
  test_size = test_size
  validation_size = test_size
  
  values = mat
  labels = labels
  
  X_train_val, X_test, y_train_val, y_test = train_test_split(values, labels, test_size=test_size, random_state=0, shuffle= True) 
  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size/(1 - test_size), random_state=0, shuffle= True)
  
  
  return X_train, X_val, X_test, y_train, y_val, y_test

def get_data_split_intra(name):
    read_path = ''.join(['Your intra preprocessed data path','processed_',name,'.csv'])
    X = pd.read_csv(read_path, delimiter = ',',header=None)
    row_num, col_num = X.shape   
    mat = X.iloc[:,0:col_num-1]
    label = pd.DataFrame({'label': X.iloc[:,col_num-1]})
    label.columns = ['celltype']
    
    label_dict = type_to_label_dict(label['celltype'])  
    label_num = np.array(convert_type_to_label(label['celltype'], label_dict))
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(mat.values, label_num, 0.1)
    
    save_path_train = ''.join(['Your intra train data path', 'splited_',name,'/','Train_',name,'.csv'])
    save_path_val = ''.join(['Your intra val data path', 'splited_',name,'/','Val_',name,'.csv'])
    save_path_test = ''.join(['Your intra test data path', 'splited_',name,'/','Test_',name,'.csv'])
    save_path_dict = ''.join(['Your intra test dict path', 'splited_',name,'/','dict_',name,'.json'])
    
    train_data_save = pd.DataFrame(X_train)
    train_data_save['label'] = y_train
    
    val_data_save = pd.DataFrame(X_val)
    val_data_save['label'] = y_val

    test_data_save = pd.DataFrame(X_test)
    test_data_save['label'] = y_test
    
    
    train_data_save.to_csv(save_path_train, index=False, header=False)
    val_data_save.to_csv(save_path_val, index=False, header=False)
    test_data_save.to_csv(save_path_test, index=False, header=False)
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)
        

    
def get_data_split_inter(name, idx):
    read_path = ''.join(['Your inter preprocessed data path','processed_',name,'.csv'])
    X = pd.read_csv(read_path, delimiter = ',',header=None)
    row_num, col_num = X.shape   
    mat = X.iloc[:,0:col_num-1]
    label = pd.DataFrame({'label': X.iloc[:,col_num-1]})
    label.columns = ['celltype']
    
    label_dict = type_to_label_dict(label['celltype'])  
    label_num = np.array(convert_type_to_label(label['celltype'], label_dict))
    
    X_train, y_train = mat.values[0:idx,], label_num[0:idx,]
    X_test, y_test = mat.values[idx:-1,], label_num[idx:-1,]
    
    
    save_path_train = ''.join(['Your inter train data path', 'splited_',name,'/','Train_',name,'.csv'])
    save_path_test = ''.join(['Your inter test data path', 'splited_',name,'/','Test_',name,'.csv'])
    save_path_dict = ''.join(['Your inter dict path', 'splited_',name,'/','dict_',name,'.json'])
    
    train_data_save = pd.DataFrame(X_train)
    train_data_save['label'] = y_train
    

    test_data_save = pd.DataFrame(X_test)
    test_data_save['label'] = y_test
    
    
    train_data_save = train_data_save.sample(frac=1).reset_index(drop=True)
    test_data_save = test_data_save.sample(frac=1).reset_index(drop=True)
    
    
    train_data_save.to_csv(save_path_train, index=False, header=False)
    test_data_save.to_csv(save_path_test, index=False, header=False)
    with open(save_path_dict, 'w') as f:
        json.dump(label_dict, f)



if __name__ == "__main__":
    set_seed(2024)
 
    dataset_intra = ['AMB', 'Baron Human', 'TM', 'Zheng 68K', 'Zheng sorted']
    
    dataset_inter = ['10Xv2', '10Xv3', 'Drop-Seq', 'inDrop', 'Seq-Well']
    
    idx_list = [6444, 3222, 3222, 3222, 3176]
      
    for i, name in enumerate(dataset_intra):
        print('{}'.format(i))
        get_data_split_intra(name)    
    
    for i, name in enumerate(dataset_inter):
        print('{}'.format(i))
        get_data_split_inter(name, idx_list[i])    
        
        
        
        
        
        
        
        