# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:50:49 2024

@author: gaga6
"""

import time
import argparse

import json

import pandas as pd
import numpy as np

from utils import set_seed, dis_loss, cfc_loss, convert_label_to_type
from discriminant import Discriminant_score
from model import GenexpNet

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from imblearn.metrics import sensitivity_score, specificity_score


import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from utils import type_to_label_dict, convert_type_to_label



import warnings
warnings.filterwarnings('ignore')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--Pre_training', type=bool, default=True,
                        help='Apply Pre-train or not')
    parser.add_argument("--n_epochs_pre", type=int, default=20,
                        help='Number of epochs for pre-training.')
    parser.add_argument("--n_epochs_cla", type=int, default=100,
                        help='Number of epochs for classificaiton training.')  
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size for training ",
                        )
    # optimization
    
    parser.add_argument('--lr_rec', type=float, default=0.01, 
                        help='pretraining learning rate.')
    parser.add_argument('--lr_cla', type=float, default=0.01, 
                        help='classification learning rate.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=100,
                        help='Regularization rate of discriminant score.') 
    parser.add_argument('--w2', type=float, default=1000,
                        help='Weight of discriminant loss.') 
    parser.add_argument('--w1', type=float, default=1e-6,
                        help='Weight of layer regularization loss.') 
    parser.add_argument('--lr_decay_steps', type=int, default=20,
                        help='Decay the learning rate interval step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='Decay the learning rate')
    
    # model dataset
    parser.add_argument('--g_num', type=int, default=2000,
                        help='Top gene number')
    parser.add_argument('--data_type', type=str, default='intra',
                        choices=[
                                'intra',                 
                                'inter',                           
                                 ],
                        help='dataset name')
    parser.add_argument('--dataset', type=str, default='AMB',
                        choices=[
                                'AMB',            #12832  512   intra     
                                'Baron Human',   #8569  256      intra               
                                'GSE10072',      #107  64      intra       
                                'GSE15471',      #78  64       intra
                                'TM',            #54865  1024    intra
                                'Zheng 5K',      #5000  256     intra
                                'Zheng 25K',     #25000  512    intra
                                'Zheng 50K',     #50000  1024   intra
                                'Zheng 68K',     #68000  1024    intra
                                'Zheng sorted',  #200000  512  intra
                                
                                '10Xv2',
                                '10Xv3',
                                'Drop-Seq',
                                'inDrop',
                                'Seq-Well'
                                 ],
                        help='dataset name')
    
    # other setting
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--seed', type=int, default=2025, 
                        help='Random seed.')

    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def Attention_PREtrain(model, train_loader, optimazer_pre, scheduler_pre, ATT_loss, LDF_index, args):
    
    model.train()
    best_epoch = 0
    best_loss = float("inf")
    history_avg = []  
    train_start =  time.time()
    
    for epoch in range(args.n_epochs_pre):
    
        All_A = []
        train_loss = 0.0   
        times = 0
        epoch_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
            times=times+1
            inputs = inputs.to(args.device)
            _,A,_,_ = model(inputs)
            All_A.append(A)
            optimazer_pre.zero_grad()
            loss = dis_loss(A,torch.from_numpy(LDF_index).to(args.device), ATT_loss)
            loss.backward()
            optimazer_pre.step()
            scheduler_pre.step()
            
            train_loss += loss.item()
            
            
        avg_train_loss = train_loss / times
        history_avg.append([epoch, avg_train_loss])
        epoch_end = time.time()
        if (epoch+1)%5 == 0:
            print('Train:[{}/{}]\n' 
                  'Training Loss: {:.8f}\n'
                  'Time: {:.4f}'.format(
                      epoch+1, args.n_epochs_pre, loss.item(), epoch_end-epoch_start))
            
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_epoch = epoch
            
    All_A = torch.cat((All_A),0)
    train_end = time.time()           
    print("---Epoch:{}/{}--- \n Training: Loss: {:.4f}\n\t Best Loss: {:.4f} at epoch {:03d}\n Time: {:.4f}s".format(
        epoch + 1, args.n_epochs_pre, avg_train_loss, best_loss, best_epoch, train_end-train_start))
     
    return history_avg, All_A        

def CLA_train(model, train_loader, val_loader, optimizer, scheduler_cla, cla_loss, ATT_loss, LDF_index, args):
    model.train()   
    best_epoch = 0
    best_loss = float("inf")
    
    history_avg = []
    
    for epoch in range(args.n_epochs_cla):  
        epoch_start = time.time()
        
        All_A = []
        W_x = []
        times = 0
        total_batch = 0
        train_loss,train_correct = 0.0,0
     
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
        
            times=times+1
            inputs, labels = inputs.to(args.device), labels.type(torch.LongTensor).to(args.device)
            outputs,A,Weighted_X,z = model(inputs)
            All_A.append(A)
            W_x.append(Weighted_X)
                       
            optimizer.zero_grad()
            
            loss1 = cla_loss(outputs.to(torch.float32).to(args.device),labels.type(torch.LongTensor).to(args.device))
            loss2 = dis_loss(A,torch.from_numpy(LDF_index).to(args.device),ATT_loss)  
            loss = loss1 + args.w2*loss2 + args.w1*cfc_loss(z)/len(inputs)

            loss.backward()
            optimizer.step()
            scheduler_cla.step()

            train_loss += loss.item()
            
            predictions = torch.max(F.softmax(outputs.data,dim = 1), 1)[1]
            correct_batch = (predictions == labels).sum().item()

            temp_batch = labels.size(0)
            total_batch += temp_batch
            train_correct += correct_batch
            
        avg_train_loss = train_loss / times
        avg_train_acc = train_correct / total_batch
        
        avg_val_loss, avg_val_acc, _, _, _, _, _, _ = CLA_val(model, val_loader, cla_loss, args)
        
        history_avg.append([epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        
        All_A = torch.cat((All_A),0)
        new_x = torch.cat((W_x),0)
        
            
        epoch_end = time.time()
        
        if (epoch+1)%5 == 0:
            print("Epoch:[{}]/[{}] \n Train Loss: {:.4f} \t Train acc: {:.2f}%\n Val Loss: {:.4f} \t Val Acc: {:.2f}% \n Time: {:.4f}s".format(
                epoch+1, args.n_epochs_cla, avg_train_loss, avg_train_acc*100,
                avg_val_loss, avg_val_acc*100, epoch_end-epoch_start))

            print(
                   'cla lr: {:.8f}\n'                
                   'Time: {:.4f}'.format(
                       optimizer.param_groups[0]['lr'],  epoch_end-epoch_start))
            print("Best Loss: {:.4f} at epoch {}".format(best_loss, best_epoch))
          
    print("------Finished Training------")
    
    return history_avg,best_epoch,All_A,new_x 
    
    
def CLA_val(model, val_loader, criterion, args):
    
    model.eval()
    val_loss, val_correct = 0.0, 0
    acc, f1, recall, precision, sensitivity, specificity= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_batch = 0
    times = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            times = times + 1
            n_row,n_col = inputs.size()
            inputs, labels = inputs.to(args.device), labels.type(torch.LongTensor).to(args.device)
            output,_,_,_ = model(inputs)
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            predictions = torch.max(F.softmax(output.data,dim = 1), 1)[1]
            correct_batch = (predictions == labels).sum().item()
            
            temp_batch = labels.size(0)
            total_batch += temp_batch
            val_correct += correct_batch
            
            acc += accuracy_score(y_true = labels.cpu(), y_pred = predictions.cpu())
            f1 += f1_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            recall += recall_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            precision += precision_score(y_true=labels.cpu(), y_pred = predictions.cpu(), average='weighted')
            sensitivity += sensitivity_score(y_true=labels.cpu().data, y_pred = predictions.cpu().data, average='weighted')
            specificity += specificity_score(y_true=labels.cpu().data, y_pred = predictions.cpu().data, average='weighted')   
            
    avg_val_loss = val_loss / times
    
    return avg_val_loss, acc/times, f1/times, recall/times, precision/times, sensitivity/times, specificity/times, predictions 
    

def set_loader(args):
    
    train_data = pd.read_csv(
        ''.join(['E:/My_project/GenexpNet/datasets/',args.data_type,'/preprocessed/splited/','Splited_',args.dataset,'/','Train_',args.dataset,'.csv']),               
        index_col = None, header=None)
    val_data = pd.read_csv(
        ''.join(['E:/My_project/GenexpNet/datasets/',args.data_type,'/preprocessed/splited/','Splited_',args.dataset,'/','Val_',args.dataset,'.csv']),
        index_col = None, header=None)
    test_data = pd.read_csv(
        ''.join(['E:/My_project/GenexpNet/datasets/',args.data_type,'/preprocessed/splited/','Splited_',args.dataset,'/','Test_',args.dataset,'.csv']),
        index_col = None, header=None)
    
    with open(''.join(['E:/My_project/GenexpNet/datasets/',args.data_type,'/preprocessed/splited/','Splited_',args.dataset,'/','dict_',args.dataset,'.json'])) as file:
        label_dict = json.load(file)
        
    n_row, n_col = train_data.shape
    
    print("number of samples is {}".format(n_row))
    
    X_train = train_data.iloc[:,0:n_col-1]
    X_val = val_data.iloc[:,0:n_col-1]
    X_test = test_data.iloc[:,0:n_col-1]
    
    y_train = train_data.iloc[:,-1]
    y_val = val_data.iloc[:,-1]
    y_test = test_data.iloc[:,-1]
    
    X_all = pd.concat([X_train, X_val, X_test], axis=0)
    y_all  = pd.concat([y_train, y_val, y_test], axis=0)
    
      
    dicts = type_to_label_dict(y_all)
    
    y_train = convert_type_to_label(y_train, dicts)
    y_val = convert_type_to_label(y_val, dicts)
    y_test = convert_type_to_label(y_test, dicts)
    y_all = convert_type_to_label(y_all, dicts)

    args.n_class = len(set(np.array(y_all)))
    
     
    tensor_TrainValues = torch.FloatTensor(np.array(X_train)).float()
    tensor_ValValues = torch.FloatTensor(np.array(X_val)).float()
    tensor_TestValues = torch.FloatTensor(np.array(X_test)).float()
     
    tensor_TrainLabels = torch.FloatTensor(y_train).float()
    tensor_ValLabels = torch.FloatTensor(y_val).float()
    tensor_TestLabels = torch.FloatTensor(y_test).float()
    
    args.data_dim = n_col-1
    
    return tensor_TrainValues, tensor_ValValues, tensor_TestValues, tensor_TrainLabels, tensor_ValLabels, tensor_TestLabels, label_dict

def set_model(args):
    
    model = GenexpNet(input_dim = args.data_dim, n_class = args.n_class, dropout = args.dropout)
   
    para_cla = model.parameters()
     
    para_att = [
              {'params': model.se_enc.parameters()},
              {'params': model.se_tanh.parameters()},
              {'params': model.se_drop.parameters()},
              {'params': model.se_dec.parameters()},
              {'params': model.se_sigmoid.parameters()}
                ]


    optimazer_pre = torch.optim.Adam(para_att,lr=args.lr_rec)
    optimizer_cla = torch.optim.Adam(para_cla,lr=args.lr_cla)

    
    scheduler_1 = StepLR(optimazer_pre, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    
    scheduler_2 = StepLR(optimizer_cla, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    
    cla_loss = torch.nn.CrossEntropyLoss()
    ATT_loss = torch.nn.MSELoss()
    
    
    model.to(args.device)
    cla_loss.to(args.device)
    ATT_loss.to(args.device)

    return model, cla_loss, ATT_loss, optimazer_pre, optimizer_cla, scheduler_1, scheduler_2

def main():
    
        args = parse_args()
        set_seed(args.seed)
            
        Train_data, Val_data, Test_data, Train_label, Val_label, Test_label, label_dict = set_loader(args)
            
        LDF_index = Discriminant_score(Train_data.numpy(), Train_label.numpy(),args.beta)
            
        Train_dataset = Data.TensorDataset(Train_data, Train_label)
        Val_dataset = Data.TensorDataset(Val_data, Val_label)
        Test_dataset = Data.TensorDataset(Test_data, Test_label)
            
        train_loader = torch.utils.data.DataLoader(
                    Train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False
                )
            
        val_loader = torch.utils.data.DataLoader(
                    Val_dataset,
                    batch_size=1000000,
                    shuffle=False,
                    drop_last=False
                )
            
        test_loader = torch.utils.data.DataLoader(
                    Test_dataset,
                    batch_size=1000000,
                    shuffle=False,
                    drop_last=False
                )
            
        acc_list = list()
        f1_list = list()
        recall_list = list()
        precision_list = list()
        sensitivity_list = list()
        specificity_list = list()
        times_list = list()
        
        predict_list = list()
            
        for i in range(args.iterations):
                print(args.dataset)
                print(i)
                print(args.g_num)
                print("iteration: {}".format(i+1))
                time_stratime = time.time()
                
                model, cla_loss, ATT_loss, optimazer_pre, optimizer_cla, scheduler_pre, scheduler_cla = set_model(args)
                
                print("==> Pre_training..")
                history_avg, All_A  = Attention_PREtrain(model, train_loader, optimazer_pre, scheduler_pre, ATT_loss, LDF_index, args)
                print("==> classification learning..")
                history_avg,best_epoch,All_A,new_x  = CLA_train(model, train_loader, val_loader, optimizer_cla, scheduler_cla, cla_loss, ATT_loss, LDF_index, args)
                test_loss, test_acc, test_f1, test_recall, test_precision, test_sensitivity, test_specificity, predictions = CLA_val(model, test_loader, cla_loss, args)
                
                  
                time_endtime = time.time()
                time_cost = time_endtime-time_stratime
                
                acc_list.append(test_acc)
                f1_list.append(test_f1)
                recall_list.append(test_recall)
                precision_list.append(test_precision)
                sensitivity_list.append(test_sensitivity)
                specificity_list.append(test_specificity)
                predict_list.append(convert_label_to_type(predictions.detach(),label_dict))
                times_list.append(time_cost)
                
                print("==> Test.. \n  Test acc: {:.6f} | Test f1: {:.6f} | Test recall: {:.6f} | Test precision: {:.6f} | Test sensitivity: {:.6f} | Test specificity: {:.6f} | Time: {:.2f}".format(
                     test_acc*100, test_f1*100, test_recall*100, test_precision*100, test_sensitivity*100, test_specificity*100, time_cost))
                
        print('all iterations')    
        print(np.mean((np.array(acc_list))))
        print(np.mean((np.array(f1_list))))
        print(np.mean((np.array(recall_list))))
        print(np.mean((np.array(precision_list))))
        print(np.mean((np.array(sensitivity_list))))
        print(np.mean((np.array(specificity_list))))
        print(np.mean((np.array(times_list))))
            
        contact_list = {"acc" : acc_list,
                        "f1" : f1_list,
                        "recall" : recall_list,
                        "precision" : precision_list,
                        "sensitivity" : sensitivity_list,
                        "specificity" : specificity_list,
                        "time_cost" : times_list   
            }
        
        true_str_label = convert_label_to_type(Test_label, label_dict)
             
        cm_list = list()
        
        xy_lablels = sorted(list(set(true_str_label)))
        
        
        for i in range(len(predict_list)):
            cm_list.append(confusion_matrix(true_str_label, predict_list[i], normalize='true', labels=xy_lablels))
        
        
        cm_sum = np.zeros((len(set(true_str_label)), len(set(true_str_label))), dtype=float)
        
        for  i in range(len(cm_list)):
            cm_sum += cm_list[i]
            
      
        # Calculate the average confusion matrix
        average_cm = cm_sum / len(cm_list)
        
        # Heatmap of the average confusion matrix
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(7, 7))
        cmap = LinearSegmentedColormap.from_list("gray_red_black", ["lightgray", "darkred", "black"])

        ax = sns.heatmap(average_cm, annot=False, fmt='.2f', cmap=cmap,cbar_kws={'shrink': 0.6},square=True, xticklabels=xy_lablels, yticklabels=xy_lablels)
        
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)

        
        plt.title('GenexpNet')
        plt.xlabel('Predicted label', fontsize=20)
        plt.ylabel('True Label', fontsize=20)
        plt.tight_layout()
        plt.savefig(''.join(['E:/My_project/GenexpNet/datasets/result/',args.data_type,'/',args.dataset,'/Heatmap_GenexpNet_',args.dataset,'.png']), dpi=300)
        plt.show()
            
        result =  pd.DataFrame(contact_list)
            
        result.to_csv(''.join(['E:/My_project/GenexpNet/datasets/result/',args.data_type,'/',args.dataset,'/','Result_GenexpNet_',args.dataset,'.csv'])
                    ,encoding='gbk',index=None)
                
                
if __name__ == "__main__":
    main()        