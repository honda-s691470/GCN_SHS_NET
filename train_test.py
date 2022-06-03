"""
Copyright (c) 2021 txWang
the original source code of def prepare_trval_data, gen_trval_adj_mat is released under the MIT License
https://github.com/txWang/MOGONET/blob/main/LICENSE
"""

import os
import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from utils import *

cuda = True if torch.cuda.is_available() else False


def prepare_trval_data(config, i, mode="train_val"):
    if mode == "train_val":
        labels_tr = np.loadtxt(os.path.join(config["data_folder"], f'{i}', "labels_tr.csv"), delimiter=',')
        labels_val = np.loadtxt(os.path.join(config["data_folder"], f'{i}', "labels_val.csv"), delimiter=',')
        labels_tr = labels_tr.astype(int)
        labels_val = labels_val.astype(int)

        data_tr = np.loadtxt(os.path.join(config["data_folder"], f'{i}', "1_tr.csv"), delimiter=',')
        data_val = np.loadtxt(os.path.join(config["data_folder"], f'{i}', "1_val.csv"), delimiter=',')
        num_tr = data_tr.shape[0]
        num_val = data_val.shape[0]
 
    data_mat = np.concatenate((data_tr, data_val), axis=0)
    
    data_tensor = torch.FloatTensor(data_mat)
    data_tensor = data_tensor.cuda()
    
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["val"] = list(range(num_tr, (num_tr+num_val)))
    
    data_train = data_tensor[idx_dict["tr"]].clone()
    data_validation = data_tensor[idx_dict["val"]].clone()
    data_all = torch.cat((data_tensor[idx_dict["tr"]].clone(),data_tensor[idx_dict["val"]].clone()),0)
    labels = np.concatenate((labels_tr, labels_val))
    
    return data_train, data_validation, data_all, idx_dict, labels


def gen_trval_adj_mat(data_tr, data_trte, trte_idx, adj_parameter):
    adj_metric = "cosine" 
    adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr, adj_metric)
    adj_train = gen_adj_mat_tensor(data_tr, adj_parameter_adaptive, adj_metric)
    adj_test = gen_val_adj_mat_tensor(data_trte, trte_idx, adj_parameter_adaptive, adj_metric)
    
    return adj_parameter_adaptive, adj_train, adj_test

def feat_imp(config, best_model_index, gcn_list, best_gcn_name, dim_list):
    featimp_all = []
    featname_list = []
    feat_imp_list = []
    for k in range(config["fold_num"]):
        gcn_list[best_model_index].cuda()
        gcn_list[best_model_index].load_state_dict(torch.load(config["data_folder"] + "/" + f'{k}'+ f'/weight_best_auc_model_num_%s' % (best_model_index) + ".pth"))
        gcn_list[best_model_index].eval()

        data_tr, data_val, data_all, trval_idx, labels_trval = prepare_trval_data(config, k)
        adj_parameter_adaptive, adj_tr, adj_all = gen_trval_adj_mat(data_tr, data_all, trval_idx, config["adj_parameter"])

        val_idx = trval_idx["val"]
        ci_list = []

        ci = gcn_list[best_model_index](data_all, adj_all)
        ci_list.append(ci.detach())
        c = ci_list[0]
        c = c[val_idx,:]

        prob = F.softmax(c, dim=1).data.cpu().numpy()
        auc, f1, acc = performance_ind (labels_trval, trval_idx, prob)
        print(f"{k+1}th_auc", auc)

        df = pd.read_csv(config["data_folder"] + "/" + f'{k}' + "/1_featname.csv", header=None)
        featname_list.append(df.values.flatten())

        feat_imp = {"feat_name":featname_list[k]}
        feat_imp['imp'] = np.zeros(dim_list)

        for j in range(dim_list):
            feat_tr = data_tr[:,j].clone()
            feat_all = data_all[:,j].clone()
            data_tr[:,j] = 0
            data_all[:,j] = 0

            adj_parameter_adaptive, adj_tr, adj_all = gen_trval_adj_mat(data_tr, data_all, trval_idx, config["adj_parameter"])

            ci_list = []

            ci = gcn_list[best_model_index](data_all, adj_all)
            ci_list.append(ci.detach())
            c = ci_list[0]
            c = c[val_idx,:]

            prob = F.softmax(c, dim=1).data.cpu().numpy()

            auc_tmp, _, _ = performance_ind (labels_trval, trval_idx, prob)
            feat_imp['imp'][j] = (auc-auc_tmp)*dim_list

            data_tr[:,j] = feat_tr.clone()
            data_all[:,j] = feat_all.clone()

        feat_imp_list.append(pd.DataFrame(feat_imp).set_index('feat_name', drop=True))
    df_featimp = pd.concat(feat_imp_list, axis=1).copy(deep=True)
    sumstat_imp = pd.DataFrame(df_featimp.T.describe())
    sumstat_imp_sort = sumstat_imp.T.sort_values(by='mean',ascending=False)
    sumstat_imp_sort["mean"].to_csv(config["data_folder"] + "/df_featimp_wrist.csv", index=True)

    sumstat_imp_sort.reset_index(inplace=True)
    figure = plt.figure(figsize=(15, 15))
    sns.barplot('mean','feat_name', data=sumstat_imp_sort,orient='h')
    plt.title('GCN - Feature Importance',fontsize=28)
    plt.ylabel('Features',fontsize=18)
    plt.xlabel('Importance',fontsize=18)
    plt.show()

def train_val (config):
    for k in range(config["fold_num"]): 
        print(f'======{k+1}th fold======')
        data_tr, data_val, data_all, trval_idx, labels_trval = prepare_trval_data(config, k)

        labels_tr_tensor = torch.LongTensor(labels_trval[trval_idx["tr"]])
        labels_val_tensor = torch.LongTensor(labels_trval[trval_idx["val"]])
        sample_weight_tr = torch.FloatTensor(cal_sample_weight(labels_trval[trval_idx["tr"]], config["num_class"], use_sample_weight=False))
        sample_weight_val = torch.FloatTensor(cal_sample_weight(labels_trval[trval_idx["val"]], config["num_class"], use_sample_weight=False))

        if cuda:
            label_tr = labels_tr_tensor.cuda()
            label_val = labels_val_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
            sample_weight_val = sample_weight_val.cuda()
        torch.cuda.empty_cache()

        adj_parameter_adaptive, adj_tr, adj_all = gen_trval_adj_mat(data_tr, data_all, trval_idx, config["adj_parameter"])

        d_today = datetime.date.today()
        out_dim=2

        #display number of features
        dim_list=len(data_tr[0])
        print("number of features" ,dim_list)

        loss_dict = {}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion2 = torch.nn.MSELoss(reduction='none')

        val_idx = trval_idx["val"]
        gcn_list=[]
        dataframe = []
        model_num = 0
        for h in range(config["num_test_model"]):
            for i in range(len(config["slope"])):
                for j in range(len(config["dropout"])):
                    gcn_name = models.__all__[h]
                    print("=======================")
                    print("slope", config["slope"][i])
                    print("gcn_dropout", config["dropout"][j])
                    print("model_name", gcn_name)

                    #load model
                    gcn_list.append(models.__dict__[gcn_name](dim_list, out_dim, config["dropout"][j], config["slope"][i]))
                    gcn_list[model_num].cuda()

                    optimizer = torch.optim.Adam(gcn_list[model_num].parameters(), lr=config["lr"])
                    scheduler = scheduler_maker(optimizer, config)

                    # start training
                    train_loss_list = []
                    val_loss_list = []
                    auc_list = []
                    acc_list = []
                    f1_list = []
                    best_auc = 0.20
                    best_f1 = 0.20
                    for epoch in range(1, config["epochs"]+1):

                        gcn_list[model_num].train()
                        optimizer.zero_grad()
                        ci = gcn_list[model_num](data_tr, adj_tr)
                        ci_loss_tr = torch.mean(torch.mul(criterion(ci, label_tr),sample_weight_tr))

                        ci_loss_tr.backward()
                        optimizer.step()

                        if config["scheduler"] == 'CosineAnnealingLR':
                            scheduler.step()
                        elif config["scheduler"] == 'ReduceLROnPlateau':
                            scheduler.step(ci_loss_tr)
                        elif config["scheduler"] == 'ConstantLR':
                            pass

                        gcn_list[model_num].eval()
                        with torch.no_grad():
                            ci_list = []
                            
                            ci = gcn_list[model_num](data_all, adj_all)
                            ci_list.append(ci.detach())
                            c = ci_list[0]
                            c = c[val_idx,:]

                            ci_loss_val = torch.mean(torch.mul(criterion(c, label_val),sample_weight_val))
                            prob = F.softmax(c, dim=1).data.cpu().numpy()
                            train_loss_list.append(ci_loss_tr.item())
                            val_loss_list.append(ci_loss_val.item())

                            acc_list, f1_list, auc_list, best_auc = performance_calc (config, k, labels_trval, trval_idx, 
                                                                                      prob, auc_list, f1_list, acc_list, 
                                                                                      best_auc, gcn_list, model_num)

                    print("Test AUC: {:.3f}".format(max(auc_list)))
                    max_idx = np.array(auc_list).argmax()
                    print("Test F1: {:.3f}".format(f1_list[max_idx]))
                    print("Test ACC: {:.3f}".format(acc_list[max_idx]))

                    dataframe.append([gcn_name, config["slope"][i], config["dropout"][j], max(acc_list), max(auc_list), max(f1_list)])

                    #visualize learning curves and performance indicator
                    figure_maker (config, train_loss_list, val_loss_list, acc_list, auc_list, f1_list, k)
                    model_num += 1

        dataframe_pd=pd.DataFrame(dataframe).set_axis(['gcn_name', 'slope', 'dropout', 'acc', 'auc', 'f1'], axis='columns', inplace=False)
        dataframe_pd.to_csv(config["data_folder"] + "/" + f'{k}' + "/result_" + str(d_today) + ".csv", index=False)
    return dataframe, gcn_list, dim_list