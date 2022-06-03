import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

cuda = True if torch.cuda.is_available() else False

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    
    #g is the adjacency matrix with zero diagonal and converted to 1 for cases with close cosine similarity
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
        
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda(0)
        
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)

    adj_sparse = to_sparse(adj)
    
    return adj_sparse


def gen_val_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["val"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["val"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    torch.cuda.empty_cache()
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj


def scheduler_maker(optimizer, config):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']/4, eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler

def split_matrix (config):
    df_tr_val = pd.read_csv(config["train_val_data"], sep=",", engine='python')
    os.makedirs(config["data_folder"], exist_ok=True)
    for i in range (config["fold_num"]):
        os.makedirs(config["data_folder"] + "/" + f'{i}', exist_ok=True)

    feature = pd.DataFrame(df_tr_val.columns.tolist()[2:])
    for i in range(config["fold_num"]):
        feature.to_csv(config["data_folder"] + "/" + f'{i}'+  "/1_featname.csv", index=False, header=False)
        feature.to_csv(config["data_folder"] + "/" + f'{i}'+  "/1_featname.csv", index=False, header=False)

    df_train_list=[]
    df_val_list=[]
    kf = KFold(n_splits=config["fold_num"], random_state=42, shuffle=True)
    for train, val in kf.split(df_tr_val):
        df_train_list.append(df_tr_val.iloc[train,:])
        df_val_list.append(df_tr_val.iloc[val,:])
       
    pheno=df_tr_val.columns.tolist()[1]
    idx = df_tr_val.columns.tolist()[0]
    for i in range(config["fold_num"]):
        labels_tr = df_train_list[i][pheno]
        labels_val = df_val_list[i][pheno]
        img_id_tr = df_train_list[i][idx]
        img_id_val = df_val_list[i][idx]
        train = df_train_list[i].drop(columns=[pheno]).drop(columns=[idx])
        val = df_val_list[i].drop(columns=[pheno]).drop(columns=[idx])

        labels_tr.to_csv(config["data_folder"] + "/" + f'{i}' + "/labels_tr.csv", index=False, header=False)
        labels_val.to_csv(config["data_folder"] + "/" + f'{i}' + "/labels_val.csv", index=False, header=False)
        img_id_tr.to_csv(config["data_folder"] + "/" + f'{i}' + "/id_tr.csv", index=False, header=True)
        img_id_val.to_csv(config["data_folder"] + "/" + f'{i}' + "/id_val.csv", index=False, header=True)
        train.to_csv(config["data_folder"] + "/" + f'{i}' + "/1_tr.csv", index=False, header=False)
        val.to_csv(config["data_folder"] + "/" + f'{i}' + "/1_val.csv", index=False, header=False)
        feature.to_csv(config["data_folder"] + "/" + f'{i}' + "/1_featname.csv", index=False, header=False)
        
        
        
def performance_ind (labels_trval, trval_idx, prob):
    fpr, tpr, thresholds = roc_curve(labels_trval[trval_idx["val"]], prob[:,1])
    Youden_index_candidates = tpr-fpr
    index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
    cutoff = thresholds[index]
    cutoff_result = np.where(prob[:,1] < cutoff, 0, 1)
   
    auc_temp = roc_auc_score(labels_trval[trval_idx["val"]], prob[:,1])
    f1_temp = f1_score(labels_trval[trval_idx["val"]],cutoff_result)
    acc_temp = accuracy_score(labels_trval[trval_idx["val"]], cutoff_result) 
    
    return auc_temp, f1_temp, acc_temp
        
def performance_calc (config, k, labels_trval, trval_idx, prob, auc_list, f1_list, acc_list, best_auc, gcn_list, model_num):
    auc_temp, f1_temp, acc_temp = performance_ind (labels_trval, trval_idx, prob)
    auc_list.append(auc_temp)
    f1_list.append(f1_temp)
    acc_list.append(acc_temp)

    try:
        if  auc_temp > best_auc:
            torch.save(gcn_list[model_num].state_dict(), config["data_folder"] + "/" + f'{k}'+ f'/weight_best_auc_model_num_%s' % (model_num)  + ".pth")
            best_auc = auc_temp
            df1 = pd.DataFrame(labels_trval[trval_idx["val"]])
            df2 = pd.DataFrame(prob[:,1])
            df_concat = pd.concat([df1, df2], axis=1)
            df_concat = df_concat.set_axis(['label', 'pred'], axis='columns')  
            df_concat["Grade"] = df_concat["label"].replace(1, 'true').replace(0, 'false')
    
    except (PermissionError) as p:
        print(p)
    return acc_list, f1_list, auc_list, best_auc

def figure_maker (config, train_loss_list, val_loss_list, acc_list, auc_list, f1_list, k):
    fig = plt.figure()
    plt.plot(range(config["epochs"]), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(config["epochs"]), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.show()
    fig.savefig(config["data_folder"] + "/" + f'{k}'+ "/tr_val_curve.png")

    fig = plt.figure()
    plt.plot(range(config["epochs"]), acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(config["epochs"]), auc_list, color='green', linestyle='--', label='val_auc')
    plt.plot(range(config["epochs"]), f1_list, color='orange', linestyle='-.', label='val_f1')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('acc_f1_auc')
    plt.grid()
    plt.show()
    fig.savefig(config["data_folder"] + "/" + f'{k}'+ "/acc_f1_auc.png")
    torch.cuda.empty_cache()
    
def result_recoder(dataframe, config):
    
    d_today = datetime.date.today()
    
    for i in range(config["fold_num"]):
        if i == 0:
            dataframe0 = pd.read_csv(config["data_folder"] + "/" + f'{i}' + "/result_" + str(d_today) + ".csv")
            dataframe0 = dataframe0[["gcn_name", "slope", "dropout","auc"]]
        if i > 0:
            dataframe = pd.read_csv(config["data_folder"] + "/" + f'{i}' + "/result_" + str(d_today) + ".csv")
            dataframe = dataframe["auc"]
            dataframe0 = pd.concat([dataframe0, dataframe],axis=1)

    all_auc = dataframe0.T.iloc[3:,0:].astype(float)
    print("========="+"Display of the best model results"+"=========")
    best_model_index = all_auc.describe().iloc[1].idxmax()
    best_gcn_name = dataframe0.T.iloc[0, best_model_index]
    print(dataframe0.T.iloc[0:, best_model_index])
    print("Mean AUC of cross-validation", all_auc.describe().iloc[1].max())
    print("best model index", best_model_index)
    marge_df = pd.concat([dataframe0.T, all_auc.describe()],axis=0).T
    marge_df.to_csv(config["data_folder"] + "/" + "All_cross_val_results_" + config["data_folder"] + "_" + str(d_today) + ".csv", index=False)
    return best_model_index, best_gcn_name
  