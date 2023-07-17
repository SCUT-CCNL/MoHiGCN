""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,precision_score,recall_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from collections import Counter
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
cuda = True if torch.cuda.is_available() else False
from sklearn.model_selection import train_test_split


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))

    # load g2g
    data_g2g_list=[]
    for i in [1,2]:
        adj_g2g=np.loadtxt(os.path.join(data_folder, str(i)+"_g2g.csv"), delimiter=',')

        adj_g2g=torch.FloatTensor(adj_g2g)
        I = torch.eye(adj_g2g.shape[0])
        adj_g2g = F.normalize(adj_g2g+I, p=0)
        data_g2g_list.append(adj_g2g)
        if cuda:
            data_g2g_list[i-1] = data_g2g_list[i-1].cuda()

    return data_train_list, data_all_list, idx_dict, labels, data_g2g_list


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))

        #te_parameter== cal_adj_mat_parameter(adj_parameter, data_tete_list[i], adj_metric)
        #adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        #adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_parameter, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, adj_g2g_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()

    fusion_list=[0,1,2]
    num_view=len(fusion_list)
    for i in fusion_list:
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        if i==2:
            hi = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            ci = model_dict["C{:}".format(i + 1)](hi)
        else:
            hi = model_dict["E{:}".format(i+1)](data_list[i],adj_list[i],adj_g2g_list[i])
            ci = model_dict["C{:}".format(i+1)](hi)
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
        prob = F.softmax(ci, dim=1).data.cpu().numpy()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        hi_list = []
        for i in range(num_view):
            if i==2:
                hi = model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])
                hi_list.append(hi)
                ci_list.append(model_dict["C{:}".format(i+1)](hi))
            else:
                hi = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i], adj_g2g_list[i])
                hi_list.append(hi)
                ci_list.append(model_dict["C{:}".format(i + 1)](hi))
        c = model_dict["C"](ci_list, hi_list)
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
        prob = F.softmax(c, dim=1).data.cpu().numpy()
    return loss_dict, prob
    

def test_epoch(data_list, adj_list, adj_g2g_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    ci_list = []
    hi_list = []
    fusion_list=[0,1,2] #[0,1,2]
    num_view=len(fusion_list)
    for i  in fusion_list:
        if i == 2:
            hi = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            hi_list.append(hi)
            ci_list.append(model_dict["C{:}".format(i + 1)](hi))
        else:
            hi = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i], adj_g2g_list[i])
            hi_list.append(hi)
            ci_list.append(model_dict["C{:}".format(i + 1)](hi))

    if num_view >= 2:
        c = model_dict["C"](ci_list, hi_list)
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, weight_decay, num_epoch_pretrain, num_epoch):
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2#
        dim_he_list = [200,200,100]#
    if data_folder[:11] == 'OurDBs/BRCA':
        adj_parameter = 10
        dim_he_list = [800, 400, 200]
    if data_folder[:11] == 'OurDBs/HNSC':
        adj_parameter = 2
        dim_he_list = [800, 400, 200]
    if data_folder[:11] == 'OurDBs/CESC':
        adj_parameter = 2
        dim_he_list = [800, 400, 200]
    if data_folder[:11] == 'OurDBs/STAD':
        adj_parameter = 2
        dim_he_list = [800, 400, 200]
    if data_folder[:10] == 'OurDBs/LGG':
        adj_parameter = 10
        dim_he_list = [800, 400, 200]
    data_tr_list, data_trte_list, trte_idx, labels_trte, data_g2g_list = prepare_trte_data(data_folder, view_list)
    print(len(trte_idx['tr']))
    print(len(trte_idx['te']))
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()


    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c, weight_decay)
    for epoch in range(num_epoch+1):
        loss, tr_prob=train_epoch(data_tr_list, adj_tr_list, data_g2g_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, data_g2g_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            print('loss:', loss['C'])
            if num_class == 2:
                print("Train ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                print("Train F1: {:.3f}".format(f1_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                #print("Train Precison: {:.3f}".format(precision_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                print("Train AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["tr"]], tr_prob[:,1])))
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                #print("Test Precison: {:.3f}".format(precision_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Train ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                print("Train F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1), average='weighted')))
                print("Train F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1), average='macro')))
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()
    if num_class == 2:
        ACC=accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        F1=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        #F1 = precision_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        AUC=roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
        return ACC, F1, AUC, model_dict
    else:
        ACC=accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
        F1_weighted=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
        F1_macro=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
        return ACC, F1_weighted, F1_macro, model_dict

