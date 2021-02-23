from __future__ import division
import math
import scipy.misc

import numpy as np
import random
import copy
import pickle
import pandas as pd
import csv
import os
import sys
import torch
import torch.nn.functional as f
import shutil
import pickle
import numpy as np
import bottleneck as bn
import collections
from scipy import sparse
from scipy.special import softmax

import matplotlib.pyplot as plt
import pylab

def save_weights_pkl(fname, weights):
    with open(fname, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

def load_weights_pkl(fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f)
    return weights

def get_parameters(model, bias=False):
    for k, m in model.named_parameters():
        if bias:
            if k.endswith('.bias'):
                yield m
        else:
            if k.endswith('.weight'):
                yield m

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    assert pd.unique(tp_tr["uid"]).shape[0] == end_idx - start_idx + 1
    assert pd.unique(tp_te["uid"]).shape[0] == end_idx - start_idx + 1

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

def DCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    topk_idx = np.argsort(-X_pred)[:,:k]
    return DCG, topk_idx

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


# "Controlling Popularity Bias in Learning-to-Rank Recommendation" https://dl.acm.org/citation.cfm?id=3109912
def Apt_at_k_batch(X_pred, heldout_batch, item_mapper, k=100, tail_number = 2.0):
    # TAIL NUMBER PARAMETER
    # 0 - short head
    # 1 - medium tail
    # 2 - long tail
    batch_users = X_pred.shape[0]

    #idx = bn.argpartition(-X_pred, k, axis=1) # top k
    idx = np.argpartition(-X_pred, k, axis=1) # top k
    apt_list = []
    for user in idx[:,:k]:
        categs = item_mapper.loc[item_mapper['new_movieId'].isin(user)]['categ'].values
        dict_out = dict(collections.Counter(categs))
        if tail_number in dict_out.keys(): apt_list.append(dict_out[tail_number]/k)
        else: apt_list.append(0)
    #if 1.0 in dict_out.keys(): apt += dict_out[1.0]
    return [np.sum(apt_list)/batch_users]

def dcg_k_rounds(scores_rounds):
    dcgs_rounds = []
    for iround in range(scores_rounds.shape[0]):
        dcg_rounds.append(dcg_k_users(scores_rounds[iround,:,:]))
    return np.array(dcgs_rounds)

def dcg_k_users(scores):
    dcg_round = []
    for user in range(scores.shape[0]):
        dcg_round.append(dcg_single_ranking(scores[user,:]))
    return np.array(dcg_round)

def dcg_single_ranking(scores):
    dcg = 0.0
    for idx in range(len(scores)):
        curr = scores[idx]/np.log2(idx + 2)
        dcg += curr
    return dcg

def calc_pop_bias(pcounts):
    popb=[]
    for row in pcounts:
        popb_iter=row[0]
        for j in np.arange(1,len(row),1):
            popb_iter+=(row[j]/np.log2(j+1))
        row = -np.sort(-row)
        ipopb_iter=row[0]
        for j in np.arange(1,len(row),1):
            ipopb_iter+=(row[j]/np.log2(j+1))
        popb.append(popb_iter/ipopb_iter)
    return popb

def att_rel(logit, data_tr, count_weight, play_count, cuda, k=10, p=0.5):
    
    # TOPK SCORES (BATCH_SIZE x N_ITEMS -> BATCH_SIZE x k)
    #logit_k, idxs =torch.topk(logit,k)
    #rel_norm = f.normalize(logit_k, p=1, dim=1)
    
    sort_rel, idx_rel = torch.sort(logit,descending=True)

    # count and count_norm repeated in the logit format
    cnt = count_weight.repeat(logit.shape[0]).view(logit.shape)
    pcount = play_count.repeat(logit.shape[0]).view(logit.shape)
    if cuda: 
        cnt = cnt.to('cuda')
        pcount = pcount.to('cuda')
    
    indexes = torch.tensor(range(k))
    if cuda: indexes = indexes.to('cuda')
    cnt = torch.index_select(torch.gather(cnt, 1, idx_rel),1,indexes)
    #cnt = f.normalize(cnt, p=1, dim=1)
    pcount = torch.index_select(torch.gather(pcount, 1, idx_rel),1,indexes)
    #cnt,_ = torch.sort(cnt, descending=True)
    
    #plot_line(rel_norm.cpu().detach().numpy()[0,:],att.detach().numpy()[0,:],"attention")
    #plot_line(rel_norm.cpu().detach().numpy()[0,:],cnt.cpu().detach().numpy()[0,:],"popularity")

    return cnt.cpu().detach().numpy(), pcount.cpu().detach().numpy()

