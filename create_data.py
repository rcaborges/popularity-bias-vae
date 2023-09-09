#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from scipy import sparse
import pandas as pd
from collections import Counter


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp.loc[tp['movieId'].isin(itemcount.loc[itemcount['size'] >= min_sc].movieId)]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp.loc[tp['userId'].isin(usercount.loc[usercount['size'] >= min_uc].userId)]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list, raw_list = list(), list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
            raw_list.append(group)
        else:
            tr_list.append(group)
            raw_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    data_raw = pd.concat(raw_list)

    return data_tr, data_te, data_raw

def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


parser = argparse.ArgumentParser("Data creation for VAE based collaborative filtering")
parser.add_argument('--dataset_name', type=str, default='ml-20m', help='dataset name', choices=['ml-20m','netflix','lfm1b'])
parser.add_argument('--out_data_dir', type=str, default='../data/ml-20m/pro_sg/', help='output data directory')
parser.add_argument('--base_dir', type=str, default='../data/ml-20m/', help='rating file')
args = parser.parse_args()
print(args)

pro_dir = args.out_data_dir

if not os.path.isdir(pro_dir): os.mkdir(pro_dir)
os.makedirs(args.out_data_dir, exist_ok=True)

if args.dataset_name == "netflix":
    DATA_DIR = args.base_dir
    raw_data_train = pd.read_csv(os.path.join(DATA_DIR, 'NF_TRAIN/nf.train.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data_valid = pd.read_csv(os.path.join(DATA_DIR, 'NF_VALID/nf.valid.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data_test = pd.read_csv(os.path.join(DATA_DIR, 'NF_TEST/nf.test.txt'), sep='\t', header=None, names=['userId','movieId','rating'])
    raw_data_orig = pd.concat([raw_data_train, raw_data_valid, raw_data_test])
    # binarize the data (only keep ratings >= 4)
    raw_data_orig = raw_data_orig[raw_data_orig['rating'] > 3.5]
    
    max_seq_len = 1000
    # Remove users with greater than $max_seq_len number of watched movies
    raw_data_orig = raw_data_orig.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)
    
    # Sort data values with the timestamp
    #raw_data_orig = raw_data_orig.groupby(["userId"]).apply(lambda x: x.sort_values(["timestamp"], ascending = True)).reset_index(drop=True)
    
    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data_orig, min_sc=50)

elif args.dataset_name == "ml-20m":
    raw_data_orig = pd.read_csv(args.base_dir+'ratings.csv', sep=',', header=0)
    # binarize the data (only keep ratings >= 4)
    raw_data_orig = raw_data_orig[raw_data_orig['rating'] > 3.5]
    print(raw_data_orig)
    #raw_data_orig = pd.read_csv(args.rating_file, sep=',', header=None, columns=['userId','artistId','albumId','movieId','timestamp'])
    
    max_seq_len = 1000
    # Remove users with greater than $max_seq_len number of watched movies
    raw_data_orig = raw_data_orig.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)
    
    # Sort data values with the timestamp
    #raw_data_orig = raw_data_orig.groupby(["userId"]).apply(lambda x: x.sort_values(["timestamp"], ascending = True)).reset_index(drop=True)
    
    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data_orig, min_uc=10, min_sc=10)

sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index

# shuffle user index
np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = int(0.1*len(raw_data.userId.unique()))

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
unique_sid = pd.unique(train_plays['movieId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
item_mapper = pd.DataFrame.from_dict(show2id, orient='index', columns=['new_movieId'])
item_mapper['movieId'] = item_mapper.index
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
user_mapper = pd.DataFrame.from_dict(profile2id, orient='index', columns=['new_userId'])
user_mapper['userId'] = user_mapper.index

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
vad_plays_tr, vad_plays_te, vad_plays_raw = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
test_plays_tr, test_plays_te, test_plays_raw = split_train_test_proportion(test_plays)

train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
vad_data = numerize(vad_plays_raw, profile2id, show2id)
vad_data.to_csv(os.path.join(pro_dir, 'validation.csv'), index=False)
test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
test_data = numerize(test_plays_raw, profile2id, show2id)
test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)

def long_tail_categ(value,q1,q2):
    if value >= q1 : return 0
    elif value < q1 and value >= q2 : return 1
    elif value < q2  : return 2

def find_quantile(df):
    qt1 = 0
    idx1=0
    for i in range(0,df.shape[0]):
        tot = df['counts'].sum()*0.2
        q1 = df.iloc[:i,:]['counts'].sum()
        if q1 >= tot:
            qt1 = df.iloc[i,:]['counts']
            idx1 = i
            break
    qt2 = 0
    for j in range(0,df.shape[0]):
        tot = df['counts'].sum()*0.8
        q2 = df.iloc[:j,:]['counts'].sum()
        #q2 = df.loc[df['counts'] < j]['counts'].sum()
        if q2 >= tot:
            qt2 = df.iloc[j,:]['counts']
            idx2 = j
            break
    return qt1, qt2, idx1, idx2


def build_dataset_categ(data, categ_map,item_col):
    categ_map = categ_map.filter(items=[item_col,'categ','counts'])
    return pd.merge(data, categ_map, on=item_col, how='inner')


# ITEM MAPPER
item_categ = pd.DataFrame(raw_data_orig['movieId'].value_counts())
item_categ.columns = ['counts']
item_categ['movieId'] = item_categ.index
item_categ = item_categ.sort_values(['counts'], ascending=False)
item_categ = item_categ.reset_index()
qt1,qt2, idx1, idx2 = find_quantile(item_categ)
print("QUANTILES: {},{}, INDEXES: {},{}".format(qt1,qt2,idx1,idx2) )

item_categ['categ'] = item_categ.apply (lambda row: long_tail_categ(row['counts'],qt1,qt2), axis=1)
item_categ['idx'] = item_categ.index

item_mapper_categ = item_categ.merge(item_mapper, how='inner', on='movieId')
item_mapper_categ.to_csv(os.path.join(args.out_data_dir, 'item_mapper.csv'), index=False)

# ITEM MAPPER TRAIN
train_data_full = raw_data.loc[raw_data.userId.isin(np.concatenate((tr_users, vd_users), axis=None))]
item_categ = pd.DataFrame(train_data_full['movieId'].value_counts())
item_categ.columns = ['counts']
item_categ['movieId'] = item_categ.index
item_categ = item_categ.sort_values(['counts'], ascending=False)
item_categ = item_categ.reset_index()
qt1,qt2, idx1, idx2 = find_quantile(item_categ)
print("QUANTILES: {},{}, INDEXES: {},{}".format(qt1,qt2,idx1,idx2) )

item_categ['categ'] = item_categ.apply (lambda row: long_tail_categ(row['counts'],qt1,qt2), axis=1)
item_categ['idx'] = item_categ.index

item_mapper_categ = item_categ.merge(item_mapper, how='inner', on='movieId')
item_mapper_categ.to_csv(os.path.join(args.out_data_dir, 'item_mapper_train.csv'), index=False)

# USER PROFILE
user_categ = pd.DataFrame(raw_data_orig['userId'].value_counts())
user_categ.columns = ['counts']
user_categ['userId'] = user_categ.index
user_categ = user_categ.sort_values(['counts'], ascending=False)
user_categ = user_categ.reset_index()
qt1,qt2, idx1, idx2 = find_quantile(user_categ)
print("QUANTILES: {},{}, INDEXES: {},{}".format(qt1,qt2,idx1,idx2) )

user_categ['categ'] = user_categ.apply(lambda row: long_tail_categ(row['counts'],qt1,qt2), axis=1)
user_categ['idx'] = user_categ.index

user_mapper_categ = user_categ.merge(user_mapper, how='inner', on='userId')
user_mapper_categ.to_csv(os.path.join(args.out_data_dir, 'user_mapper.csv'), index=False)


