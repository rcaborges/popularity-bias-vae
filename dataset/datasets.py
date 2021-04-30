#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import time
import utils
import glob
import tqdm
from scipy import sparse
import pandas as pd

class MovieLensDataset(data.Dataset):
    def __init__(self, root, fold_in=True, split='train', conditioned_on=None, upper=-1):
        super(MovieLensDataset, self).__init__()
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        #assert len(img_files) > 0
        #self.img_files = img_files
        assert os.path.exists(root), "root: {} not found.".format(root)

        assert split in ['test', 'inference', 'train', 'valid']
        #fname = os.path.join(root, self.img_files)

        #self.train_data, self.vad_data_tr, self.vad_data_te, self.test_data_tr, self.test_data_te = utils.load_weights_pkl(fname)

        out_data_dir = root

        unique_sid = list()
        with open(os.path.join(out_data_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())

        n_items = len(unique_sid)

        self.train_data = utils.load_train_data(os.path.join(out_data_dir, 'train.csv'), n_items)
        self.vad_data_tr, self.vad_data_te = utils.load_tr_te_data(os.path.join(out_data_dir, 'validation_tr.csv'), os.path.join(out_data_dir, 'validation_te.csv'), n_items)
        self.test_data_tr, self.test_data_te = utils.load_tr_te_data(os.path.join(out_data_dir, 'test_tr.csv'), os.path.join(out_data_dir, 'test_te.csv'), n_items)

        assert self.train_data.shape[1] == self.vad_data_tr.shape[1]
        assert self.vad_data_tr.shape == self.vad_data_te.shape
        assert self.test_data_tr.shape == self.test_data_te.shape

        self.split = split
        self.fold_in = fold_in

        if self.split == 'train':
            self.n_users = self.train_data.shape[0]
        elif self.split == 'valid':
            self.n_users = self.vad_data_tr.shape[0]
        elif self.split == 'test':
            self.n_users = self.test_data_tr.shape[0]
        else:
            raise NotImplementedError

        self.n_users = self.n_users if upper <= 0 else min(self.n_users, upper)

    def __len__(self):
        return self.n_users

    def __getitem__(self, index):
        #prof = np.zeros(1)
        if self.split == 'train':
            data_tr, data_te = self.train_data[index], np.zeros(1)
        elif self.split == 'valid':
            data_tr, data_te = self.vad_data_tr[index], self.vad_data_te[index]
        elif self.split == 'test':
            data_tr, data_te = self.test_data_tr[index], self.test_data_te[index]

        if sparse.isspmatrix(data_tr):
            data_tr = data_tr.toarray()
        data_tr = data_tr.astype('float32')
        data_tr = data_tr[0]

        if sparse.isspmatrix(data_te):
            data_te = data_te.toarray()
        data_te = data_te.astype('float32')
        data_te = data_te[0]

        return data_tr, data_te, index


class NetflixDataset(data.Dataset):

    def __init__(self, root, fold_in=True, split='train', upper=-1):

        super(NetflixDataset, self).__init__()
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        assert os.path.exists(root), "root: {} not found.".format(root)

        assert split in ['test', 'inference', 'train', 'valid']

        out_data_dir = root

        unique_sid = list()
        with open(os.path.join(out_data_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())

        n_items = len(unique_sid)

        self.train_data = utils.load_train_data(os.path.join(out_data_dir, 'train.csv'), n_items)
        self.vad_data_tr, self.vad_data_te = utils.load_tr_te_data(os.path.join(out_data_dir, 'validation_tr.csv'), os.path.join(out_data_dir, 'validation_te.csv'), n_items)
        self.test_data_tr, self.test_data_te = utils.load_tr_te_data(os.path.join(out_data_dir, 'test_tr.csv'), os.path.join(out_data_dir, 'test_te.csv'), n_items)

        assert self.train_data.shape[1] == self.vad_data_tr.shape[1]
        assert self.vad_data_tr.shape == self.vad_data_te.shape
        assert self.test_data_tr.shape == self.test_data_te.shape

        self.split = split
        self.fold_in = fold_in

        if self.split == 'train':
            self.n_users = self.train_data.shape[0]
        elif self.split == 'valid':
            self.n_users = self.vad_data_tr.shape[0]
        elif self.split == 'test':
            self.n_users = self.test_data_tr.shape[0]
        else:
            raise NotImplementedError

        self.n_users = self.n_users if upper <= 0 else min(self.n_users, upper)

    def __len__(self):
        return self.n_users

    def __getitem__(self, index):
        #prof = np.zeros(1)
        if self.split == 'train':
            data_tr, data_te = self.train_data[index], np.zeros(1)
        elif self.split == 'valid':
            data_tr, data_te = self.vad_data_tr[index], self.vad_data_te[index]
        elif self.split == 'test':
            data_tr, data_te = self.test_data_tr[index], self.test_data_te[index]

        if sparse.isspmatrix(data_tr):
            data_tr = data_tr.toarray()
        data_tr = data_tr.astype('float32')
        data_tr = data_tr[0]

        if sparse.isspmatrix(data_te):
            data_te = data_te.toarray()
        data_te = data_te.astype('float32')
        data_te = data_te[0]

        return data_tr, data_te
