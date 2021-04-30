# -*- coding: utf-8 -*-

import datetime
import math
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import torch
from torch.autograd import Variable
import torch.nn.functional as f

import utils
from utils import AverageMeter
import tqdm
import pandas as pd
from scipy.optimize import curve_fit

class Trainer(object):

    def __init__(self, cmd, cuda, model, optim=None,
                 train_loader=None, valid_loader=None, test_loader=None, log_file=None,
                 interval_validate=1, lr_scheduler=None, dataset_name = None,
                 start_step=0, total_steps=1e5, beta=0.05, start_epoch=0, bias=False, alpha = 0.0,
                 total_anneal_steps=200000, anneal_cap=0.1, do_normalize=True, item_mapper = None, user_mapper = None,
                 checkpoint_dir=None, result_dir=None, print_freq=1, result_save_freq=1, checkpoint_freq=1):

        self.cmd = cmd
        self.cuda = cuda
        self.model = model
        self.item_mapper = item_mapper
        self.user_mapper = user_mapper
        self.dataset_name = dataset_name
        self.bias = bias
        self.alpha = alpha

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap


        self.ndcg, self.recall = [], []
        self.loss, self.kl, self.posb, self.popb = [],[],[],[]
        self.att, self.rel, self.cnt, self.pcount = [],[],[],[]
        
        # sanity check
        #self.count_weight = torch.from_numpy(np.ones(item_mapper.shape[0])).type('torch.FloatTensor')


    def validate(self, cmd="valid", k=100):
        assert cmd in ['valid', 'test']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        end = time.time()

        n10_list, n100_list, r10_list, r100_list = [], [], [], []
        preds = []
        embs_list = []
        att_round, rel_round, cnt_round, pcount_round = [], [], [], []

        loader_ = self.valid_loader if cmd == 'valid' else self.test_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, uindex) in tqdm.tqdm(enumerate(loader_), total=len(loader_),
                                   desc='{} check epoch={}, len={}'.format('Valid' if cmd == 'valid' else 'Test',
                                                               self.epoch, len(loader_)), ncols=80, leave=False):
            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
            data_tr = Variable(data_tr)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr)
                #POP
                #pop_values = self.item_mapper.sort_values(['new_movieId']).counts.values
                #pop_values = np.tile(pop_values, (logits.shape[0],1)).astype(np.float)
                #logits = torch.tensor(pop_values).cuda()
                
                pred_val = logits.cpu().detach().numpy()
                pred_val[data_tr.cpu().detach().numpy().nonzero()] = -np.inf

                data_te_csr = sparse.csr_matrix(data_te.numpy())
                n10_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=10))
                n100_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=100))
                r10_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=10))
                r100_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=100))

                #cnt, pcount = utils.att_rel(f.softmax(logits,dim=1), data_tr, self.count_norm, self.play_count, self.cuda, k=k)
                cnt, pcount = utils.att_rel(logits, data_tr, self.count_norm, self.play_count, self.cuda, k=k)
                pcount_round.append(pcount)
                cnt_round.append(cnt)
                #udx_list.append(uindex + batch_idx*pred_val.shape[0])
 
                embs_list.append(mu_q.cpu().detach().numpy())

        n10_list = np.concatenate(n10_list, axis=0)
        n100_list = np.concatenate(n100_list, axis=0)
        r10_list = np.concatenate(r10_list, axis=0)
        r100_list = np.concatenate(r100_list, axis=0)
        cnt_round = np.concatenate(cnt_round, axis=0)
        pcount_round = np.concatenate(pcount_round, axis=0)
        embs_list = np.concatenate(embs_list, axis=0)

        metrics = []
        if cmd == 'valid':

            self.ndcg.append(np.mean(n100_list))
            self.recall.append(np.mean(r100_list))
            self.cnt.append(cnt_round)
            self.pcount.append(np.mean(np.sum(cnt_round, axis=1)))
            popb = utils.calc_pop_bias(cnt_round)
            self.popb.append(np.mean(popb))

            np.save('results/'+self.dataset_name+'_n10_'+str(self.bias)+'_'+str(self.alpha)+'.npy', self.ndcg)
            np.save('results/'+self.dataset_name+'_r10_'+str(self.bias)+'_'+str(self.alpha)+'.npy', self.recall)
            np.save('results/{}_pcounts_{}_{}.npy'.format(self.dataset_name, str(self.bias), str(self.alpha)), self.pcount)
            np.save('results/'+self.dataset_name+'_popb_'+str(self.bias)+'_'+str(self.alpha)+'.npy', np.array(self.popb))

            # SAVE MODEL
            with open(self.checkpoint_dir+self.dataset_name+'_vae_'+str(self.bias)+'_'+str(self.alpha)+'.pt', 'wb') as model_file: torch.save(self.model, model_file)
            #torch.save({'state_dict': self.model.state_dict()}, self.checkpoint_dir+'vae')

            #self.ufair.append(ufair)
            #metrics.append(max_metrics)
            metrics.append("NDCG@10,{:.5f},{:.5f}".format(np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
            metrics.append("NDCG@100,{:.5f},{:.5f}".format(np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
            metrics.append("Recall@10,{:.5f},{:.5f}".format(np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
            metrics.append("Recall@100,{:.5f},{:.5f}".format(np.mean(r100_list), np.std(r100_list) / np.sqrt(len(r100_list))))
            metrics.append("POPB@{},{:.5f}".format(k,np.mean(popb)))
            print('\n' + ",\n".join(metrics))
            print("PLAY-COUNTS@{},{}".format(k,np.mean(np.sum(cnt_round, axis=1))))

        else:

            popb = utils.calc_pop_bias(cnt_round)
            
            metrics.append("NDCG@10,{:.5f},{:.5f}".format(np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
            metrics.append("NDCG@100,{:.5f},{:.5f}".format(np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
            metrics.append("Recall@10,{:.5f},{:.5f}".format(np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
            metrics.append("Recall@100,{:.5f},{:.5f}".format(np.mean(r100_list), np.std(r100_list) / np.sqrt(len(r100_list))))
            metrics.append("POPB@{},{:.5f}".format(k,np.mean(popb)))
            print('\n' + ",\n".join(metrics))
            print("PLAY-COUNTS@{},{}".format(k,np.mean(np.sum(cnt_round, axis=1))))

            #print(np.array(udx_list).shape)
            #print(np.array(n100_list).shape,np.array(r100_list).shape)
            #print(np.array(popb).shape,np.array(np.sum(cnt_round, axis=1)).shape)
            #df = pd.DataFrame()
            #df['uid'] = udx_list
            #df['recall'] = r100_list
            #df['ndcg'] = n100_list
            #df['popb'] = popb
            ##df['embs'] = embs_list
            ##print(len(n100_list), len(demo_list))
            #df['pcount'] = np.sum(cnt_round, axis=1)
            #df.to_csv('result_prof.csv', index=False)
        
            #np.save('embs.npy', embs_list)

        self.model.train()

    def train_epoch(self):
        cmd = "train"
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        for batch_idx, (data_tr, data_te, uidx) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train check epoch={}, len={}'.format(self.epoch, len(self.train_loader)), ncols=80, leave=False):
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                #prof = prof.cuda()
            data_tr = Variable(data_tr)
            #prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr)

            #log_soft_var = log_softmax_var * rep_tensor

            # LOSS
            if self.bias: 
                rep_tensor = self.count_weight.repeat(logits.shape[0]).view(logits.shape).to('cuda')
                #logits = logits*rep_tensor
                log_softmax_var = f.log_softmax(logits, dim=1)
                #neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))
                neg_ll = - torch.mean(torch.sum(rep_tensor*(log_softmax_var * data_tr), dim=1))
            else: 
                log_softmax_var = f.log_softmax(logits, dim=1)
                neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))


            l2_reg = self.model.get_l2_reg()

            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.step / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            loss = neg_ll + self.anneal * KL + l2_reg 

            # backprop
            self.model.zero_grad()
            loss.backward()
            self.optim.step()


    def func_powerlaw_trunc(self, x, a, b, c):
        return (a + b*x)**(-c)

    def calc_fit_params(self, data):
        x = np.array(range(len(data)))
        y = [float(i)/np.sum(data) for i in data]
        params,_ = curve_fit(self.func_powerlaw_trunc, x, y, maxfev=10000)
        return params

    def build_count_weight(self):
        #item_mapper_sort = self.item_mapper.sort_values(['counts'],ascending=False)
        item_mapper_sort = self.item_mapper.sort_values(['new_movieId'])
        #item_mapper_sort["count_weight"] = np.ones(item_mapper_sort.shape[0])-(item_mapper_sort['counts'].values/np.max(item_mapper_sort['counts'].values))
        item_mapper_sort["count_norm"] = item_mapper_sort['counts'].values/np.sum(item_mapper_sort['counts'].values)
        
        if self.dataset_name == "ml-20m":
            if self.bias and self.alpha > 0.0:
                #params = [166.82, 2.89, 1.06]
                params = self.calc_fit_params(self.item_mapper.sort_values(['counts'],ascending=False)['counts'].values)
                print(params)

                item_mapper_sort = item_mapper_sort.sort_values(['counts'],ascending=False)
                item_mapper_sort['count_weight'] = self.func_powerlaw_trunc(np.array(range(item_mapper_sort.shape[0])),params[0],params[1],self.alpha*params[2])
                #item_mapper_sort['count_weight'] = np.ones(item_mapper_sort.shape[0]) - item_mapper_sort['count_weight'].values/np.max(item_mapper_sort['count_weight'].values)
             
                count_weight = item_mapper_sort['count_weight'].values 
                # subtract min and  divide by (max - min)
                count_weight = (count_weight - np.min(count_weight))/(np.max(count_weight) - np.min(count_weight))
                # take the complement (1-x)
                item_mapper_sort['count_weight'] = np.ones(item_mapper_sort.shape[0]) - count_weight
            else:
                #item_mapper_sort["count_weight"] = np.ones(item_mapper_sort.shape[0])-(item_mapper_sort['counts'].values/np.max(item_mapper_sort['counts'].values))
                count_weight = item_mapper_sort['counts'].values
                # subtract min and  divide by (max - min)
                count_weight = (count_weight - np.min(count_weight))/(np.max(count_weight) - np.min(count_weight))
                item_mapper_sort["count_weight"] = np.ones(item_mapper_sort.shape[0])- count_weight

        elif self.dataset_name == "netflix":
            if self.bias and float(self.alpha) > 0.0:
                #params = [46.24, 0.14, 1.6]
                params = self.calc_fit_params(self.item_mapper.sort_values(['counts'],ascending=False)['counts'].values)
                print(params)
                item_mapper_sort = item_mapper_sort.sort_values(['counts'],ascending=False)
                item_mapper_sort['count_weight'] = self.func_powerlaw_trunc(np.array(range(item_mapper_sort.shape[0])),params[0],params[1],self.alpha*params[2])
                #item_mapper_sort['count_weight'] = np.ones(item_mapper_sort.shape[0]) - item_mapper_sort['count_weight'].values/np.max(item_mapper_sort['count_weight'].values)
                
                count_weight = item_mapper_sort['count_weight'].values 
                # subtract min and  divide by (max - min)
                count_weight = (count_weight - np.min(count_weight))/(np.max(count_weight) - np.min(count_weight))
                # take the complement (1-x)
                item_mapper_sort['count_weight'] = np.ones(item_mapper_sort.shape[0]) - count_weight
            else:
                count_weight = item_mapper_sort['counts'].values
                # subtract min and  divide by (max - min)
                count_weight = (count_weight - np.min(count_weight))/(np.max(count_weight) - np.min(count_weight))
                item_mapper_sort["count_weight"] = np.ones(item_mapper_sort.shape[0])- count_weight

        item_mapper_sort = item_mapper_sort.sort_values(['new_movieId'])
        self.count_weight = torch.tensor(item_mapper_sort["count_weight"]).type('torch.FloatTensor')
        self.count_norm = torch.tensor(item_mapper_sort["count_norm"]).type('torch.FloatTensor')
        self.play_count = torch.tensor(item_mapper_sort['counts'].values).type(torch.FloatTensor)

    def train(self):

        self.build_count_weight()
        #self.plot_count_weight()
        max_epoch = 200
        for epoch in tqdm.trange(0, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            self.lr_scheduler.step()
            self.validate(cmd='valid')
            #self.validate(cmd='test')
    
    def test(self):
        self.build_count_weight()
        self.validate(cmd='test')
