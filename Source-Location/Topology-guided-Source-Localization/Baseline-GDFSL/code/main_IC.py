import sys

import args
import torch.nn as nn
import numpy as np
import setproctitle

import torch
import random
import networkx as nx
import heapq
import pickle
import time
from tqdm import tqdm
import math
import torch.nn.functional as F

from dataReader import DataReader_snapshot
from dataLoader import GraphData
from torch.utils.data import DataLoader
from model import GaussianDiffusionForwardTrainer_future, GaussianDiffusionSampler
from model import GaussianDiffusionForwardTrainer_future_un, GaussianDiffusionSampler_un
torch.cuda.set_device(0)



def collate_batch(batch):  
    B = len(batch)  
    Chanels = batch[0][-2].shape[1]  
    N_nodes_max = batch[0][-2].shape[0]
    x = batch[0][-2]

    A = batch[0][0]
    P = batch[0][2]
    labels = batch[0][1]

    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    influence = batch[0][4]
    
    return [A, labels, P, x, N_nodes, influence]  

def train_noise_un(train_loader, args):
    
    args.device = 'cuda'

    start = time.time()
    train_loss, n_samples = 0, 0
    for batch_idx, data in enumerate(train_loader): 
        opt.zero_grad()
        final_miu = torch.ones(data[-2], 1).cuda() + data[2].unsqueeze(-1).float().cuda()
        t = torch.randint(args.T-forward_Trainner.t_start, size=(1,)).cuda()      
        x_t, noise = forward_Trainner.forward_model_IC(data[2].unsqueeze(1).cuda(), t, final_miu, data[0].cuda())

        x_t = x_t.float()
        noise = noise.float()
        pred_noise = forward_Trainner.model(x_t, t, data[3][:, 7:].cuda(), data[0].cuda(), data[3].cuda()).float()  

        loss = F.mse_loss(pred_noise.cuda(), noise.cuda()).float().cuda()
        loss.backward()
        opt.step()

        time_iter = time.time() - start
        train_loss += loss.item() * 1
        n_samples += 1
        
        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            print('Noise Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                time_iter / (batch_idx + 1)))

def sample_and_test(train_loader, args):
    global F_score_global, exe_time
    args.device = 'cuda'
    args.test_sample = 300

    start = time.time()
    
    n_samples = 0
    sampler.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader): 
            if batch_idx > args.test_sample:
                break
            final_miu = torch.ones(data[-2], 1).cuda() + data[2].unsqueeze(-1).float().cuda()
            x_T = torch.randn_like(final_miu).cuda() + final_miu.cuda()  
            pred_x_0 = sampler(x_T, final_miu, data[3][:, 7:].cuda(), data[0].cuda(), data[3].cuda(), observation=data[2].unsqueeze(-1).float().cuda())  
        
            pred_x_0 = pred_x_0.squeeze(-1).cuda()
            true_label_indices = torch.where(data[1] == 1)[0].cuda()
            true_label_predictions = pred_x_0[true_label_indices].cuda()
            sorted_indices = torch.argsort(pred_x_0, descending=True).cuda()
            
            ranks = torch.zeros_like(pred_x_0, dtype=torch.long).cuda()
            ranks[sorted_indices] = torch.arange(1, len(pred_x_0) + 1).cuda()  
            
            true_label_ranks = ranks[true_label_indices][0].cuda()
            F_score_global += 2*1*(1/true_label_ranks)/(1+1/true_label_ranks)

            time_iter = time.time() - start
            n_samples += 1
            exe_time += 1
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:  
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                    epoch + 1, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), F_score_global / exe_time, time_iter / (batch_idx + 1)))
    return F_score_global / exe_time


with open('3a_test_100persent_not_extended.pkl', 'rb') as f:
    all_propagation_Twitter = pickle.load(f)
all_propagation_Twitter_test = []
for prop in all_propagation_Twitter:
    if len(prop.nodes) > 300:
        continue
    else:
        all_propagation_Twitter_test.append(prop)
with open('networkx_Twitter_union_graph_inf.pkl', 'rb') as f:
    Twitter_union_graph_inf = pickle.load(f)

rnd_state = np.random.RandomState(1111)
datareader_Twitter = DataReader_snapshot(all_propagation_Twitter_test,
                                rnd_state=rnd_state,
                                folds=10,
                                union_graph_inf = Twitter_union_graph_inf)

flag = True
n_folds = 10
for fold_id in range(n_folds):
    loaders_Twitter = []

    for split in ['train', 'test']:
        gdata_Twitter = GraphData(fold_id=fold_id,
                                  datareader=datareader_Twitter,  
                                  split=split)

        loader_Twitter = DataLoader(gdata_Twitter,  
                                    batch_size=1,  
                                    shuffle=True,  
                                    num_workers=4,
                                    collate_fn=collate_batch)  
        loaders_Twitter.append(
            loader_Twitter)

    print('\nDatasets: FOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders_Twitter[0].dataset),
                                                            len(loaders_Twitter[1].dataset)))

    t_start = int(1 / 10 * args.T)  

    forward_Trainner = GaussianDiffusionForwardTrainer_future_un(t_start=t_start, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T, feature_hidden_dim=64).cuda()
    sampler = GaussianDiffusionSampler_un(model=forward_Trainner, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).cuda()

    opt = torch.optim.Adam([
        {'params': forward_Trainner.parameters(), 'lr': 0.0005},
      
    ])

    all_acc = []
    for epoch in range(args.epochs):
        if flag == True:
            F_score_global = 0
            exe_time = 0
            train_noise_un(loaders_Twitter[0], args)
            if (epoch+1) % 1 == 0:
                g_loss = sample_and_test(loaders_Twitter[0], args)
                all_acc.append(g_loss)
                if len(all_acc) >= 5:
                    if all_acc[-1] < all_acc[-2] and all_acc[-1] < all_acc[-3] and all_acc[-1] < all_acc[-4] and all_acc[-1] < all_acc[-5]:
                        print("best F-score: ", max(all_acc))
                        break

