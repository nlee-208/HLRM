
import os
import time
import json
import copy
import argparse
import logging

import numpy as np
from tqdm.auto import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from _utils import df_to_mat, mat_to_dict, str2bool
from loader import fetch_data
from model import HLRM
from generator import BatchGenerator


def get_args():
    """ retrieve arguments for training """
    parser = argparse.ArgumentParser(description="Args for HLRM Training")

    parser.add_argument('--train_batch_size',
                    default = 5000,
                    type = int,
    )
    parser.add_argument('--eval_batch_size',
                    default = 10000,
                    type = int,
    )
    parser.add_argument('--num_inter',
                    default = 50,
                    type = int,
    )
    parser.add_argument('--hlrm_type',
                    default = 'base',
                    choices = ['base', 'plus'],
                    type = str,
    )
    parser.add_argument('--epochs',
                    default = 1,
                    type = int,
    )
    parser.add_argument('--lr',
                    default = 0.00002,
                    type = float,
    )
    parser.add_argument('--lr_scheduler_gamma',
                    default = 0.98,
                    type = float,
    )
    parser.add_argument('--clip',
                    default = 3,
                    type = int,
    )
    parser.add_argument('--patience',
                    default = 10,
                    type = int,
    )
    parser.add_argument('--emb_size',
                    default = 100,
                    type = int,
    )    
    parser.add_argument('--num_relations',
                    default = 20,
                    type = int,
    )        
    parser.add_argument('--is_pretrained_embs',
                    default = False,
                    type = str2bool,
                    help = 'boolean flag'
    )
    parser.add_argument('--freeze_embs',
                    default = False,
                    type = str2bool,
                    help = 'boolean flag'
    )
    parser.add_argument('--save_model',
                    default = True,
                    type = str2bool,
                    help = 'boolean flag'
    )
    parser.add_argument('--save_dir',
                    default = 'hlrm_base.pt',
                    type = str,
    )
    parser.add_argument('--loss_margin',
                    default = 0.2,
                    type = float,
    )
    args = parser.parse_args()
    return args


def loss_func_triplet(user,item_p,item_n):
    return F.triplet_margin_loss(user,item_p,item_n, margin = args.loss_margin)

def print_metrics(split, metrics):
    print("%s loss: %.6f "% (split, metrics['loss']))


def run_epoch(model, generator, optim, clip, split):
    # Model mode
    model.determine_mode(split)

    loss_sum,cnt = 0,0
    st_time = time.time()

    # Iterate data
    for batch_num in range(generator[split].n_batches-1):
        batch = generator[split]._batch_triplets(batch_num)
        if split == 'train':
            optim.zero_grad()
          
            user,item_p,item_n,rel_pos,rel_neg = model(*batch)
            loss_triplet = loss_func_triplet(user,(item_p-rel_pos),(item_n-rel_neg))
            loss = loss_triplet
          
            loss.backward()

            if clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optim.step()
        else:
            with torch.no_grad():
                user,item_p,item_n,rel_pos,rel_neg = model(*batch)
                loss_triplet = loss_func_triplet(user,(item_p-rel_pos),(item_n-rel_neg))
                loss = loss_triplet
#         print(split, batch_num)
                
        loss_sum +=  loss.item()
        cnt += 1
        
        if cnt % 300 == 0:
            print(f'Elapsed: {time.time() - st_time:.2f} for Step #{batch_num+1}')

    # Loss
    loss_mean = loss_sum / cnt
#     print(f'Elapsed: {time.time() - st_time:.2f} for Epoch #{batch_num})
    
    return {'loss': loss_mean}


def train_model(model,generator, max_epoch, optim_lr,gamma = 0.98, clip = 3, finish_improvement_fail_cnt = 10):

    best_val_metric = None
    selected_results = dict()
    improvement_fail_cnt = 0

    splits = ['train', 'val']
    early_stop_split = 'val'

    metric_names = ['loss']
    
    my_model = model.to('cuda')
    save_model = model
    
    # optim = torch.optim.SGD([param for param in my_model.parameters() if param.requires_grad == True], lr=optim_lr, momentum = 0.9)
    optim = torch.optim.Adam([param for param in my_model.parameters() if param.requires_grad == True], lr=optim_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    st_time = time.time()

    
    for epoch in range(max_epoch):
        print("-" * 80)
        print("Epoch: %d, Global Epoch: %d" % (epoch, my_model.global_step))

        split2metrics = dict()
        results = dict()

        for split in splits:
            print(split)
            metrics = run_epoch(my_model,generator, optim, clip, split)
            split2metrics[split] = metrics
            print_metrics(split, metrics)
            results = {**results,
                       **{'%s__%s' % (split, name): float(metrics[name]) for name in metric_names}}
        lr_scheduler.step()
        my_model.global_step += 1

        if best_val_metric is None or best_val_metric > split2metrics[early_stop_split]['loss']:
            print("Best!")
            best_val_metric = split2metrics[early_stop_split]['loss']
            selected_results = results
            improvement_fail_cnt = 0

            save_model.load_state_dict(my_model.state_dict())
        else:
            improvement_fail_cnt += 1

        print("Elapsed: %.3f" % (time.time() - st_time))
        
        if improvement_fail_cnt >= finish_improvement_fail_cnt:
            break
    
    total_duration = (time.time() - st_time)

    return save_model, selected_results, total_duration

##########################################
if __name__ == '__main__':
    args = get_args()

    with open('configs.json', 'rb') as f:
        params = json.load(f)

    processed_data = fetch_data(params)

    train_inter = processed_data['train_interactions']
    valid_inter = processed_data['valid_interactions']
    test_inter = processed_data['test_interactions']

    train_inter_dict = (mat_to_dict(train_inter))
    valid_inter_dict = (mat_to_dict(valid_inter))
    test_inter_dict = (mat_to_dict(test_inter))

    train_valid_inter_dict = copy.deepcopy(train_inter_dict)
    for k,v in valid_inter_dict.items():
        if k in train_valid_inter_dict.keys():
            train_valid_inter_dict[k] += v

    generator = {}
    generator['train'] = BatchGenerator(
                    interactions = train_inter,
                    batch_size = args.train_batch_size,
                    num_negatives = 1,
                    max_num_prefs = args.num_inter,
                    )
    generator['val'] = BatchGenerator(
                    interactions = valid_inter,
                    batch_size = args.eval_batch_size,
                    num_negatives = 1,
                    max_num_prefs = args.num_inter,
                    interactions_eval = train_inter_dict
                    )

    model = HLRM(
                args.emb_size,
                args.num_relations,
                processed_data['n_users'],
                processed_data['n_items'],
                pretrained_embs = args.is_pretrained_embs,
                # user_embs = args.pretrained_user_embs,
                # item_embs = args.pretrained_item_embs,
                model_type = args.hlrm_type,
                )

    if args.freeze_embs:
        model.user_emb.weight.requires_grad = False
        model.item_emb.weight.requires_grad = False

    save_model, selected_results,total_duration = train_model(model, generator, args.epochs, args.lr, 
                                                            gamma = args.lr_scheduler_gamma, clip = args.clip, 
                                                            finish_improvement_fail_cnt= args.patience)

    if args.save_model:
        torch.save(save_model.state_dict(), args.save_dir)

    print(save_model, selected_results, total_duration)




