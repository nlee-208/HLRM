import os
import time
import json
import copy
import argparse
import logging
import random

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



parser = argparse.ArgumentParser()


def get_args():
    """ retrieve arguments for evaluation """
    parser = argparse.ArgumentParser(description="Args for HLRM Model Evaluation")

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
    parser.add_argument('--save_dir',
                    default = 'hlrm_base.pt',
                    type = str,
    )
    parser.add_argument('--eval_size',
                    default = 1000,
                    type = int,
    )
    parser.add_argument('--emb_size',
                    default = 100,
                    type = int,
    )
    parser.add_argument('--topk',
                    default = 10,
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
    args = parser.parse_args()
    return args

class Metric:
    """
    Abstract class for a recommendation metric
    """
    def __init__(self, k, rank_method='local_rank'):
        self.k = k
        self.rank_method = rank_method

    def eval(self, reco_items, ref_user_items):
        """
        Abstract
        :param reco_items:
        :param ref_user_items:
        :return:
        """
        raise NotImplementedError(
            'eval method should be implemented in concrete model')

class HITRATE(Metric):
    """
    recall@k score metric.
    """
    def __str__(self):
        return f'hit_rate@{self.k}'

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K hit_rate
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: hit_rate@k
        """

        hit_rate = []
        for user_id, tops in reco_items.items():
            # remove first element of the sequence
            ref_set = ref_user_items.get(user_id, [])
            user_hits = np.array([1 if it in ref_set else 0 for it in tops],
                                 dtype=np.float32)
            hit_rate.append(float(np.sum(user_hits[:self.k])) / len(ref_set))
        return np.mean(hit_rate)

class MAP(Metric):
    """
    map@k score metric.
    """
    def __str__(self):
        return f'map@{self.k}'

    @classmethod
    def precision_at_k(cls, r, k):
        """
        :param r:
        :param k:
        :return:
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    @classmethod
    def ap(cls, r):
        """
        Average precision
        :param r:
        :return:
        """
        r = np.asarray(r) != 0
        out = [cls.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.sum(out) / len(r)

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K MAP for a particular user given the predicted scores
        to items.
        :param reco_items: reco items dictionary (contains also metadata
                                                  necessary, e.g. art_ids)
        :param ref_user_items:
        :return: map@k
        """
        map_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            map_metric.append(self.ap(user_hits[:self.k]))
        return np.mean(map_metric)

class MRR(Metric):
    """
    mrr@k score metric.
    """
    def __str__(self):
        return f'mrr@{self.k}'

    @classmethod
    def mrr_at_k(cls, user_hits, k):
        assert k >= 1
        user_hits = np.asarray(user_hits)[:k] != 0
        res = 0.
        for index, item in enumerate(user_hits):
            if item == 1:
                res += 1 / (index + 1)
        return res

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K MRR for a particular user given the predicted scores
        to items.
        :param reco_items: reco items dictionary (contains also metadata
                                                  necessary, e.g. art_ids)
        :param ref_user_items:
        :return: map@k
        """
        mrr_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            mrr_metric.append(self.mrr_at_k(user_hits, self.k))
        return np.mean(mrr_metric)

class NDCG(Metric):
    """
    nDCG@k score metric.
    """
    @classmethod
    def dcg_at_k(cls, r, k):
        """
        Discounted Cumulative Gain calculation method
        :param r:
        :param k:
        :return: float, DCG value
        """
        assert k >= 1
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def eval(self, reco_items, ref_user_items):
        local_res = []
        global_res = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            local_ideal_rels = np.array(sorted(user_hits, reverse=True))
            global_ideal_rels = self._global_ideal_rels(ref_set)

            dcg_k = self.dcg_at_k(user_hits, self.k)
            local_ideal_dcg = self.dcg_at_k(local_ideal_rels, self.k)
            if local_ideal_dcg > 0.:
                loc_ndcg = dcg_k / local_ideal_dcg
                local_res.append(loc_ndcg)

            global_ideal_dcg = self.dcg_at_k(global_ideal_rels, self.k)
            if global_ideal_dcg > 0.:
                glob_ndcg = dcg_k / global_ideal_dcg
                global_res.append(glob_ndcg)

        loc_ndcg = np.mean(local_res) if len(local_res) > 0 else 0.
        glob_ndcg = np.mean(global_res) if len(global_res) > 0 else 0.
        return loc_ndcg, glob_ndcg

    def _global_ideal_rels(self, ref_set):
        if len(ref_set) >= self.k:
            ideal_rels = np.ones(self.k)
        else:
            ideal_rels = np.pad(np.ones(len(ref_set)),
                                (0, self.k - len(ref_set)),
                                'constant')
        return ideal_rels

    def __str__(self):
        return f'ndcg@{self.k}'

class PRECISION(Metric):
    """
    precision@k score metric.
    """
    def __str__(self):
        return f'precision@{self.k}'

    @classmethod
    def precision_at_k(cls, r, k):
        """
        Precision at k
        :param r:
        :param k:
        :return:
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        # return np.mean(r)
        return sum(r) / len(r)

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K PRECISION
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: precision@k
        """
        prec_metric = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            prec_metric.append(self.precision_at_k(user_hits, self.k))
        return np.mean(prec_metric)

class RECALL(Metric):
    """
    recall@k score metric.
    """
    def __str__(self):
        return f'recall@{self.k}'

    @classmethod
    def user_rec(cls, user_hits, ref_len, k):
        score = 0.0
        user_hits = np.asfarray(user_hits)[:k]
        sum_hits = np.sum(user_hits)

        # in the case where the list contains no hit, return score 0.0 directly
        if sum_hits == 0:
            return score
        return float(sum_hits) / ref_len

    def eval(self, reco_items, ref_user_items):
        """
        Compute the Top-K recall
        :param reco_items: reco items dictionary
        :param ref_user_items:
        :return: recall@k
        """
        recall = []
        for user_id, top_items in reco_items.items():
            ref_set = ref_user_items.get(user_id, set())
            user_hits = np.array([1 if it in ref_set else 0 for it in top_items],
                                 dtype=np.float32)
            recall.append(self.user_rec(user_hits, len(ref_set), self.k))
        return np.mean(recall)

def get_new_users(model,user_ind,generator,num_items, emb_size = 100):
    cnt = 0
    # Iterate data
    embs = torch.empty(size=(0,emb_size))

    model.to('cuda')
    model.eval()
    
    for batch_num in range(generator.n_batches):
        batch = generator._batch_test_sample_per_user(user_ind,num_items,batch_num)
        with torch.no_grad():
            user_w_rel = model.inference(*batch).detach().cpu()
            embs = torch.vstack([embs,user_w_rel])
        torch.cuda.empty_cache()
    
    return torch.as_tensor(np.array(embs))

def recommend_single(user, item, topk = 10):
    """ 
    Recommend
    # sample size = 1
    user : user embeddings of size (n_items, emb_size)
    item : item embeddings of size (n_items, emb_size)
    
    return indexes of recomm_items of size (n_users,topk)
    """
    dist = F.pairwise_distance(user,item)
    dist,ind = torch.topk(dist, k=topk, largest = False)
    return ind

def recommend(model,eval_users,generator,item_embs, k,emb_size):
    recoms = {}
    for sample_user_id in tqdm(eval_users):
        users_ = get_new_users(model,sample_user_id, generator,num_items=item_embs.size(0), emb_size = emb_size)
        recoms[sample_user_id] = recommend_single(users_,item_embs)
    return recoms


##########################################
if __name__ == '__main__':
    args = get_args()

    with open('configs.json', 'rb') as f:
        params = json.load(f)
    params['dataset']['file_format'] = 'csv'

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

    test_generator = BatchGenerator(
                interactions = test_inter,
                batch_size = args.eval_batch_size,
                num_negatives = 1,
                max_num_prefs = args.num_inter,
                interactions_eval = train_valid_inter_dict,
                max_num_inter = args.num_inter
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

    model.load_state_dict(torch.load(args.save_dir))

    item_embs = model.item_emb.weight.detach().to('cpu')

    test_users_with_inter = set([k for k,v in test_inter_dict.items() if len(v) > 0])
    eval_users = random.sample(test_users_with_inter, args.eval_size) 

    topk = args.topk

    recoms = recommend(model, eval_users, test_generator, item_embs, topk, args.emb_size)

    prec = PRECISION(k=topk)
    ndcg = NDCG(k=topk)
    hit = HITRATE(k=topk)
    map_ = MAP(k=topk)
    rec = RECALL(k=topk)
    mrr = MRR(k=topk)

    print('HLRM Metrics for Implementations')
    for met in [map_,mrr,ndcg,prec,rec,hit]:
        print(met,met.eval(reco_items=recoms, ref_user_items = test_inter_dict))

