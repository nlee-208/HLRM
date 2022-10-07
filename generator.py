
import numpy as np
import torch

from _utils import mat_to_dict

class Sampler:
    """
    Negative sampler
    """
    def __init__(self, user_items, n_items,
                 n_negatives):
        """
        Initialize a new sampler
        :param n_items: total number of items in dataset
        :param n_negatives: number of negative
        """
        self.user_items = user_items
        self.n_items = n_items
        self.n_negatives = n_negatives

    def sampling(self, user_ids):
        """
        Negative sampling
        :param user_ids:
        :return:
        """
        neg_samples = np.random.choice(self.n_items,
                                       size=(len(user_ids), self.n_negatives))
        for i, uid, negatives in zip(range(len(user_ids)), user_ids,
                                     neg_samples):
            for j, neg in enumerate(negatives):
                while neg in self.user_items[uid]:
                    neg_samples[i, j] = neg = np.random.choice(self.n_items)
        return neg_samples

    

class BatchGenerator:
    """
    Batch Generator is responsible for train/valid
    batch data generation
    """
    def __init__(self, interactions, batch_size,
                 num_negatives, max_num_prefs,interactions_eval = None,random_state=42,
                 user_items=None,
                 **kwargs):
        """
        
        Initialize a data generator
        :param interactions:
        :param batch_size:
        :param num_negatives:
        :param random_state:
        :param user_items: default user items
        :param kwargs:
        
        When Evaluating
        interactions : test set
        interactions_eval: train/valid
        
        """
        self.interactions = interactions
        self.interactions_eval = interactions_eval
        
        if user_items is None:
            self.user_items = mat_to_dict(self.interactions,
                                          criteria=None)
        else:
            self.user_items = user_items
        self.batch_size = batch_size
        self.random_state = random_state
        n_interactions = self.interactions.count_nonzero()
        
        self.n_batches = int(n_interactions / self.batch_size)
        
        if self.interactions_eval: 
            self.n_batches = int(self.interactions.shape[1] / self.batch_size)
            
        
        if self.n_batches * self.batch_size < n_interactions:
            self.n_batches += 1
            

        self.current_batch_idx = 0
        # positive user item pairs
        
        self.user_pos_item_pairs = np.asarray(self.interactions.nonzero()).T
        
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(self.user_pos_item_pairs)
        
        self.sampler = Sampler(user_items=self.user_items,
                               n_items=self.interactions.shape[1],
                               n_negatives=num_negatives)
        self.ground_user_items = None
        self.ground_item_users = None
        
        self.max_num_prefs = max_num_prefs
        
        if self.max_num_prefs > 0:
            if self.interactions_eval:
                self.pref_items = self._get_pref_items(self.interactions_eval,
                                                   user_items,eval_ = True)
            
            else:
                self.pref_items = self._get_pref_items(self.interactions,
                                                   user_items)


    def _batch_triplets(self, batch_index):
        """
        Generate triplets & inter (user, pos_id, neg_ids, inter_ids)
        :param batch_index:
        :return:
        """
        # user_ids, pos_ids
        batch_size = self.batch_size
        batch_user_pos_items_pairs = self.user_pos_item_pairs[
                                     batch_index * batch_size:
                                     (batch_index + 1) * batch_size, :]
        # batch user_ids
        batch_user_ids = np.array(
            [uid for uid, _ in batch_user_pos_items_pairs])
        # batch positive item_ids
        batch_pos_ids = np.array([iid for _, iid in batch_user_pos_items_pairs])
        # batch negative item_ids
        batch_neg_ids = self.sampler.sampling(batch_user_ids)
        
        if batch_index == self.n_batches-1:
            batch_size = len(batch_pos_ids)
        
        batch_inter_inds = np.zeros(shape=(batch_size,self.max_num_prefs))

        for r,user_id in enumerate(batch_user_ids):
            inter_user = self.pref_items[user_id]
            batch_inter_inds[r] = np.array(inter_user[-self.max_num_prefs:] + [0]*(self.max_num_prefs - len(inter_user)))
        
        batch_user_ids = torch.as_tensor(batch_user_ids, device='cuda')
        batch_pos_ids = torch.as_tensor(batch_pos_ids, device='cuda')
        batch_neg_ids = torch.as_tensor(batch_neg_ids, device='cuda').squeeze(1)
        batch_inter_inds = torch.as_tensor(batch_inter_inds, device='cuda').int()
        
        return batch_user_ids, batch_pos_ids, batch_neg_ids, batch_inter_inds
    
    
    def _batch_test_sample_per_user(self,user_id,num_items,batch_index):
        """
        Generate triplets (user,item, inter_ids)
        :param batch_index:
        :return:
        """
#         # user_ids, pos_ids
        batch_size = self.batch_size
        batch_item_no = list(range(num_items))[batch_index * batch_size:(batch_index + 1) * batch_size]
        # batch user_ids
        batch_user_ids = np.array([user_id for _ in batch_item_no])

#         batch item_ids
        batch_item_ids = np.array(batch_item_no)

        if batch_index == self.n_batches-1:
            batch_size = len(batch_user_ids)
#         print(batch_index, batch_size)
        
        batch_inter_inds = np.zeros(shape=(batch_size,self.max_num_prefs))

        for r,user_id in enumerate(batch_user_ids):
            inter_user = self.pref_items[user_id]
            batch_inter_inds[r] = np.array(inter_user[-self.max_num_prefs:] + [0]*(self.max_num_prefs - len(inter_user)))
        
        batch_user_ids = torch.as_tensor(batch_user_ids, device='cuda')
        batch_item_ids = torch.as_tensor(batch_item_ids, device='cuda')
        batch_inter_inds = torch.as_tensor(batch_inter_inds, device='cuda').int()
        
        return batch_user_ids, batch_item_ids, batch_inter_inds

    @classmethod
    def _get_pref_items(cls, interactions, user_items, eval_ = False):
        """
        Get favorite items in user preferences
        :param interactions:
        :param user_items:
        :return:
        """
        # get ground truth user items
        if eval_:
            ground_user_items = interactions
        else: 
            ground_user_items = mat_to_dict(interactions)
        if user_items is None:
            # in the case training, pref items are
            # the same as user items
            pref_items = ground_user_items
        else:
            # in the case validation, pref items are the one
            # in training data, not in current interactions
            pref_items = {
                uid: items.difference(ground_user_items[uid])
                if uid in ground_user_items else items
                for uid, items in user_items.items()}
        return pref_items
