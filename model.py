import torch
import torch.nn as nn
import torch.nn.functional as F

class UserAttention(nn.Module):
    """ Basic Item Attention for HLRM """
    def __init__(self, emb_size,num_relations):
        super().__init__()
        self.emb_size = emb_size
        self.num_relations = num_relations
        self.key_ = nn.Embedding(self.num_relations,self.emb_size)
        self.val_ = nn.Embedding(self.num_relations,self.emb_size)


    def get_i2i_inter_rel(self,item,inter):
        """
        Input:
            - Item: (b,emb)
            - Inter: (b,num_inter,emb)

        Output:
            - Rel_Vecs : (b,num_inter,emb)
        """
        w = torch.einsum('ri,ri -> ri', self.key_.weight,self.val_.weight)
        return torch.einsum('mbj,ij -> bmj', (item*inter.transpose(0,1)), w)

    def get_u2i_attn_weights(self,user, rel_vecs):
        """
        Input:
            - User: (b,emb)
            - Rel_vecs: (b,emb)

        Output:
            - Attn_weights : (b, emb)
        """
        bef = torch.einsum('bji,bi -> bj', rel_vecs, user)
        return F.softmax(bef, dim = 1)

    def forward(self,user,item, inter):
        """
        Input:
            - User: (b,emb)
            - Item: (b,emb)
            - Inter: (b, num_inter, emb)

        Output:
            - Weighted_rel_vecs : (b, emb)
        
        """
        rel_vecs = self.get_i2i_inter_rel(item, inter)
        attn = self.get_u2i_attn_weights(user, rel_vecs)

        return torch.einsum('bi,bij -> bj', attn, rel_vecs)

    
class ItemAttention(nn.Module):
    """ Additional User Attention for HLRM ++ """
    def __init__(self, emb_size,num_relations):
        super().__init__()
        self.emb_size = emb_size
        self.num_relations = num_relations
        self.key_ = nn.Embedding(self.num_relations,self.emb_size)
        self.val_ = nn.Embedding(self.num_relations,self.emb_size)

    def get_u2i_inter_rel(self,user,inter):
        """
        Input:
            - User: (b,emb)
            - Inter: (b,num_inter,emb)

        Output:
            - Rel_Vecs : (b,num_inter,emb)
        """
        w = torch.einsum('ri,ri -> ri', self.key_.weight,self.val_.weight)
        return torch.einsum('mbj,ij -> bmj', (user*inter.transpose(0,1)), w)


    def get_i2i_attn_weights(self,item, rel_vecs):
        """
        Input:
            - Item: (b,emb)
            - Rel_vecs: (b,emb)

        Output:
            - Attn_weights : (b, emb)
        """
        bef = torch.einsum('bji,bi -> bj', rel_vecs, item)
        return F.softmax(bef, dim = 1)

    def forward(self,user,item, inter):
        """
        Input:
            - User: (b,emb)
            - Item: (b,emb)
            - Inter: (b, num_inter, emb)

        Output:
            - Weighted_rel_vecs : (b, emb)
        
        """
        rel_vecs = self.get_u2i_inter_rel(user, inter)
        attn = self.get_i2i_attn_weights(item, rel_vecs)

        return torch.einsum('bi,bij -> bj', attn, rel_vecs)


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def determine_mode(self, split):
        if split == 'train':
            self.train()
        else:
            self.eval()

            
class HLRM(BaseModel):
    """ Hierarhical Latent Relation Modeling  """
    def __init__(self, 
                emb_size, 
                num_relations, 
                user_num, item_num, 
                pretrained_embs = False,
                user_embs = None,
                item_embs = None,
                model_type = 'base'):

        super().__init__()
        self.emb_size = emb_size
        self.num_relations = num_relations
        self.user_num = user_num
        self.item_num = item_num
        self.pretrained_embs = pretrained_embs
        self.user_embs_pretrained = user_embs
        self.item_embs_pretrained = item_embs

        
        self.userattn = UserAttention(self.emb_size, self.num_relations)
        
        if model_type == 'plus':
            self.itemattn = ItemAttention(self.emb_size, self.num_relations)

        if self.pretrained_embs:
            self.user_emb = nn.Embedding.from_pretrained(self.user_embs_pretrained,max_norm = 1)
            self.item_emb = nn.Embedding.from_pretrained(self.item_embs_pretrained,max_norm = 1)

        else :
            self.user_emb = nn.Embedding(self.user_num,self.emb_size,max_norm = 1)
            self.item_emb = nn.Embedding(self.item_num,self.emb_size, max_norm = 1)


        self.model_type = model_type
        
    def inference(self, user_id,item_id, inter_id):
        """ Inference for a single {user,item,interaction} triplet """
        user_feat = self.user_emb(user_id)
        item_feat = self.item_emb(item_id)
        inter_feat = self.item_emb(inter_id)
        
        rel = self.userattn(user_feat,item_feat,inter_feat)
        
        if self.model_type == 'plus':
            rel += self.itemattn(user_feat,item_feat,inter_feat)
            
        return user_feat + rel
                                      
    def forward(self, user_id, item_id_p, item_id_n, inter_id):
        """ 
        Forward to retrieve relational vector of pos_item * neg_item for triplet loss 
        """
        user_feat = self.user_emb(user_id)      # (b, emb)
        item_feat_p = self.item_emb(item_id_p)  # (b, emb)
        item_feat_n = self.item_emb(item_id_n)  # (b, emb)
        inter_feat = self.item_emb(inter_id)    # (b, num_inter, emb)
        
        # retrieve relation vectors from user-attention 
        rel_p = self.userattn(user_feat,item_feat_p,inter_feat) # (b,emb)
        rel_n = self.userattn(user_feat,item_feat_n,inter_feat) # (b,emb)

        # retrieve relation vectors from item-attention if HLRM++
        if self.model_type == 'plus':
            rel_p += self.itemattn(user_feat,item_feat_p,inter_feat)    # (b,emb)
            rel_n += self.itemattn(user_feat,item_feat_n,inter_feat)    # (b,emb)

        return user_feat,item_feat_p,item_feat_n, rel_p, rel_n




class ItemAttention_Extended(nn.Module):
    """ 
    Extended Item Attention for HLRM_Aug 
    Different from the original ItemAttention as utilizing the item's previous user interaction
    """
    def __init__(self, emb_size,num_relations):
        super().__init__()
        self.emb_size = emb_size
        self.num_relations = num_relations
        self.key_ = nn.Embedding(self.num_relations,self.emb_size)
        self.val_ = nn.Embedding(self.num_relations,self.emb_size)


    def get_u2u_inter_rel(self,user,i2u_inter):
        """
        Input:
            - User: (b,emb)
            - Inter: (b,num_inter,emb)

        Output:
            - Rel_Vecs : (b,num_inter,emb)
        """
        w = torch.einsum('ri,ri -> ri', self.key_.weight,self.val_.weight)
        return torch.einsum('mbj,ij -> bmj', (user*i2u_inter.transpose(0,1)), w)

    def get_u2i_attn_weights(self,item, rel_vecs):
        """
        Input:
            - User: (b,emb)
            - Rel_vecs: (b,emb)

        Output:
            - Attn_weights : (b, emb)
        """
        bef = torch.einsum('bji,bi -> bj', rel_vecs, item)
        return F.softmax(bef, dim = 1)

    def forward(self,user,item, i2u_inter):
        """
        Input:
            - User: (b,emb)
            - Item: (b,emb)
            - Inter: (b, num_inter, emb)

        Output:
            - Weighted_rel_vecs : (b, emb)
        
        """
        rel_vecs = self.get_u2u_inter_rel(user, i2u_inter)
        attn = self.get_u2i_attn_weights(item, rel_vecs)

        return torch.einsum('bi,bij -> bj', attn, rel_vecs)