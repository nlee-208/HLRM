from collections import defaultdict
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import argparse

def df_to_mat(df, n_rows, n_cols, binary=True):
    """
    Convert dataframe to matrix
    :param df:
    :param n_rows:
    :param n_cols:
    :param binary:
    :return:
    """
    dtype = np.int32 if binary is True else np.float32
    interactions_mat = sp.dok_matrix((n_rows, n_cols),
                                     dtype=dtype)
    interactions_mat[
        df.user.tolist(), df.item.tolist()] = 1
    interactions_mat = interactions_mat.tocsr()
    return interactions_mat

def mat_to_dict(interactions, criteria=None):
    """
    Convert sparse matrix to dictionary of set
    :param interactions: scipy sparse matrix
    :param criteria:
    :return:
    """
    if not sp.isspmatrix_lil(interactions):
        interactions = sp.lil_matrix(interactions)
    n_rows = interactions.shape[0]
    res = {
        u: list(set(interactions.rows[u])) for u in range(n_rows)
        if criteria is None or
           (criteria is not None and criteria(interactions, u) is True)
    }
    return res



def read_data(params):
    """
    Fetch interactions from file
    :param params: dataset parameters
    """
    interaction_path = os.path.join(params['path'],
                                    params['interactions'])
    # read interaction csv file
    sep = params.get('sep', ',')
    encoding = params.get('encoding', 'utf8')
    data = pd.read_csv(interaction_path,
                       sep=sep,
                       header=0,
                       encoding=encoding)
    data['rating'] = data['rating'].astype(int)
    data.columns = ['org_user','org_item','rating', 'timestamp']
    return data


def split_data(df, split_method = 'fo', test_size=.2, random_state=42):
    if split_method == 'fo':
        train_set, test_set = train_test_split(df,
                                            test_size=test_size,
                                            random_state=random_state)
    return train_set, test_set



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')