import os
import json
import torch
import numpy as np
import random
from math import radians, cos, sin, asin, sqrt, floor
import math
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
from einops import reduce, repeat
from math import cos, sin, exp
from torch.nn.functional import normalize
import scipy.sparse as sp
import torch.nn as nn

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def get_attn_mask(data_size, max_len, device):
    mask = torch.stack([(torch.triu(torch.ones(max_len, max_len))).transpose(0, 1) for _ in range(len(data_size))])
    return mask.cuda()

def get_attn_mask_pretrain(data_size, max_len, device):
    mask = torch.stack([(torch.triu(torch.ones(max_len, max_len))).transpose(0, 1) for _ in range(data_size)])
    return mask.cuda()

def get_pad_mask(data_size, max_len, device):
    mask = torch.zeros([len(data_size), max_len, max_len])
    for i in range(len(data_size)):
        mask[i][0: data_size[i], 0: data_size[i]] = torch.ones(data_size[i], data_size[i])
    return mask.cuda()

def get_key_pad_mask(data_size, max_len, num_neg, device):
    mask = torch.zeros([len(data_size), max_len, max_len])
    for i in range(len(data_size)):
        mask[i][0: data_size[i], 0: data_size[i]] = torch.ones(data_size[i], data_size[i])
    mask = mask.repeat(1, num_neg + 1, num_neg + 1).cuda()
    return mask

def get_mem_mask(data_size, max_len, num_neg, device):
    mem_mask = []
    for _ in range(len(data_size)):
        mask = torch.zeros((1 + num_neg) * max_len, (1 + num_neg) * max_len)
        attend_items = (torch.triu(torch.ones(max_len, max_len))).transpose(0, 1)
        for i in range(0, (1 + num_neg) * max_len, max_len):
            mask[i:i + max_len, i:i + max_len] = attend_items
        mem_mask.append(mask)
    mask = torch.stack(mem_mask).cuda()
    return mask

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    #print(np.sum(adj_mat, axis=1).squeeze(1),"sum shape")
    # row sum
    adj_list = [i[0] for i in  np.sum(adj_mat, axis=1).tolist()]
    deg_mat_row = np.asmatrix(np.diag(adj_list))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))
    print("id_mat")
    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        print("entering hat_rw_normd_lap_mat")
        print(deg_mat.shape, adj_mat.shape, id_mat.shape)
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        print("wid_adj_mat")
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

class LadderSampler(Sampler):
    def __init__(self, data_source, batch_sz, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_sz * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def fix_length(sequences, n_axies, max_len, dtype='source'):
    if dtype != 'eval trg loc':
        padding_term = torch.zeros_like(sequences[0])
        length = padding_term.size(0)
        # (l, any) -> (1, any) -> (max_len any)
        if n_axies == 1:
            padding_term = reduce(padding_term, '(h l) -> h', 'max', l=length)
            padding_term = repeat(padding_term, 'h -> (repeat h)', repeat=max_len)
        elif n_axies == 2:
            padding_term = reduce(padding_term, '(h l) any -> h any', 'max', l=length)
            padding_term = repeat(padding_term, 'h any -> (repeat h) any', repeat=max_len)
        else:
            padding_term = reduce(padding_term, '(h l) any_1 any_2 -> h any_1 any_2', 'max', l=length)
            padding_term = repeat(padding_term, 'h any_1 any_2 -> (repeat h) any_1 any_2', repeat=max_len)
        
        sequences.append(padding_term)
        tensor = pad_sequence(sequences, True)
        return tensor[:-1]
    else:
        tensor = pad_sequence(sequences, True)
        return tensor


def get_visited_locs(dataset):
    user_visited_locs = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_locs[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        user_visited_locs[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_locs[user].add(check_in[1])
    return user_visited_locs


def build_st_matrix(dataset, sequence, max_len):
    seq_len = len(sequence)
    temporal_matrix = torch.zeros(max_len, max_len)
    spatial_matrix = torch.zeros(max_len, max_len)
    seq_loc = [r[1] for r in sequence]
    seq_time = [r[3] for r in sequence]
    for x in range(seq_len):
        for y in range(x):
            # unit: hour
            temporal_matrix[x, y] = floor((seq_time[x] - seq_time[y]) / 3600)
            # unit: km
            spatial_matrix[x, y] = haversine(dataset.idx2gps[seq_loc[x]], dataset.idx2gps[seq_loc[y]])
    return temporal_matrix, spatial_matrix

def haversine(point_1, point_2):
    lat_1, lng_1 = point_1
    lat_2, lng_2 = point_2
    lat_1, lng_1, lat_2, lng_2 = map(radians, [lat_1, lng_1, lat_2, lng_2])

    d_lon = lng_2 - lng_1
    d_lat = lat_2 - lat_1
    a = sin(d_lat / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return floor(c * r)

def extract_sub_matrix(start_idx, end_idx, seq_len, max_len, temporal_matrix, spatial_matrix):
    sub_t_mat = torch.zeros(max_len, max_len)
    sub_s_mat = torch.zeros(max_len, max_len)
    if seq_len == max_len:
        sub_t_mat = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    else:
        sub_t_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    return sub_t_mat, sub_s_mat

def extract_sub_matrix_coo(start_idx, end_idx, seq_len, max_len, temporal_matrix, spatial_matrix):
    sub_t_mat = np.zeros([max_len, max_len])
    sub_s_mat = np.zeros([max_len, max_len])
    if seq_len == (max_len+1):
        sub_t_mat = temporal_matrix[start_idx: end_idx, start_idx: end_idx].numpy()
        sub_s_mat = spatial_matrix[start_idx: end_idx, start_idx: end_idx].numpy()
    else:
        sub_t_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = temporal_matrix[start_idx: end_idx, start_idx: end_idx].numpy()
        sub_s_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = spatial_matrix[start_idx: end_idx, start_idx: end_idx].numpy()
    idx_t_ = sub_t_mat.nonzero()
    idx_s_ = sub_s_mat.nonzero()
    data_t_ = sub_t_mat[idx_t_]
    data_s_ = sub_s_mat[idx_s_]
    idx_t = torch.LongTensor(np.vstack(idx_t_))
    idx_s = torch.LongTensor(np.vstack(idx_s_))
    data_t = torch.LongTensor(data_t_)
    data_s = torch.LongTensor(data_s_)
    sub_t_mat, sub_s_mat = torch.sparse_coo_tensor(idx_t, data_t, sub_t_mat.shape), torch.sparse_coo_tensor(idx_s, data_s, sub_s_mat.shape)
    return sub_t_mat, sub_s_mat


def extract_sub_matrix_(start_idx, end_idx, seq_len, max_len, temporal_matrix, spatial_matrix):
    sub_t_mat = torch.zeros([max_len, max_len])
    sub_s_mat = torch.zeros([max_len, max_len])
    if seq_len == (max_len+1):
        sub_t_mat = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    else:
        sub_t_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = temporal_matrix[start_idx: end_idx, start_idx: end_idx]
        sub_s_mat[start_idx - start_idx: end_idx - start_idx, start_idx - start_idx: end_idx - start_idx] = spatial_matrix[start_idx: end_idx, start_idx: end_idx]
    return sub_t_mat, sub_s_mat


def build_st_exp_decay_matrix(dataset, sequence, max_len):
    lambda_t = 0.1
    lambda_s = 100.0
    f_t = lambda delta_t: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*lambda_t)) # hover cosine + exp decay
    f_s = lambda delta_s: torch.exp(-torch.tensor(delta_s*lambda_s)) # exp decay
    seq_len = len(sequence)
    temporal_matrix = torch.zeros(max_len, max_len)
    spatial_matrix = torch.zeros(max_len, max_len)
    seq_loc = [r[1] for r in sequence]
    seq_time = [r[3] for r in sequence]
    for x in range(seq_len):
        for y in range(x):
            temporal_matrix[x, y] = f_t((torch.tensor(seq_time[x] - seq_time[y])))#floor((seq_time[x] - seq_time[y]) / 3600)
            spatial_matrix[x, y] = f_s(haversine(dataset.idx2gps[seq_loc[x]], dataset.idx2gps[seq_loc[y]]))#haversine(dataset.idx2gps[seq_loc[x]], dataset.idx2gps[seq_loc[y]])
    return temporal_matrix, spatial_matrix

def rs_mat2s(dataset, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(1, l_max, l_max)  # (L)
    mat = torch.zeros(l_max, l_max)  # mat (L, L)
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            mat[i, j] = haversine(dataset.idx2gps[int(loc1)], dataset.idx2gps[int(loc2)])
    return mat  # (L, L)



def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(-1)).unsqueeze(0).cuda()      # A^=A+I
    row_sum = matrix.sum(-1)                          # D
    d_inv_sqrt = torch.pow(row_sum, -0.5)#.flatten()  # D^-1/2
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(1,2).matmul(d_mat_inv_sqrt)
    )                                                #D^-1/2 * A^ * D^-1/2
    #normalized_laplacian = torch.sigmoid(normalized_laplacian)
    return normalized_laplacian

def build_adj_matrix_coo(dataset, sequence, time, median_coor, median_time, max_len, dis_cluster_seq):
    adj_spatial_mat = []
    adj_temporal_mat = []
    adj_interest_mat = []
    sequence_adj_i = []
    sequence_adj_all = []
    data_size = []
    pad_len_seq = max_len - len(sequence)
    for seq_index in range(len(sequence)):
        adj_spatial_matrix = np.zeros((max_len))
        adj_temporal_matrix = np.zeros((max_len))
        adj_interest_matrix = np.zeros((max_len, max_len))

        seq = sequence[:(seq_index+1)]

        loc_set = set([i for i in seq])
        seq_adj_mask = list(loc_set)
        # loc_count = [0 for _ in range(len(seq_adj_mask))]
        # for i in range(len(seq)):
        #     loc_count[seq_adj_mask.index(seq[i])] += 1
        pad_len_i = max_len - len(seq_adj_mask)
        seq_adj_i = seq_adj_mask + [0] * pad_len_i
        pad_len_all = max_len - len(seq)
        seq_adj_all = seq + [0] * pad_len_all

        for i in range(len(seq)):
            adj_spatial_matrix[i] = haversine(dataset.idx2gps[int(seq[i])], median_coor[int(dis_cluster_seq[i])])
            adj_temporal_matrix[i] = abs(floor((time[i] - median_time) / 3600))
        for i in range(len(seq)):
            # if len(seq_adj_mask) == 1:
            #     adj_interest_matrix[seq_adj_mask.index(seq[i]), seq_adj_mask.index(seq[i])] += 1 
            #     continue
            adj_interest_matrix[seq_adj_mask.index(seq[i-1]), seq_adj_mask.index(seq[i])] += 1 
        # idx_s_ = adj_spatial_matrix.nonzero()
        # idx_t_ = adj_temporal_matrix.nonzero()
        # idx_i_ = adj_interest_matrix.nonzero()
        # data_s = torch.LongTensor(adj_spatial_matrix[idx_s_])
        # data_t = torch.LongTensor(adj_temporal_matrix[idx_t_])
        # data_i = torch.LongTensor(adj_interest_matrix[idx_i_])
        # idx_s = torch.LongTensor(np.vstack(idx_s_))
        # idx_t = torch.LongTensor(np.vstack(idx_t_))
        # idx_i = torch.LongTensor(np.vstack(idx_i_))
        # adj_spatial_matrix = torch.sparse_coo_tensor(idx_s, data_s, adj_spatial_matrix.shape)
        # adj_temporal_matrix = torch.sparse_coo_tensor(idx_t, data_t, adj_temporal_matrix.shape)
        # adj_interest_matrix = torch.sparse_coo_tensor(idx_i, data_i, adj_interest_matrix.shape)
        adj_spatial_mat.append(adj_spatial_matrix)
        adj_temporal_mat.append(adj_temporal_matrix)
        adj_interest_mat.append(adj_interest_matrix)
        sequence_adj_i.append(seq_adj_i)
        sequence_adj_all.append(seq_adj_all)
        data_size.append(len(seq_adj_mask))
    sequence_adj_i.extend([[0]*max_len]*pad_len_seq)
    sequence_adj_all.extend([[0]*max_len]*pad_len_seq)
    adj_spatial_mat.extend([[0.0]*max_len]*pad_len_seq)
    adj_temporal_mat.extend([[0.0]*max_len]*pad_len_seq)
    adj_interest_mat.extend([[[0]*max_len]*max_len]*pad_len_seq)
    data_size.extend([0]*pad_len_seq)
    


    #convert to sparse_coo
    sequence_adj_i = np.array(sequence_adj_i)
    sequence_adj_all = np.array(sequence_adj_all)
    adj_spatial_mat = np.array(adj_spatial_mat)
    adj_temporal_mat = np.array(adj_temporal_mat)
    adj_interest_mat = np.array(adj_interest_mat)
    data_size = np.array(data_size)

    idx_seq_i_ = sequence_adj_i.nonzero()
    idx_seq_all_ = sequence_adj_all.nonzero()
    idx_s_ = adj_spatial_mat.nonzero()
    idx_t_ = adj_temporal_mat.nonzero()
    idx_i_ = adj_interest_mat.nonzero()
    idx_ds_ = data_size.nonzero()
    data_seq_i = torch.LongTensor(sequence_adj_i[idx_seq_i_])
    data_seq_all = torch.LongTensor(sequence_adj_all[idx_seq_all_])
    data_s = torch.LongTensor(adj_spatial_mat[idx_s_])
    data_t = torch.LongTensor(adj_temporal_mat[idx_t_])
    data_i = torch.LongTensor(adj_interest_mat[idx_i_])
    data_ds = torch.LongTensor(data_size[idx_ds_])
    idx_seq_i = torch.LongTensor(np.vstack(idx_seq_i_))
    idx_seq_all = torch.LongTensor(np.vstack(idx_seq_all_))
    idx_s = torch.LongTensor(np.vstack(idx_s_))
    idx_t = torch.LongTensor(np.vstack(idx_t_))
    idx_i = torch.LongTensor(np.vstack(idx_i_))
    idx_ds = torch.LongTensor(np.vstack(idx_ds_))
    adj_seq_i_matrix = torch.sparse_coo_tensor(idx_seq_i, data_seq_i, [max_len, max_len])
    adj_seq_all_matrix = torch.sparse_coo_tensor(idx_seq_all, data_seq_all, [max_len, max_len])
    adj_spatial_matrix = torch.sparse_coo_tensor(idx_s, data_s, [max_len, max_len])
    adj_temporal_matrix = torch.sparse_coo_tensor(idx_t, data_t, [max_len, max_len])
    adj_interest_matrix = torch.sparse_coo_tensor(idx_i, data_i, [max_len, max_len, max_len])
    data_size_matrix = torch.sparse_coo_tensor(idx_ds, data_ds, [max_len])
    return adj_seq_i_matrix, adj_seq_all_matrix, adj_spatial_matrix, adj_temporal_matrix, adj_interest_matrix, data_size_matrix


def build_adj_matrix(dataset, sequence, time, median_coor, median_time, max_len):
    adj_spatial_mat = []
    adj_temporal_mat = []
    adj_interest_mat = []
    sequence_adj = []
    data_size = []
    pad_len_seq = max_len - len(sequence)
    for seq_index in range(len(sequence)):
        adj_spatial_matrix = np.zeros((max_len))
        adj_temporal_matrix = np.zeros((max_len))
        adj_interest_matrix = np.zeros((max_len, max_len))
        seq = sequence[:(seq_index+1)]
        loc_set = set([i for i in seq])
        seq_adj_mask = list(loc_set)
        loc_count = [0 for _ in range(len(seq_adj_mask))]
        for i in range(len(seq)):
            loc_count[seq_adj_mask.index(seq[i])] += 1
        pad_len = max_len - len(seq_adj_mask)
        seq_adj = seq_adj_mask + [0] * pad_len
        for i in range(len(seq_adj_mask)):
            adj_spatial_matrix[i] = haversine(dataset.idx2gps[int(seq_adj_mask[i])], median_coor)
            adj_temporal_matrix[i] = abs(floor((time[i] - median_time) / 3600))
            if len(seq_adj_mask) == 1:
                adj_interest_matrix[seq_adj_mask.index(seq[i]), seq_adj_mask.index(seq[i])] += 1 
                continue
            adj_interest_matrix[seq_adj_mask.index(seq[i-1]), seq_adj_mask.index(seq[i])] += 1 
        # idx_s_ = adj_spatial_matrix.nonzero()
        # idx_t_ = adj_temporal_matrix.nonzero()
        # idx_i_ = adj_interest_matrix.nonzero()
        # data_s = torch.LongTensor(adj_spatial_matrix[idx_s_])
        # data_t = torch.LongTensor(adj_temporal_matrix[idx_t_])
        # data_i = torch.LongTensor(adj_interest_matrix[idx_i_])
        # idx_s = torch.LongTensor(np.vstack(idx_s_))
        # idx_t = torch.LongTensor(np.vstack(idx_t_))
        # idx_i = torch.LongTensor(np.vstack(idx_i_))
        # adj_spatial_matrix = torch.sparse_coo_tensor(idx_s, data_s, adj_spatial_matrix.shape)
        # adj_temporal_matrix = torch.sparse_coo_tensor(idx_t, data_t, adj_temporal_matrix.shape)
        # adj_interest_matrix = torch.sparse_coo_tensor(idx_i, data_i, adj_interest_matrix.shape)
        adj_spatial_mat.append(adj_spatial_matrix)
        adj_temporal_mat.append(adj_temporal_matrix)
        adj_interest_mat.append(adj_interest_matrix)
        sequence_adj.append(seq_adj)
        data_size.append(len(sequence))
    sequence_adj.extend([[0]*max_len]*pad_len_seq)
    adj_spatial_mat.extend([[0.0]*max_len]*pad_len_seq)
    adj_temporal_mat.extend([[0.0]*max_len]*pad_len_seq)
    adj_interest_mat.extend([[[0]*max_len]*max_len]*pad_len_seq)
    data_size.extend([0]*pad_len_seq)
    


    #convert to sparse_coo
    adj_seq_matrix = torch.from_numpy(np.array(sequence_adj))
    adj_spatial_matrix = torch.from_numpy(np.array(adj_spatial_mat))
    adj_temporal_matrix = torch.from_numpy(np.array(adj_temporal_mat))
    adj_interest_mat = torch.from_numpy(np.array(adj_interest_mat))
    data_size_matrix = torch.from_numpy(np.array(data_size))
    return adj_seq_matrix, adj_spatial_matrix, adj_temporal_matrix, adj_interest_mat, data_size_matrix

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]