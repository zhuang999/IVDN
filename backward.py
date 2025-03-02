import copy
import math
import time
from unicodedata import category
import numpy as np
import joblib
import gc
from tqdm import tqdm
from nltk import ngrams
from collections import defaultdict
from torch.utils.data import Dataset
from utils import build_st_matrix, extract_sub_matrix, extract_sub_matrix_coo, build_st_exp_decay_matrix, rs_mat2s, neg_sample
from quadkey_encoder import latlng2quadkey
from torchtext.data import Field
import random
import torch
import os
from utils import *
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

class LBSNData(Dataset):
    def __init__(self, data_name, data_path, min_loc_freq, min_user_freq, map_level, args):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.args = args
        self.n_loc = 1
        self.build_vocab(data_name, data_path, min_loc_freq)
        self.user_seq, self.user2idx, self.quadkey2idx, self.n_user, self.n_quadkey, self.quadkey2loc = \
            self.processing(data_name, data_path, min_user_freq, map_level)

    def build_vocab(self, data_name, data_path, min_loc_freq):
        #for line in open(data_path, encoding='gbk'):
        for index,line in enumerate(open(data_path, encoding='UTF-8')):
            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng, city, category
                if index == 0:
                    continue
                line = line.strip().split(',')
                if len(line) != 7:
                    continue
                loc = line[1]
                coordinate = float(line[3]), float(line[4])
            elif data_name == 'brightkite' or data_name == 'gowalla':
                # user, time(2010-10-19T23:55:27Z), lat, lng, loc
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                loc = line[4]
                coordinate = float(line[2]), float(line[3])
            elif data_name == 'cc':
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                loc = line[2]
                coordinate = float(line[3]), float(line[4])
            self.add_location(loc, coordinate)
        if min_loc_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_loc_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.locidx2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.locidx2freq[idx - 1] = self.loc2count[loc]

    def add_location(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def processing(self, data_name, data_path, min_user_freq, map_level):
        user_seq = {}
        user_seq_array = list()
        quadkey2idx = {}
        idx2quadkey = {}
        quadidx2loc = defaultdict(set)
        n_quadkey = 1
        user2idx = {}
        n_user = 1
        # for line in open(data_path, encoding='gbk'):
        for index,line in enumerate(open(data_path, encoding='UTF-8')):
            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng
                if index == 0:
                    continue
                line = line.strip().split(',')
                if len(line) != 7:
                    continue
                user, loc, t, lat, lng, city, category = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y-%m-%dT%H:%M:%S')
                timestamp = time.mktime(t)
            elif data_name == 'gowalla' or data_name == 'brightkite':
                # user, time(2010-10-19T23:55:27Z), lat, lng, loc
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                user, t, lat, lng, loc = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
                timestamp = time.mktime(t)
            elif data_name == 'cc':
                # user, time(20101019235527), loc, lat, lng
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                user, t, loc, lat, lng = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y%m%d%H%M%S')
                timestamp = time.mktime(t)
            if loc not in self.loc2idx:
                continue
            loc_idx = self.loc2idx[loc]
            quadkey = latlng2quadkey(lat, lng, map_level)
            if quadkey not in quadkey2idx:
                quadkey2idx[quadkey] = n_quadkey
                idx2quadkey[n_quadkey] = quadkey
                n_quadkey += 1
            quadkey_idx = quadkey2idx[quadkey]
            quadidx2loc[quadkey_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, quadkey, lat, lng, timestamp])
        for user, seq in user_seq.items():
            if len(seq) >= min_user_freq:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, quadkey, lat, lng, timestamp in sorted(seq, key=lambda e:e[-1]):
                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, quadkey, timestamp, True))
                    else:
                        seq_new.append((user_idx, loc, quadkey, timestamp, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_user_freq / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)

        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[2]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                all_quadkeys.append(region_quadkey_bigram)
                user_seq_array[u][i] = (check_in[0], check_in[1], region_quadkey_bigram, check_in[3], check_in[4])

        self.loc2quadkey = ['NULL']
        for l in range(1, self.n_loc):
            lat, lng = self.idx2gps[l]
            quadkey = latlng2quadkey(lat, lng, map_level)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_bigram = quadkey_bigram.split()
            self.loc2quadkey.append(quadkey_bigram)
            all_quadkeys.append(quadkey_bigram)

        self.QUADKEY = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.QUADKEY.build_vocab(all_quadkeys)
        return user_seq_array, user2idx, quadkey2idx, n_user, n_quadkey, quadidx2loc

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def data_partition(self, args, max_len, st_matrix):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        test_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        test_seq = []

        # # Building Spatial-Temporal Relation Matrix
        # if os.path.exists(args.candidate_mat_path):
        #     with open(args.candidate_mat_path, 'rb') as file:
        #         mat2s = _pickle.load(file)
        # else:
        #     #mat2s = torch.zeros((100,100))#rs_mat2s(self, self.n_loc-1)
        #     mat2s = rs_mat2s(self, self.n_loc-1)
        #     joblib.dump(mat2s, args.candidate_mat_path, protocol=4)
        #     #torch.save(mat2s, args.candidate_mat_path)
        #     with open(args.candidate_mat_path, 'rb') as file:
        #         mat2s = _pickle.load(file) 
        self.su, self.sl, self.tu, self.tl = 0, 0, 0, 0
        for user in tqdm(range(len(self)), ncols=70):
            seq = self[user]
            temporal_matrix, spatial_matrix = st_matrix[user]
            if temporal_matrix.max() > self.tu:
                self.tu = temporal_matrix.max()
            if temporal_matrix.min() < self.tl:
                self.tl = temporal_matrix.min()
            if spatial_matrix.max() > self.su:
                self.su = spatial_matrix.max()
            if spatial_matrix.min() < self.sl:
                self.sl = spatial_matrix.min()
            for i in reversed(range(len(seq))):    #Delete the POI that appears for the first time in the validation set
                if not seq[i][-1]:
                    break
            #i = len(seq)
            eval_trg = seq[i-1:i]
            test_trg = seq[i:i+1]
            eval_src_seq = seq[max(0, i - max_len - 1): i - 1]
            test_src_seq = seq[max(0, i - max_len): i]

            eval_t_mat, eval_s_mat = extract_sub_matrix(max(0, i - max_len - 1),
                                                        i-1,
                                                        len(eval_src_seq),
                                                        max_len,
                                                        temporal_matrix,
                                                        spatial_matrix)

            # eval_t_decay_mat, eval_s_decay_mat = extract_sub_matrix(max(0, i - max_len - 1 ),
            #                                                         i-1,
            #                                                         len(eval_src_seq),
            #                                                         max_len,
            #                                                         temporal_decay_matrix,
            #                                                         spatial_decay_matrix)
            eval_mat2t, _ = extract_sub_matrix(max(0, i - max_len),
                                        i,
                                        len(eval_src_seq),
                                        max_len,
                                        temporal_matrix,
                                        spatial_matrix)                    
            eval_seq.append((eval_src_seq, eval_trg, eval_t_mat, eval_s_mat, eval_mat2t))


            test_t_mat, test_s_mat = extract_sub_matrix(max(0, i - max_len),
                                                        i,
                                                        len(test_src_seq),
                                                        max_len,
                                                        temporal_matrix,
                                                        spatial_matrix)                             
            # test_t_decay_mat, test_s_decay_mat = extract_sub_matrix(max(0, i - max_len),
            #                                                         i,
            #                                                         len(test_src_seq),
            #                                                         max_len,
            #                                                         temporal_decay_matrix,
            #                                                         spatial_decay_matrix)
            test_mat2t, _ = extract_sub_matrix(max(0, i - max_len + 1),
                                        i + 1,
                                        len(eval_src_seq),
                                        max_len,
                                        temporal_matrix,
                                        spatial_matrix)
            test_seq.append((test_src_seq, test_trg, test_t_mat, test_s_mat, test_mat2t))

            n_instance = math.floor((i + max_len - 2) / max_len)
            i = i - 1
            for k in range(n_instance):
                if (i - k * max_len) > max_len * 1.1:
                    train_trg_seq = seq[i - (k + 1) * max_len: i - k * max_len]
                    train_src_seq = seq[i - (k + 1) * max_len - 1: i - k * max_len - 1]

                    t_mat, s_mat = extract_sub_matrix(i - (k + 1) * max_len - 1,
                                                    i - k * max_len - 1,
                                                    len(train_src_seq),
                                                    max_len,
                                                    temporal_matrix,
                                                    spatial_matrix)
                    # t_decay_mat, s_decay_mat = extract_sub_matrix(i - (k + 1) * max_len - 1,
                    #                                             i - k * max_len - 1,
                    #                                             len(train_src_seq),
                    #                                             max_len,
                    #                                             temporal_matrix,
                    #                                             spatial_matrix)
                    mat2t, _ = extract_sub_matrix(i - (k + 1) * max_len,
                                                    i - k * max_len,
                                                    len(train_src_seq),
                                                    max_len,
                                                    temporal_matrix,
                                                    spatial_matrix)
                    user_truncated_segment_index = k
                    train_seq.append((train_src_seq, train_trg_seq, t_mat, s_mat, mat2t, user_truncated_segment_index))
                else:
                    train_trg_seq = seq[max(i - (k + 1) * max_len, 0): i - k * max_len]
                    train_src_seq = seq[max(i - (k + 1) * max_len - 1, 0): i - k * max_len - 1]
                    t_mat, s_mat = extract_sub_matrix(max(i - (k + 1) * max_len - 1, 0),
                                                    i - k * max_len - 1,
                                                    len(train_src_seq),
                                                    max_len,
                                                    temporal_matrix,
                                                    spatial_matrix)
                    # t_decay_mat, s_decay_mat = extract_sub_matrix(max(i - (k + 1) * max_len - 1, 0),
                    #                                             i - k * max_len - 1,
                    #                                             len(train_src_seq),
                    #                                             max_len,
                    #                                             temporal_matrix,
                    #                                             spatial_matrix)
                    mat2t, _ = extract_sub_matrix(max(i - (k + 1) * max_len, 0),
                                                    i - k * max_len,
                                                    len(train_src_seq),
                                                    max_len,
                                                    temporal_matrix,
                                                    spatial_matrix)
                    user_truncated_segment_index = k
                    train_seq.append((train_src_seq, train_trg_seq, t_mat, s_mat, mat2t, user_truncated_segment_index))
                    break

        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        test_data.user_seq = test_seq
        return train_data, eval_data, test_data

    def spatial_temporal_matrix_building(self, path):
        user_matrix = {}
        for u in tqdm(range(len(self))):
            seq = self[u]
            temporal_matrix, spatial_matrix = build_st_matrix(self, seq, len(seq))
            user_matrix[u] = (temporal_matrix, spatial_matrix)
        joblib.dump(user_matrix, path)
        del user_matrix
        gc.collect()

class PretrainData(LBSNData):
    def __init__(self, data_name, data_path, min_loc_freq, min_user_freq, map_level, args):
        super().__init__(data_name, data_path, min_loc_freq, min_user_freq, map_level, args)
        self.args = args
        self.part_sequence = []
        self.long_sequence = []
        self.t_mat_seq = []
        self.s_mat_seq = []
        self.max_len = args.max_len
        self.mask_id = self.n_loc
        self.item_size = self.n_loc + 1
        self.split_sequence(args.st_matrix)
    def __len__(self):
        return len(self.part_sequence)
    def split_sequence(self, st_matrix):
        self.su, self.sl, self.tu, self.tl = 0, 0, 0, 0
        for user,infos in enumerate(self.user_seq):
            temporal_matrix, spatial_matrix = st_matrix[user]
            if temporal_matrix.max() > self.tu:
                self.tu = temporal_matrix.max()
            if temporal_matrix.min() < self.tl:
                self.tl = temporal_matrix.min()
            if spatial_matrix.max() > self.su:
                self.su = spatial_matrix.max()
            if spatial_matrix.min() < self.sl:
                self.sl = spatial_matrix.min()
            user_seq = []
            for info in infos:
                user_seq.append(info[1])
                self.long_sequence.append(info[1])
            input_ids_all = user_seq[:-2]
            n_instance = math.ceil((len(input_ids_all) - 2) / (self.max_len+1))
            for n in range(n_instance):
                input_ids = input_ids_all[n*(self.max_len+1):min((n+1)*(self.max_len+1), len(input_ids_all))]
                t_mat, s_mat = extract_sub_matrix(n*(self.max_len+1),
                                                      min((n+1)*(self.max_len+1), 
                                                      len(input_ids_all)),
                                                      len(input_ids),
                                                      (self.max_len+1),
                                                      temporal_matrix,
                                                      spatial_matrix)
                for i in range(len(input_ids)):
                    self.part_sequence.append(input_ids[:i+1])
                    sub_t_mat, sub_s_mat = extract_sub_matrix_coo(0,
                                                      i, 
                                                      i+1,
                                                      self.max_len,
                                                      t_mat,
                                                      s_mat)
                    self.t_mat_seq.append(sub_t_mat)
                    self.s_mat_seq.append(sub_s_mat)

    def __getitem__(self, index):
        sequence_ = self.part_sequence[index] # pos_items
        t_mat = self.t_mat_seq[index]
        s_mat = self.s_mat_seq[index]
        sequence = sequence_[:-1]
        target = sequence_[1:]
        seq_t_mat = t_mat
        seq_s_mat = s_mat
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        target_items = [0] * pad_len + target
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment



        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        target_items = target_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

    

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len
        assert len(target_items) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),
                       torch.tensor(target_items, dtype=torch.long),
                       seq_t_mat,
                       seq_s_mat,
                       )
        return cur_tensors