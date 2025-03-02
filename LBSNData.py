import copy
import math
import time
from unicodedata import category
import numpy as np
import joblib
import gc
from tqdm import tqdm
from nltk import ngrams
#from near_location_query import Loc_Query_System
from neg_sampler import *
from near_location_query import *
from collections import defaultdict
from torch.utils.data import Dataset
from utils import build_st_matrix, extract_sub_matrix, extract_sub_matrix_coo, build_st_exp_decay_matrix, rs_mat2s, neg_sample, calculate_laplacian_with_self_loop, build_adj_matrix, build_adj_matrix_coo
from quadkey_encoder import latlng2quadkey
from torchtext.data import Field
import geohash
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
import random
import torch
import os
import argparse
import matplotlib.pyplot as plt
from utils import *
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

def calLaplacianMatrix(adjacentMatrix):
    
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def Spect(data, cluster_nodes):
    data=np.squeeze(data)
    Sp = SpectralClustering(n_clusters=cluster_nodes).fit(data)
    return Sp.labels_

def spKmeans(H, cluster_nodes):
    sp_kmeans = KMeans(n_clusters=cluster_nodes,n_init=30,random_state=9).fit(H)
    return sp_kmeans.labels_

def poi_distance_cluster(l_max, mat_, poi, cluster_nodes):
    np.set_printoptions(threshold = np.inf) 
    #cluster_nodes = 4
    #print(poi[:4])
    
    # adj1=np.zeros((l_max, l_max))
    # count=0
    # for i in range(l_max):
    #     if i == 0:
    #         continue
    #     order=np.argsort(mat[i,:])
    #     var=np.var(mat[i,:])
    #     if var == 0:
    #         print(mat[i,:])
    #     for j in range(10):
    #         if np.exp(-mat[i,order[j]]/var)>0.95:
    #             adj1[i,order[j]]=np.exp(-mat[i,order[j]]/var)
    #             count+=1
    # adj1[0,0] = 1
    mat = mat_ + mat_.T
    #print(mat[:4])
    adj2 = np.zeros((l_max, l_max))
    for i in range(l_max):
        var = np.var(mat[i,:])

        if var == 0:
            print(mat[:4])
        for j in range(l_max):
            # if var == 0:
            #     adj2[i,j] = -1e9
            # else:
            adj2[i,j] = np.exp(-mat[i,j]/var)
    #adj2[0,0] = 1
    Laplacian = calLaplacianMatrix(adj2)
    
    lam, H = np.linalg.eig(Laplacian) # H'shape is n*n
    lam = lam.real
    H = H.real
    H = H[:,0:20]
    #idx=spKmeans(H,cluster_nodes)
    idx=Spect(H,cluster_nodes)
    #print(idx)

    #print("idx")
    return idx  # (L, L)



class LBSNData(Dataset):
    def __init__(self, data_name, data_path, min_loc_freq, min_user_freq, map_level, args):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.loc2gps2 = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.args = args
        self.n_loc = 1
        self.build_vocab(data_name, data_path, min_loc_freq)
        self.user_seq, self.user2idx, self.quadkey2idx, self.n_user, self.n_quadkey, self.quadkey2loc = \
            self.processing(data_name, data_path, min_user_freq, map_level)

    def build_vocab(self, data_name, data_path, min_loc_freq):
        #for index,line in enumerate(open(data_path, encoding='ISO-8859-1')):
        for index,line in enumerate(open(data_path, encoding='UTF-8')):
            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng, city, category
                # if index == 0:
                #     continue
                line = line.strip().split('\t')
                if len(line) != 5:
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
            elif data_name == 'nyc':
                if index == 0:
                    continue
                #user_id,POI_id,POI_catid,POI_catid_code,POI_catname,latitude,longitude,timezone,UTC_time,local_time,day_of_week,norm_in_day_time,trajectory_id,norm_day_shift,norm_relative_time
                line = line.strip().split(',')
                if len(line) != 15:
                    continue
                loc = line[1]
                coordinate = float(line[5]), float(line[6])
            elif data_name == 'tky':
                if index == 0:
                    continue
                # user_id:token	venue_id:token	timestamp:float	longitude:float	latitude:float
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                loc = line[1]
                coordinate = float(line[4]), float(line[3])
            elif data_name == '4sq':
                #user_id  timestamp:2013-12-20T15:24:43Z	longitude:float	latitude:float	venue_id
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                loc = line[4]
                coordinate = float(line[2]), float(line[3])
            self.add_location(loc, coordinate)
        if min_loc_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_loc_freq:
                    self.add_location(loc, self.loc2gps[loc])
                    if loc not in self.loc2gps2:
                        self.loc2gps2[loc] = self.loc2gps[loc]
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
        check = []
        lat_list = []
        #for index,line in enumerate(open(data_path, encoding='ISO-8859-1')):
        for index,line in enumerate(open(data_path, encoding='UTF-8')):
            if data_name == 'weeplaces':
                # user, loc, time(2010-10-22T23:44:29), lat, lng
                # if index == 0:
                #     continue
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                user, loc, t, lat, lng = line
                lat = float(lat)
                lng = float(lng)
                #t = time.strptime(t, '%Y-%m-%dT%H:%M:%S')
                timestamp = float(t)#time.mktime(t)
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
            elif data_name == 'nyc':
                if index == 0:
                    continue
                # user_id,POI_id,POI_catid,POI_catid_code,POI_catname,latitude,longitude,timezone,UTC_time,local_time,day_of_week,norm_in_day_time,trajectory_id,norm_day_shift,norm_relative_time
                line = line.strip().split(',')
                if len(line) != 15:
                    continue
                user,loc,POI_catid,POI_catid_code,POI_catname,latitude,longitude,timezone,UTC_time,local_time,day_of_week,norm_in_day_time,trajectory_id,norm_day_shift,norm_relative_time = line
                # lat = float(latitude)
                # lng = float(longitude)
                # if loc not in self.loc2idx:
                #     continue
                # lat, lng = self.loc2gps2[loc]
                if latitude == self.loc2gps[loc][0]:
                    print(self.loc2gps[loc][0], latitude, self.loc2idx[loc])
                
                UTC_time = UTC_time.split('+')[0]
                t = time.strptime(UTC_time, '%Y-%m-%d %H:%M:%S')
                timestamp = time.mktime(t)
            elif data_name == 'tky':
                if index == 0:
                    continue
                # user_id:token	venue_id:token	timestamp:float	longitude:float	latitude:float
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                user, loc, time_, longitude, latitude = line
                lat = float(latitude)
                lng = float(longitude)
                timestamp = float(time_)
            elif data_name == '4sq':
                #user_id  timestamp:2013-12-20T15:24:43Z	longitude:float	latitude:float	venue_id
                line = line.strip().split('\t')
                if len(line) != 5:
                    continue
                user, t, lat, lng, loc = line
                lat = float(lat)
                lng = float(lng)
                t = time.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
                timestamp = time.mktime(t)

            if loc not in self.loc2idx:
                continue
            loc_idx = self.loc2idx[loc]
            if lat not in lat_list:
                lat_list.append(lat)
            quadkey = latlng2quadkey(lat, lng, map_level)
            if quadkey not in quadkey2idx:
                quadkey2idx[quadkey] = n_quadkey
                idx2quadkey[n_quadkey] = quadkey
                n_quadkey += 1
                check.append(quadkey)
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

        self.all_quadkeys = []

        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[2]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                self.all_quadkeys.append(region_quadkey_bigram)
                user_seq_array[u][i] = (check_in[0], check_in[1], region_quadkey_bigram, check_in[3], check_in[4])

        self.loc2quadkey = ['NULL']
        check1 = []
        for l in range(1, self.n_loc):
            lat, lng = self.idx2gps[l]
            quadkey = latlng2quadkey(lat, lng, map_level)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_bigram = quadkey_bigram.split()
            self.loc2quadkey.append(quadkey_bigram)
            self.all_quadkeys.append(quadkey_bigram)
            if quadkey not in check1:
                check1.append(quadkey)
        self.QUADKEY = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.QUADKEY.build_vocab(self.all_quadkeys)
        #print(len(self.loc2gps2), len(lat_list),"*************************list*********************")
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

            # n_instance = math.floor((i + max_len - 2) / max_len)
            # i = i - 1
            n_instance = math.floor((i + max_len - 1) / max_len)
            #i = i - 1
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
                    train_trg_seq = seq[max(i - (k + 1) * max_len, 1): i - k * max_len]
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
    
    # def spatial_temporal_matrix_building_(self, path):
    #     seq = self[0][:20]
    #     seq_loc = [r[1] for r in seq]
    #     seq_time = [r[3] for r in seq]
    #     print(seq_time)
    #     torch.set_printoptions(threshold=np.inf)
    #     temporal_matrix, spatial_matrix = build_st_matrix(self, seq, len(seq))
    #     print(temporal_matrix)


    def building_bar(self, st_matrix):
        temporal_max, temporal_min = 0, 0
        spatial_max, spatial_min = 0, 0
        print("building Spatial-Temporal maximum and minimum...")
        for u in tqdm(range(len(self.user_seq))):
            temporal_matrix, spatial_matrix = st_matrix[u]
            temporal_max_ = temporal_matrix.max()
            temporal_matrix_mask = temporal_matrix != 0
            temporal_min_ = torch.masked_select(temporal_matrix, temporal_matrix_mask).min()
            spatial_max_ = spatial_matrix.max()
            spatial_matrix_mask = spatial_matrix != 0
            spatial_min_ = torch.masked_select(spatial_matrix, spatial_matrix_mask).min()
            if temporal_max_ >= temporal_max:
                temporal_max = temporal_max_
            elif temporal_min_ < temporal_min:
                temporal_min = temporal_min_
            if spatial_max_ >= spatial_max:
                spatial_max = spatial_max_
            elif spatial_min_ < spatial_min:
                spatial_min = spatial_min_
        #temporal_matrix, spatial_matrix = st_matrix
        self.temporal_max = temporal_max
        self.spatial_max = spatial_max
        print("max temporal:", self.temporal_max)
        print("max spatial:", self.spatial_max)
        interval_num_temporal = 20
        interval_num_spatial = 20
        count_temporal = [0 for _ in range(interval_num_temporal)]
        count_spatial = [0 for _ in range(interval_num_spatial)]
        self.interval_temporal = (temporal_max - temporal_min) / interval_num_temporal
        self.interval_spatial = (spatial_max - spatial_min) / interval_num_spatial
        temporal_axis_x = [(temporal_min+(self.interval_temporal/2)+(i*self.interval_temporal)) for i in range(interval_num_temporal)]
        spatial_axis_x = [(spatial_min+(self.interval_spatial/2)+(i*self.interval_spatial)) for i in range(interval_num_spatial)]
        #print(temporal_max, temporal_min, spatial_max, spatial_min, interval_temporal, interval_spatial) #22443 19772 935.1250 988.6

        # print("building Spatial-Temporal Analysis Chart...")
        # for u in tqdm(range(len(self))):
        #     temporal_matrix, spatial_matrix = st_matrix[u]
        #     for x in range(temporal_matrix.size(0)):
        #         for y in range(x):
        #             if temporal_matrix[x,y] != temporal_max:
        #                 count_temporal[((temporal_matrix[x,y]-temporal_min)//interval_temporal).int().item()] += 1
        #             else:
        #                 count_temporal[-1] += 1
        #                 print("attention temporal")
        #             if spatial_matrix[x,y] != spatial_max:
        #                 count_spatial[((spatial_matrix[x,y]-spatial_min)//interval_spatial).int().item()] += 1
        #             else:
        #                 count_spatial[-1] += 1
        #                 print("attention spatial")
        # count = np.array([count_temporal, count_spatial])
        # np.save('count_24_20.npy', count)

        print("making plot bar...")
        count = np.load('count_20.npy', allow_pickle=True).tolist()
        #print(count,"count")
        count_temporal, count_spatial = count[0], count[1]
        count_temporal_max, count_temporal_min = max(count_temporal), min(count_temporal)
        count_spatial_max, count_spatial_min = max(count_spatial), min(count_spatial)
        fig = plt.figure(1)
        plt.style.use("ggplot")
        # label = ["a","b","c","d","e"]
        # x = [0,1,2,3,4]
        # y = [30,20,15,25,10]
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('temporal interval') # 指定x轴说明
        ax.set_ylabel('number of occurrences') # 指定y轴说明
        ax.set_title("Time period quantity chart")
        ax.set_xlim((temporal_min, temporal_max))
        ax.set_ylim((count_temporal_min, count_temporal_max))
        #print(temporal_axis_x, count_temporal)
        ax.bar(temporal_axis_x, count_temporal, width=0.25, label='time')#label='time'表示图例
        #ax.bar(x, y, label=label)
        ax.grid() #生成网格
        ax.legend() #显示图例，图例包括图标等
        # ax = fig.add_subplot(1,2,2)
        # ax.set_xlabel('spatial interval') # 指定x轴说明
        # ax.set_ylabel('number of occurrences') # 指定y轴说明
        # ax.set_title("Time period quantity chart")
        # ax.bar(spatial_axis_x, count_spatial, width=0.25, label='distance')#label='time'表示图例
        # ax.grid() #生成网格
        # ax.legend() #显示图例，图例包括图标等
        plt.show()
        plt.savefig("./plot/count_24_20_spatial.png")

    def building_bar_(self, st_matrix):
        temporal_max, temporal_min = 0, 0
        spatial_max, spatial_min = 0, 0
        print("building Spatial-Temporal maximum and minimum...")
        temporal_matrix, spatial_matrix = st_matrix[0]
        torch.set_printoptions(threshold=np.inf)
        #print(temporal_matrix[:2])

class PretrainData(LBSNData):
    def __init__(self, data_name, data_path, min_loc_freq, min_user_freq, map_level, args):
        super().__init__(data_name, data_path, min_loc_freq, min_user_freq, map_level, args)
        self.args = args
        self.n_user = self.n_user
        self.src_part_sequence = []
        self.trg_part_sequence = []
        self.dis_cluster_sequence = []
        self.user_sequence = []
        self.long_sequence = []
        self.time_sequence = []
        self.src_treatment_neg_sequence = []
        self.trg_treatment_neg_sequence = []
        self.t_mat_seq = []
        self.s_mat_seq = []
        self.seq_adj_i = []
        self.seq_adj_all = []
        self.adj_spatial_matrix_seq = [] 
        self.adj_temporal_matrix_seq = [] 
        self.adj_interest_matrix_seq = []
        self.adj_data_size_seq = []
        self.max_len = args.max_len
        self.mask_id = self.n_loc
        self.item_size = self.n_loc
        self.sampler = KNNSampler_treatment(args.loc_query_sys, args.treatment_neg_num, args.user_visited_locs, 'training', True)
        #self.building_bar(args.st_matrix)
        self.split_sequence(args.st_matrix)
    def __len__(self):
        return len(self.src_part_sequence)
    def split_sequence(self, st_matrix):
        self.su, self.sl, self.tu, self.tl = 0, 0, 0, 0
        self.args.cluster_nodes = 3
        np.set_printoptions(threshold = np.inf) 
        print("split pretrain sequence")
        temporal_max, temporal_min = 0, 0
        spatial_max, spatial_min = 0, 0

        for user,infos in enumerate(tqdm(self.user_seq, ncols=70)):
            temporal_matrix, spatial_matrix = st_matrix[user]
            n_loc_user = temporal_matrix.size(0)
            seq_len = len(infos)
            # if user < 4281:
            #     continue
            #print(temporal_matrix.shape, spatial_matrix.shape, "shape")
            # if user == 10:
            #     break
            # interval_num = 20
            # temporal_matrix1 = torch.zeros_like(temporal_matrix)
            # temporal_matrix2 = torch.zeros_like(temporal_matrix)
            # spatial_matrix1 = torch.zeros_like(spatial_matrix)
            # spatial_matrix2 = torch.zeros_like(spatial_matrix)
            # for i in reversed(range(interval_num)):
            #     if i == (interval_num - 1):
            #         temporal_matrix1 = (temporal_matrix<=((i+1)*self.interval_temporal)) * (i+1)
            #     else:
            #         temporal_matrix2 = (temporal_matrix<=((i+1)*self.interval_temporal)) * 1
            #         temporal_matrix1 = temporal_matrix1 - temporal_matrix2
            #     if i == (interval_num - 1):
            #         spatial_matrix1 = (spatial_matrix<=((i+1)*self.interval_spatial)) * (i+1)
            #     else:
            #         spatial_matrix2 = (spatial_matrix<=((i+1)*self.interval_spatial)) * 1
            #         spatial_matrix1 = spatial_matrix1 - spatial_matrix2
            # temporal_matrix = temporal_matrix1
            # spatial_matrix = spatial_matrix1
            
            temporal_max_ = temporal_matrix.max()
            temporal_matrix_mask = temporal_matrix != 0
            temporal_min_ = torch.masked_select(temporal_matrix, temporal_matrix_mask)
            if temporal_min_.shape[0] != 0:
                # print(temporal_min_.shape,"111")
                # print(temporal_min_,"222")
                temporal_min_ = temporal_min_.min()
            else:
                temporal_min_ = 0
            
            spatial_max_ = spatial_matrix.max()
            spatial_matrix_mask = spatial_matrix != 0
            spatial_min_ = torch.masked_select(spatial_matrix, spatial_matrix_mask)
            #print(spatial_min_.shape)
            if spatial_min_.shape[0] != 0:
                # print(spatial_min_.shape,"111")
                # print(spatial_min_,"222")
                spatial_min_ = spatial_min_.min()
            else:
                spatial_min_ = 0
            
            if temporal_max_ >= temporal_max:
                temporal_max = temporal_max_
            elif temporal_min_ < temporal_min:
                temporal_min = temporal_min_
            if spatial_max_ >= spatial_max:
                spatial_max = spatial_max_
            elif spatial_min_ < spatial_min:
                spatial_min = spatial_min_
            
            self.tu, self.tl, self.su, self.sl = temporal_max, temporal_min, spatial_max, spatial_min

            #dis_cluster = poi_distance_cluster(seq_len, spatial_matrix.numpy(), infos, self.args.cluster_nodes)
            #print(dis_cluster)
            #data_cluster, median_coor = self.func_listdata(dis_cluster, infos, self.args.cluster_nodes, seq_len)
            # if temporal_matrix.max() > self.tu:
            #     self.tu = temporal_matrix.max()
            # if temporal_matrix.min() < self.tl:
            #     self.tl = temporal_matrix.min()
            # if spatial_matrix.max() > self.su:
            #     self.su = spatial_matrix.max()
            # if spatial_matrix.min() < self.sl:
            #     self.sl = spatial_matrix.min()
            user_seq = []
            user_ = []
            time_ = []

            for info in infos:
                user_seq.append(info[1])
                self.long_sequence.append(info[1])
                user_.append(info[0])
                time_.append(info[3])
            
            input_ids_all = user_seq[:-2]
            user_all = user_[:-2]
            src = input_ids_all#[:-1]
            trg = input_ids_all#[1:]
            #dis_cluster_src = dis_cluster[:-1]
            src_time = time_#[:-1]
            # lon = np.mean(np.array([self.idx2gps[i][0] for i in input_ids_all]))
            # lat = np.mean(np.array([self.idx2gps[i][1] for i in input_ids_all]))
            # median_coor = (lon, lat)
            #median_time = np.mean(np.array(time_))
            #sample treatment
            treatment_neg_ = self.sampler(src, self.args.treatment_neg_num, user=user_all[0])
            #treatment_neg_trg_ = self.sampler(trg, self.args.treatment_neg_num, user=user_all[0])
            n_instance = math.ceil((len(input_ids_all) - 2) / (self.max_len+1))

            for n in range(n_instance):
                src_seq = src[n*self.max_len:min((n+1)*self.max_len, len(src))]
                trg_seq = trg[n*self.max_len:min((n+1)*self.max_len, len(src))]
                #dis_cluster_seq_ = dis_cluster_src[n*self.max_len:min((n+1)*self.max_len, len(src))] 
                #dis_cluster_seq = list(map(lambda x:x+1, dis_cluster_seq_))  #empty pad 0 position
                user = user_all[n*self.max_len:min((n+1)*self.max_len, len(src))]
                src_time_seq = src_time[n*self.max_len:min((n+1)*self.max_len, len(src))]
                treatment_neg = treatment_neg_[n*self.max_len:min((n+1)*self.max_len, len(src))]

                #treatment_neg_trg = treatment_neg_trg_[n*self.max_len:min((n+1)*self.max_len, len(src))]
                t_mat, s_mat = extract_sub_matrix(n*self.max_len,
                                                  min((n+1)*self.max_len,
                                                       len(src)),
                                                  len(src_seq),
                                                  self.max_len,
                                                  temporal_matrix,
                                                  spatial_matrix)
                for i in range(len(src_seq)):
                    self.src_part_sequence.append(src_seq[:i+1])
                    self.trg_part_sequence.append(trg_seq[:i+1])
                    #self.dis_cluster_sequence.append(dis_cluster_seq[:i+1])
                    self.user_sequence.append(user[:i+1])
                    self.time_sequence.append(src_time_seq[:i+1])
                    self.src_treatment_neg_sequence.append(treatment_neg[:i+1])
                    #self.trg_treatment_neg_sequence.append(treatment_neg_trg[:i+1])
                    sub_t_mat, sub_s_mat = extract_sub_matrix_coo(0,
                                                                  i+1,
                                                                  i+1,
                                                                  self.max_len,
                                                                  t_mat,
                                                                  s_mat)
                    self.t_mat_seq.append(sub_t_mat)
                    self.s_mat_seq.append(sub_s_mat)
                    # sequence_adj_i, sequence_adj_all, adj_spatial_matrix, adj_temporal_matrix, adj_interest_matrix, adj_data_size = build_adj_matrix_coo(self, src_seq[:i+1], src_time_seq[:i+1], median_coor, median_time, self.max_len, dis_cluster_seq)
                    # self.seq_adj_i.append(sequence_adj_i)
                    # self.seq_adj_all.append(sequence_adj_all)
                    # self.adj_spatial_matrix_seq.append(adj_spatial_matrix)
                    # self.adj_temporal_matrix_seq.append(adj_temporal_matrix)
                    # self.adj_interest_matrix_seq.append(adj_interest_matrix)
                    # self.adj_data_size_seq.append(adj_data_size)

    def __getitem__(self, index):
        sequence = self.src_part_sequence[index] # pos_items
        target = self.trg_part_sequence[index] #trg_items
        #dis_cluster_ = self.dis_cluster_sequence[index]
        user = self.user_sequence[index]
        seq_t_mat = self.t_mat_seq[index]
        seq_s_mat = self.s_mat_seq[index]
        # adj_seq_i = self.seq_adj_i[index]
        # adj_seq_all = self.seq_adj_all[index]
        # adj_spatial_matrix = self.adj_spatial_matrix_seq[index]
        # adj_temporal_matrix = self.adj_temporal_matrix_seq[index]
        # adj_interest_matrix = self.adj_interest_matrix_seq[index]
        # adj_data_size = self.adj_data_size_seq[index]
        treatment_neg = self.src_treatment_neg_sequence[index]
        #treatment_neg_trg = self.trg_treatment_neg_sequence[index]
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
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = sequence + [0] * pad_len
        neg_items = neg_items + [0] * pad_len
        target_items = target + [0] * pad_len
        #dis_cluster = dis_cluster_ + [0] * pad_len
        masked_segment_sequence = masked_segment_sequence  + [0] * pad_len
        pos_segment = pos_segment  + [0] * pad_len
        neg_segment = neg_segment  + [0] * pad_len


        neg_treatment_pad = torch.tensor([0]*self.args.treatment_neg_num).unsqueeze(0).repeat(pad_len, 1)
        neg_treatment = torch.cat([neg_treatment_pad, treatment_neg], dim=0)
        #neg_treatment_trg = torch.cat([neg_treatment_pad, treatment_neg_trg], dim=0)
        


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
                       neg_treatment,
                    #    neg_treatment_trg,
                       seq_t_mat,
                       seq_s_mat,
                    #    adj_seq,
                    #    adj_spatial_matrix,
                    #    adj_temporal_matrix,
                    #    adj_interest_matrix,
                    #    adj_data_size,
                    #    torch.tensor(dis_cluster, dtype=torch.long),
                    #    torch.tensor(adj_seq_i, dtype=torch.long),
                    #    torch.tensor(adj_seq_all, dtype=torch.long),
                    #    torch.tensor(adj_spatial_matrix, dtype=torch.long),
                    #    torch.tensor(adj_temporal_matrix, dtype=torch.long),
                    #    torch.tensor(adj_interest_matrix, dtype=torch.long),
                    #    torch.tensor(adj_data_size, dtype=torch.long),
                       )
        return cur_tensors

    def func_listdata(self, idx, poi, cluster_nodes, l_max):
        data_cluster = []
        median_coor = [(0,0)]
        for i in range(cluster_nodes):
            listdata = []
            for j in range(l_max):
                if idx[j] == i:
                    listdata.append(poi[j][1])
            data_cluster.append(listdata)
        #data_cluster = np.stack(data_cluster,0)
        #data_cluster = np.array(data_cluster)
        for user_seq in range(cluster_nodes):
            if isinstance(user_seq, int):
                user_seq = [user_seq]
            lon = np.mean(np.array([self.idx2gps[i][0] for i in user_seq]))
            lat = np.mean(np.array([self.idx2gps[i][1] for i in user_seq]))
            median = (lon, lat)
            median_coor.append(median)
        #print("data_cluster")
        return data_cluster, median_coor



if __name__ == "__main__":
    # Setting paths
    path_prefix = './'
    parser = argparse.ArgumentParser()
    # data management
    parser.add_argument('--data_name', default='brightkite', type=str, help='the dataset under ./data/<dataset.txt> to load')
    parser.add_argument('--max_len', default=100, type=int, help='the max length of the sequence')
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    print("Dataset: ", args.data_name)
    print("paramters", args)
    if args.data_name == 'weeplaces':
        raw_data_path = path_prefix + args.data_name + '.csv'
    else:
        raw_data_path = path_prefix + args.data_name + '.txt'
    clean_data_path = path_prefix + args.data_name + '.data'
    loc_query_path = path_prefix + args.data_name + '_loc_query.pkl'
    matrix_path = path_prefix + args.data_name + '_st_matrix.data'
    matrix_path_onlyst = path_prefix + args.data_name + '_onlyst_matrix.data'
    matrix_path_decayst = path_prefix + args.data_name + '_decayst_matrix.data'
    log_path = path_prefix + args.data_name + '.txt'
    result_path = path_prefix + args.data_name + '.txt'
    args.candidate_mat_path = path_prefix + args.data_name + '_candidate_st_matrix.data'
    # save model args

    # Data Process details
    min_loc_freq = 10
    min_user_freq = 20
    map_level = 15
    n_nearest = 2000
    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        dataset = LBSNData(args.data_name, raw_data_path, min_loc_freq, min_user_freq, map_level, args)
        serialize(dataset, clean_data_path)
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    args.item_size = dataset.n_loc + 1
    args.mask_id = dataset.n_loc
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))

    torch.cuda.empty_cache()
    # Searching nearest POIs
    quadkey_processor = dataset.QUADKEY
    loc2quadkey = dataset.loc2quadkey
    args.loc_query_sys = Loc_Query_System()
    if os.path.exists(loc_query_path):
        args.loc_query_sys.load(loc_query_path)
    else:
        args.loc_query_sys.build_tree(dataset)
        args.loc_query_sys.prefetch_n_nearest_locs(n_nearest)
        args.loc_query_sys.save(loc_query_path)
        args.loc_query_sys.load(loc_query_path)


    print("Building Spatial-Temporal Relation Matrix")
    # Building Spatial-Temporal Relation Matrix
    if os.path.exists(matrix_path):
        args.st_matrix = unserialize(matrix_path)
    else:
        dataset.spatial_temporal_matrix_building(matrix_path)
        args.st_matrix = unserialize(matrix_path)
    #dataset.spatial_temporal_matrix_building_(matrix_path)
    print("Building Spatial-Temporal bar")
    dataset.building_bar(args.st_matrix)