import torch
import torch.nn as nn
import numpy as np


class KNNSampler(nn.Module):
    def __init__(self, loc_query_sys, n_nearest, user_visited_locs, state, exclude_visited):
        nn.Module.__init__(self)
        self.loc_query_sys = loc_query_sys
        self.n_nearest = n_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        self.state = state

    def forward(self, trg_seq, num_negs, user):
        neg_samples = []
        if self.state=='training':
            for check_in in trg_seq:
                trg_loc = check_in[1]
                nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
                samples = []
                if self.exclude_visited:
                    for _ in range(num_negs):
                        sample = np.random.choice(nearby_locs)
                        while sample in self.user_visited_locs[user]:
                            sample = np.random.choice(nearby_locs)
                        samples.append(sample)
                else:
                    samples = np.random.choice(nearby_locs, size=num_negs, replace=True)
                neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        elif self.state=='evaluating':
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        else:
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        return neg_samples


class KNNSampler_treatment(nn.Module):
    def __init__(self, loc_query_sys, n_nearest, user_visited_locs, state, exclude_visited):
        nn.Module.__init__(self)
        self.loc_query_sys = loc_query_sys
        self.n_nearest = n_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        self.state = state

    def forward(self, trg_seq, num_negs, user):
        neg_samples = []
        num = 0
        if self.state=='training':
            for check_in in trg_seq:
                trg_loc = check_in
                nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
                samples = []
                if self.exclude_visited:
                    for _ in range(num_negs):
                        sample = np.random.choice(nearby_locs)
                        while sample in self.user_visited_locs[user]:
                            sample = np.random.choice(nearby_locs)
                            num += 1
                            if num == 10000:
                                sample = 0
                                break
                        samples.append(sample)
                        num = 0
                else:
                    samples = np.random.choice(nearby_locs, size=num_negs, replace=True)
                neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        elif self.state=='evaluating':
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        else:
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        return neg_samples
    
class KNNSampler_baseline(nn.Module):
    def __init__(self, loc_query_sys, n_nearest, user_visited_locs, state, exclude_visited):
        nn.Module.__init__(self)
        self.loc_query_sys = loc_query_sys
        self.n_nearest = n_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        self.state = state

    def forward(self, trg_seq, num_negs, user):
        neg_samples = []
        if self.state=='training':
            
            trg_loc = trg_seq
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            if self.exclude_visited:
                for _ in range(num_negs):
                    sample = np.random.choice(nearby_locs)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(nearby_locs)
                    samples.append(sample)
            else:
                samples = np.random.choice(nearby_locs, size=num_negs, replace=True)
            neg_samples.append(samples)
            #neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        elif self.state=='evaluating':
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        else:
            trg_loc = trg_seq[0][1]
            nearby_locs = self.loc_query_sys.get_k_nearest_locs(trg_loc, k=self.n_nearest)
            samples = []
            nearby_loc_idx = 0
            for _ in range(num_negs):
                sample = nearby_locs[nearby_loc_idx]
                while sample in self.user_visited_locs[user]:
                    nearby_loc_idx += 1
                    sample = nearby_locs[nearby_loc_idx]
                nearby_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        return neg_samples