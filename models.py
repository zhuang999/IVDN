from code import interact
from selectors import EpollSelector
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
import copy
from time_aware_pe import TAPE
from attn_modules import *
import torch.nn.functional as F
from utils import *
from TGCN import TGCN
from build_graph import build_global_POI_checkin_graph
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid, 'elu':F.elu}

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
 

class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)

class rnn_factory(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(rnn_factory, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len).cuda()#, device)
        c = self.c_strategy.on_init(user_len).cuda()#, device)
        return (h,c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)

def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return rnn_factory(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)

class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).cuda()#.to(device)

    def on_reset(self, user):
        return self.h0

def get_var(rep, dim_inp, dim_out, drop_flag, device):
    w = nn.Linear(dim_inp, dim_out).cuda()#.to(device)
    w.weight.data.normal_(mean=0, std=0.01)
    w.bias.data.fill_(0.0)
    rep = w(rep)
    if drop_flag:
        rep = nn.Dropout(0.5)(nn.ELU()(rep))
    return rep, w

def get_var_matrix(rep, dim_inp, dim_out, drop_flag, device):
    w = nn.Parameter(torch.empty((dim_inp, dim_out), requires_grad=True))
    b = nn.Parameter(torch.empty((1, dim_inp), requires_grad=True))
    nn.init.uniform_(w)
    nn.init.uniform_(b)
    # w.weight.data.normal_(mean=0, std=0.01)
    # w.bias.data.fill_(0.0)
    rep = torch.mm(rep, w) + b
    if drop_flag:
        rep = nn.Dropout(0.5)(nn.ELU()(rep))
    return rep, w

class FFN(nn.Module):
    def __init__(self, dim_in_net, dim_out_net, device):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(dim_in_net, dim_out_net).cuda()#.to(device)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(dim_out_net, dim_out_net).cuda()#.to(device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, rep, dim_inp, dim_out, drop_flag, device):
        rep = self.w_1(rep)
        if drop_flag:
            rep = nn.Dropout(0.5)(nn.ELU()(rep))
        return rep, self.w_1

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x





class STDE(nn.Module):
    def __init__(self, n_loc, n_user, n_quadkey, features, exp_factor, k_t, k_g, depth, dropout, device, args):
        super(STDE, self).__init__()
        self.item_embeddings = nn.Embedding(n_loc, features, padding_idx=0)  #item_size=n_loc + 1 (n_loc includes mask_id and pad_id  mask_id=n_loc)
        self.position_embeddings = nn.Embedding(100, features)
        self.time_embeddings = nn.Embedding(49, features, padding_idx=0)
        self.temporal_embeddings = nn.Embedding(21, features, padding_idx=0)
        self.spatial_embeddings = nn.Embedding(21, features, padding_idx=0)
        # if args.pre_model_name == 'baseline':
        #     poi_emb = torch.load("brightkite-pretrain_skipgram-epochs-100.pt")
        #     poi_emb = torch.FloatTensor(poi_emb.cpu().weight)
        #     self.item_embeddings = nn.Embedding.from_pretrained(poi_emb)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(features, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.device = device
        self.features = features
        self.hidden_size = features
        self.emb_user = nn.Embedding(n_user, features, padding_idx=0)
        self.n_loc = n_loc - 1
        self.mask_id = n_loc - 1
        self.rep_net_layer = 2
        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(features, features)
        self.mip_norm = nn.Linear(features, features)
        self.map_norm = nn.Linear(features, features)
        self.sp_norm = nn.Linear(features, features)
        self.z_linear = nn.Linear(features, features)
        self.c_linear = nn.Linear(features, features)
        self.criterion = nn.BCELoss(reduction='none')
        self.interest_tgcn = TGCN(features, args)
        self.spatial_tgcn = TGCN(features, args)
        self.temporal_tgcn = TGCN(features, args)
        self.mlp_inp = nn.Sequential(
            nn.Linear(features, features),
            nn.ELU(),
            nn.Linear(features, features))
        self.mlp_var = nn.Sequential(
            nn.Linear(features, features),
            nn.ELU(),
            nn.Linear(features, features),
            nn.Tanh())
        self.init_weights(self.mlp_inp)
        self.init_weights(self.mlp_var)

        #init weight
        # self.apply(self.init_weights)
        # self.init = tf.contrib.layers.xavier_initializer()
        #print(n_quadkey,"n_quadkey")
        self.emb_quadkey = Embedding(n_quadkey, features, True, True)
        self.geo_encoder_layer = GeoEncoderLayer(features, exp_factor, dropout)
        self.geo_encoder = GeoEncoder(features, self.geo_encoder_layer, depth=2)

        self.tape = TAPE(dropout)

        self.k_t = torch.tensor(k_t)
        self.k_g = torch.tensor(k_g)

        self.inr_awa_attn_layer = InrEncoderLayer(features * 2, exp_factor, dropout, args)
        self.inr_awa_attn_block = InrEncoder(features * 2, self.inr_awa_attn_layer, depth)

        self.inr_awa_attn = InrEncoder(features, self.inr_awa_attn_layer, depth)

        self.trg_awa_attn_decoder = TrgAwaDecoder(features * 2, dropout)

        self.self_attn_layer = InrEncoderLayer(features, exp_factor, dropout, args)
        self.self_attn = InrEncoder(features, self.self_attn_layer, depth)

        self.self_attn_layer_iv = InrEncoderLayer(features*2, exp_factor, dropout, args)
        self.self_attn_iv = InrEncoder(features*2, self.self_attn_layer_iv, depth)

        self.h0_strategy = create_h0_strategy(features, True)
        self.model = nn.LSTM(features, features, batch_first=True)
        self.flashback_fc = nn.Linear(2*features, features) #n_loc) # create outputs in lenght of locations

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(n_loc+1, dims+dims)
        self.b2 = nn.Embedding(n_loc+1, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.emb_user.weight.data.normal_(0, 1.0 / self.emb_user.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        """additional add embedding moudle"""
        #self.init_graph()


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        '''
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        '''
        sequence_output = self.aap_norm(sequence_output) # [B L H]
        sequence_output = sequence_output.view([-1, self.features, 1]) # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1,self.hidden_size])) # [B*L H]
        target_item = target_item.view([-1,self.hidden_size]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, self.features, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        '''
        :param context: [B H]
        :param segment: [B H]
        :return:
        '''
        context = self.sp_norm(context)
        score = torch.mul(context, segment) # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1)) # [B]

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()#, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def add_position(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()#, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).repeat(sequence.size(0),1)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

    def rep_net(self, inp, dim_in, dim_out, layer):
        """representation network"""
        rep, w_, b_ = [inp], [], []
        for i in range(layer):
            if i == (layer - 1):
                drop_flag = True
            else:
                drop_flag = False
            dim_in_net = dim_in if (i == 0) else dim_out
            dim_out_net = dim_out
            rep_, linear = FFN(dim_in_net, dim_out_net, self.args.device)(rep[i], dim_in_net, dim_out_net, drop_flag, self.args.device)
            w_.append(linear.weight.data)
            b_.append(linear.bias.data)
            rep.append(rep_)
        return rep[-1], w_, b_

    # def mi_net_(self, input, output, x, mi_min_max, name=None):
    #     """Mutual information network"""
    #     mu = self.mlp_inp(input)
    #     logvar = self.mlp_var(input)
    #     new_order = torch.randperm(output.size(1))
    #     output_rand = output[:, new_order].view(output.size())

    #     """get likelihood"""

    #     loglikeli = torch.sum(- \
    #                               torch.mean(torch.sum(-(output - mu) ** 2 /
    #                                                    torch.exp(logvar) - logvar, dim=-1)))

    #     """get positive and negative"""
    #     pos = - (mu - output) ** 2 / torch.exp(logvar)
    #     neg = - (mu - output_rand) **2 / torch.exp(logvar)

    #     if name == 'zy':
    #         x_rand = x[:, new_order].view(x.size())
    #         #using RBF kernel to measure distance
    #         w = torch.exp(-torch.square(x - x_rand) / (2 * self.args.sigma ** 2))
    #         w_soft = nn.Softmax(dim=1)(w)
    #     else:
    #         w_soft = 1. / (self.n_loc - 1)

    #     """Get estimation of mutual information."""
    #     if mi_min_max == 'min':
    #         pn = -1.
    #     elif mi_min_max == 'max':
    #         pn = 1.
    #     else:
    #         raise ValueError
    #     bound = torch.sum(torch.mean(pn * torch.sum(w_soft * (pos - neg), dim=-1), dim=-1))

    #     return loglikeli, bound, mu, logvar, w_soft

    # def mi_net(self, input, output, x, mi_min_max, mask_label, name=None):
    #     """Mutual information network"""
    #     mu = self.mlp_inp(input)
    #     logvar = self.mlp_var(input)

    #     """get likelihood"""
    #     loglikeli = self.masked_item_prediction(output, mu)
    #     loglikeli = torch.sigmoid(loglikeli)
    #     mask_li = mask_label
    #     loss_li = self.criterion(loglikeli, torch.ones_like(loglikeli, dtype=torch.float32)) #[B, L]
    #     #loss_li = loss_li * mask_li.flatten()
    #     loglikeli = torch.sum(loss_li)# / mask_li.sum() #[B, L]

    #     new_order = torch.randperm(self.args.max_len)
    #     output = output.view(input.size(0), self.args.max_len, -1, self.hidden_size)
    #     neg = output[:, new_order].view(input.size())
    #     """Get estimation of mutual information."""
    #     if mi_min_max == 'min':
    #         pn = -1.
    #     elif mi_min_max == 'max':
    #         pn = 1.
    #     else:
    #         raise ValueError
    #     """get likelihood"""
    #     pos_score = self.masked_item_prediction(mu, output)
    #     neg_score = self.masked_item_prediction(mu, neg)
    #     if name == 'zy':
    #         x = x.view(input.size(0), self.args.max_len, -1, self.hidden_size)
    #         x_rand = x[:, new_order].view(x.size())
    #         w = self.masked_item_prediction(x, x_rand).view([input.size(0),-1])  #[B,L]
    #         w = w.masked_fill(mask_label == 0.0, -1e9)
    #         w_soft = nn.Softmax(dim=1)(w).flatten() / (2 * self.args.sigma ** 2)
    #         # x_rand = x[:, new_order].view(x.size())
    #         # #using RBF kernel to measure distance
    #         # w = torch.exp(-torch.square(x - x_rand) / (2 * self.args.sigma ** 2))
    #         # w_soft = torch.sum(nn.Softmax(dim=1)(w),dim=-1).flatten()
    #     else:
    #         w_soft = 1. / (self.n_loc - 1)
    #     mip_distance = torch.sigmoid(pos_score - neg_score)
    #     mask_bound = mask_label
    #     #bound = torch.sum(pn * w_soft * (self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))*mask_bound.flatten()),dim=-1) / mask_bound.sum()#[B, L]
    #     # bound = torch.sum(torch.mean(pn * w_soft * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)),dim=-1)) #[B, L]
    #     #if self.args.soft:
    #     #bound = torch.sum(torch.mean(pn * w_soft * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)),dim=-1)) #[B, L]
    #     bound = torch.sum(pn * w_soft * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))) #[B, L]
    #     # else:
    #     #     bound = torch.sum(torch.mean(pn * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)),dim=-1)) #[B, L]

    #     # if self.args.soft:
    #     #     bound = torch.sum(pn * w_soft * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)),dim=-1) #[B, L]
    #     # else:
    #     #     bound = torch.sum(pn * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)),dim=-1) #[B, L]

    #     return loglikeli, bound, mu, logvar, w_soft

    def mi_net(self, input, output, x, mi_min_max, mask_label, name=None):
        """Mutual information network"""
        mu = self.mlp_inp(input)
        logvar = self.mlp_var(input)

        """get likelihood"""
        loglikeli = self.masked_item_prediction(output, mu)
        loglikeli = torch.sigmoid(loglikeli)
        mask_li = mask_label
        loss_li = self.criterion(loglikeli, torch.ones_like(loglikeli, dtype=torch.float32)) #[B, L]
        #loss_li = loss_li * mask_li.flatten()
        loglikeli = torch.sum(loss_li)# / mask_li.sum() #[B, L]

        new_order = torch.randperm(self.args.max_len)
        output = output.view(input.size(0), self.args.max_len, -1, self.hidden_size)
        #mu = mu.view(input.size(0), self.args.max_len, -1, self.hidden_size)
        neg = output[:, new_order].view(input.size())
        """Get estimation of mutual information."""
        if mi_min_max == 'min':
            pn = -1.
        elif mi_min_max == 'max':
            pn = 1.
        else:
            raise ValueError
        """get likelihood"""
        pos_score = self.masked_item_prediction(output, mu)
        neg_score = self.masked_item_prediction(output, neg)
        if name == 'zy':
            x = x.view(input.size(0), self.args.max_len, -1, self.hidden_size)
            x_rand = x[:, new_order].view(x.size())
            w = self.masked_item_prediction(x, x_rand).view([input.size(0),-1])  #[B,L]
            w = w.masked_fill(mask_label == 0.0, -1e9)
            #w = torch.exp(torch.sigmoid(w)#/ (2 * self.args.sigma ** 2))
            #w_soft = nn.Softmax(dim=1)(w).flatten()
            w_soft = nn.Softmax(dim=1)(w).flatten() / (2 * self.args.sigma ** 2)
            # x_rand = x[:, new_order].view(x.size())
            # #using RBF kernel to measure distance
            # w = torch.exp(-torch.square(x - x_rand) / (2 * self.args.sigma ** 2))
            # w_soft = torch.sum(nn.Softmax(dim=1)(w),dim=-1).flatten()
        else:
            w_soft = 1. / (self.n_loc - 1)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mask_bound = mask_label
        bound = torch.sum(pn * w_soft * self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))) #[B, L]


        return loglikeli, bound, mu, logvar, w_soft

    def layer_out(self, inp, w, b, flag):
        """Set up activation function and dropout for layers."""
        out = torch.matmul(inp, w) + b
        if flag:
            return F.dropout(nn.ELU(out), p=self.args.dropout)
        else:
            return out
    
    def calculate_calibration(self, x, y):
        """Loss of y prediction."""
        """Mutual information network"""
        mu = self.mlp_inp(self.y_pre)

        new_order = torch.randperm(self.args.max_len)
        output = y.view(self.y_pre.size(0), self.args.max_len, -1, self.hidden_size)
        neg = output[:, new_order].view(self.y_pre.size())
        """get likelihood"""
        pos_score = self.masked_item_prediction(mu, output)
        neg_score = self.masked_item_prediction(mu, neg)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        #mask_bound = mask_label

        self.loss_cx2y = torch.sum(self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))) #[B, L]


        mu = self.mlp_inp(self.x_pre)

        new_order = torch.randperm(self.args.max_len)
        output = x.view(self.x_pre.size(0), self.args.max_len, -1, self.hidden_size)
        neg = output[:, new_order].view(self.x_pre.size())
        """get likelihood"""
        pos_score = self.masked_item_prediction(mu, output)
        neg_score = self.masked_item_prediction(mu, neg)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        #mask_bound = mask_label

        self.loss_zc2x = torch.sum(self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))) #[B, L]


    def calculate_loss(self, x, y, mask_label):
        """Get loss."""
        self.calculate_calibration(x, y)
        # self.loss_cx2y = torch.tensor(0.0)
        # self.loss_zc2x = torch.tensor(0.0)
        # """Loss of y prediction."""
        # y_score = self.masked_item_prediction(self.y_pre, y)
        # y_distance = torch.sigmoid(y_score)
        # mask_y = mask_label  #[B, L]
        # y_loss = self.criterion(y_distance, torch.ones_like(y_distance, dtype=torch.float32)) #[B, L] 
        # #y_loss = y_loss * mask_y.flatten()
        # self.loss_cx2y = torch.sum(y_loss)# / mask_y.sum()
        # #self.loss_cx2y = torch.mean(torch.square(y - self.y_pre))
        # #self.loss_cx2y = torch.tensor(0.0)


        # """Loss of t prediction."""
        # x_score = self.masked_item_prediction(self.x_pre, x)
        # x_distance = torch.sigmoid(x_score)
        # mask_x = mask_label  #[B, L]
        # x_loss = self.criterion(x_distance, torch.ones_like(x_distance, dtype=torch.float32)) #[B, L] 
        # #x_loss = x_loss * mask_x.flatten()
        # self.loss_zc2x = torch.sum(x_loss)# / mask_x.sum()
        # #self.loss_zc2x = torch.mean(torch.square(x - self.x_pre))
        # #self.loss_zc2x = torch.tensor(0.0)

        

        """Loss of network regularization."""
        def w_reg(w):
            """Calculate l2 loss of network weight."""
            w_reg_sum = 0
            for w_i in range(len(w)):
                w_reg_sum = w_reg_sum + (torch.sum(w[w_i]**2) / 2)
            return w_reg_sum
        for index,(w_z,w_c) in enumerate(zip(self.w_z, self.w_c)):
            if index == 0:
                self.orth_reg = (torch.mean(w_z, dim=-1) - 1)**2 + (torch.mean(w_c, dim=-1) - 1)**2 
            else:
                orth_reg = (torch.mean(w_z, dim=-1) - 1)**2 + (torch.mean(w_c, dim=-1) - 1)**2 
                self.orth_reg = self.orth_reg * orth_reg
        self.loss_reg = (w_reg(self.w_z) + w_reg(self.w_c) +
                         w_reg(self.w_emb) + w_reg(self.w_x) + w_reg(self.w_y)) / 5. #+ torch.sum(self.orth_reg)
        
        for index,(w_z,w_c) in enumerate(zip(self.w_z, self.w_c)):
            if index == 0:
                self.orth = torch.sum(torch.mean(w_z, dim=-1) * torch.mean(w_c, dim=-1))
            else:
                orth = torch.sum(torch.mean(w_z, dim=-1) * torch.mean(w_c, dim=-1))
                self.orth = self.orth * orth
        """Losses."""
        self.loss_lld = self.args.coefs['coef_lld_zy'] * self.lld_zy + \
                        self.args.coefs['coef_lld_cx'] * self.lld_cx + \
                        self.args.coefs['coef_lld_zx'] * self.lld_zx + \
                        self.args.coefs['coef_lld_cy'] * self.lld_cy + \
                        self.args.coefs['coef_lld_zc'] * self.lld_zc

        self.loss_bound = self.args.coefs['coef_bound_zy'] * self.bound_zy + \
                          self.args.coefs['coef_bound_cx'] * self.bound_cx + \
                          self.args.coefs['coef_bound_zx'] * self.bound_zx + \
                          self.args.coefs['coef_bound_cy'] * self.bound_cy + \
                          self.args.coefs['coef_bound_zc'] * self.bound_zc + \
                          self.args.coefs['coef_reg'] * self.loss_reg

        self.loss_2stage = self.args.coefs['coef_cx2y'] * self.loss_cx2y + \
                           self.args.coefs['coef_zc2x'] * self.loss_zc2x + \
                           self.args.coefs['coef_reg'] * self.loss_reg + torch.sum(self.orth)

        return self.loss_cx2y, self.loss_zc2x, self.loss_reg, self.loss_lld, self.loss_bound, self.loss_2stage, torch.sum(self.orth_reg)

    def pretrain(self, masked_item_sequence, pos_items,  neg_items,
                 masked_segment_sequence, pos_segment, neg_segment, neg_treatment, y, t_mat, s_mat):#, dis_cluster, adj_seq_i, adj_seq_all,  adj_spatial_matrix, adj_temporal_matrix, adj_interest_matrix, adj_data_size):#, neg_treatment, neg_treatment_trg
        #in order to match target(pos and neg(k)), repeat src
        #(b, n * (1 + k))
        x_neg_treatment = torch.cat([pos_items.unsqueeze(-1), neg_treatment], dim=-1).reshape(pos_items.size(0), -1)
        # y_neg_treatment = torch.cat([y.unsqueeze(-1), neg_treatment_trg], dim=-1).reshape(x.size(0), -1)
        # (b, n * (1 + k), d)
        x = self.item_embeddings(x_neg_treatment)
        y = self.item_embeddings(pos_items)

        label_mask = ((x_neg_treatment != 0) * 1).float()

        # Encode masked sequence
        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)
        
        encoded_layers = self.item_encoder(sequence_emb,
                                           sequence_mask, 
                                           output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]
        
        
        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)

        
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)) #[B, L]
        # only compute loss at non-masked position
        mip_mask = (masked_item_sequence == self.mask_id).float()  #[B, L]
        mip_loss = torch.sum(mip_loss * mip_mask.flatten()) #/ mip_mask.sum()


        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(segment_context,
                                                   segment_mask,
                                                   output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]# [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(pos_segment_emb,
                                                       pos_segment_mask,
                                                       output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(neg_segment_emb,
                                                       neg_segment_mask,
                                                       output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :] # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(self.criterion(sp_distance,
                                           torch.ones_like(sp_distance, dtype=torch.float32))) #/ torch.ones_like(sp_distance, dtype=torch.float32).sum()

        if self.args.state_iv:
            pad_mask = ((pos_items != 0)).unsqueeze(1).repeat(1, pos_items.size(1), 1).long()
            attn_mask = get_attn_mask_pretrain(pad_mask.size(0), self.args.max_len, self.args.device).long()
            mask = pad_mask #* attn_mask
            esl, esu, etl, etu = self.emb_sl_01(mask), self.emb_su_01(mask), self.emb_tl_01(mask), self.emb_tu_01(mask)
            sl, su, tl, tu = self.args.ex
            vsl, vsu, vtl, vtu = (s_mat.to_dense() - sl).unsqueeze(-1).expand(-1, -1, -1, self.features), \
                                 (su - s_mat.to_dense()).unsqueeze(-1).expand(-1, -1, -1, self.features), \
                                 (t_mat.to_dense() - tl).unsqueeze(-1).expand(-1, -1, -1, self.features), \
                                 (tu - t_mat.to_dense()).unsqueeze(-1).expand(-1, -1, -1, self.features)#.to_dense()

            space_interval = (esl * vsu + esu * vsl) / (su - sl)
            time_interval = (etl * vtu + etu * vtl) / (tu - tl)
            v = torch.sum(space_interval + time_interval, -1)

            """build instrumental variable and confounders """
            self.rep_z, self.w_z, self.b_z_ = self.rep_net(v, v.size(-1), self.features, self.rep_net_layer)
            self.rep_c, self.w_c, self.b_c = self.rep_net(v, v.size(-1), self.features, self.rep_net_layer)
            sequence_mask = (masked_item_sequence == 0).float() * -1e8
            sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)
            self.rep_z_pos = self.add_position(self.rep_z)
            self.rep_z = self.rep_z + self.rep_z_pos
            self.rep_z = self.LayerNorm(self.rep_z)
            self.rep_z = self.dropout(self.rep_z)
            self.rep_c_pos = self.add_position(self.rep_c)
            self.rep_c = self.rep_c + self.rep_c_pos
            self.rep_c = self.LayerNorm(self.rep_c)
            self.rep_c = self.dropout(self.rep_c)

            self.rep_z = self.item_encoder(self.rep_z,
                                            sequence_mask)[-1]
            self.rep_c = self.item_encoder(self.rep_c,
                                            sequence_mask)[-1]                                
            self.rep_zc = torch.cat([self.rep_z, self.rep_c], 2)
            '''build treatment prediction network'''
            self.x_pre, self.w_x, self.b_x = self.rep_net(self.rep_zc, self.features*2, self.features, self.rep_net_layer)

        #plt
            # #print(y[0][:9], self.rep_z[0][:9])
            # data = torch.cat([y[0][:19], self.rep_z[0][19].unsqueeze(0)], 0)
            # data = torch.cat([data, self.rep_c[0][19].unsqueeze(0)], 0)
            # data = torch.cat([data, x[0][19*101+1:19*101+21]], 0)
            # data = torch.cat([data, x[0][19*101].unsqueeze(0)],0)
            # label = [0 for i in range(20)] + [1]
            # label = label + [2]
            # label = label + [3 for i in range(19)]
            # label = np.array(label + [4])
            # #print("computing t-SNE embedding")
            # tsne = TSNE(n_components=2, init='pca', random_state=0)
            # result = tsne.fit_transform(data.cpu().detach().numpy())
            # fig = plot_embedding(result, label, 'weeplaces')
            # #plt.show(fig)
            # plt.savefig('weeplaces.png', bbox_inches='tight')

            # MIP
            self.rep_v, self.w_v, self.b_v = self.rep_net(v, v.size(-1), self.features, self.rep_net_layer)
            self.rep_v = self.item_encoder(self.rep_v,
                                            sequence_mask)[-1]
            pos_item_embs = self.rep_v
            neg_item_embs = self.item_embeddings(neg_items)

            pos_score = self.masked_item_prediction(sequence_output, pos_item_embs) #[B L H]
            neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
            mip_distance = torch.sigmoid(pos_score - neg_score)
            mip_loss_treatment = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)) #[B, L]
            mip_mask = (masked_item_sequence == self.mask_id).float()  #[B, L]
            mip_loss_treatment = torch.sum(mip_loss_treatment * mip_mask.flatten()) #/ mip_mask.sum()

            #mip_loss_treatment = torch.tensor(0.0)


            """Build embedding network."""
            self.x_emb, self.w_emb, self.b_emb = self.rep_net(self.x_pre, self.hidden_size, self.hidden_size, self.rep_net_layer)
            self.rep_cx = torch.cat([self.rep_c, self.x_emb], 2)

            """Build outcome prediction network."""
            self.rep_cx = self.self_attn_iv(self.rep_cx, None, attn_mask, pad_mask)
            self.y_pre, self.w_y, self.b_y = self.rep_net(self.rep_cx, self.features*2, self.features, self.rep_net_layer)


            self.rep_z = self.z_linear(self.rep_z)
            self.rep_c = self.c_linear(self.rep_c)

            self.rep_z, self.rep_c = self.rep_z.unsqueeze(2).repeat(1, 1, (self.args.treatment_neg_num+1), 1).view(x.size()), self.rep_c.unsqueeze(2).repeat(1, 1, (self.args.treatment_neg_num+1), 1).view(x.size())
            self.y_pre, self.x_pre = self.y_pre.unsqueeze(2).repeat(1, 1, (self.args.treatment_neg_num+1), 1).view(x.size()), self.x_pre.unsqueeze(2).repeat(1, 1, (self.args.treatment_neg_num+1), 1).view(x.size())
            y = y.unsqueeze(2).repeat(1,1, (self.args.treatment_neg_num+1), 1).view(x.size())

            """Maximize MI between z and x."""
            self.lld_zx, self.bound_zx, self.mu_zx, self.logvar_zx, self.ws_zx = self.mi_net(
                self.rep_z,
                x,
                None,
                'max',
                label_mask)

            """Minimize MI between z and y given x."""
            self.lld_zy, self.bound_zy, self.mu_zy, self.logvar_zy, self.ws_zy = self.mi_net(
                self.rep_z,
                y,
                x,
                'min',
                label_mask,
                'zy')

            """Maximize MI between c and x."""
            self.lld_cx, self.bound_cx, self.mu_cx, self.logvar_cx, self.ws_cx = self.mi_net(
                self.rep_c,
                x,
                None,
                'max',
                label_mask)

            """Maximize MI between c and y."""
            self.lld_cy, self.bound_cy, self.mu_cy, self.logvar_cy, self.ws_cy = self.mi_net(
                self.rep_c,
                y,
                None,
                'max',
                label_mask)

            """Minimize MI between z and c."""
            self.lld_zc, self.bound_zc, self.mu_zc, self.logvar_zc, self.ws_zc = self.mi_net(
                self.rep_z,
                self.rep_c,
                None,
                'min',
                label_mask)

            loss_cx2y, loss_zc2x, loss_reg, loss_lld, loss_bound, loss_2stage, loss_orth = self.calculate_loss(x, y, label_mask)
        else:
            loss_cx2y, loss_zc2x, loss_reg, loss_lld, loss_bound, loss_2stage, loss_orth = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0), torch.tensor(0.0)
            self.lld_zx, self.lld_zy, self.lld_cx, self.lld_cy, self.lld_zc = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)
            self.bound_zx, self.bound_zy,  self.bound_cx, self.bound_cy, self.bound_zc = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)
            mip_loss_treatment = torch.tensor(0.0)

        return mip_loss, sp_loss, mip_loss_treatment, loss_cx2y, loss_zc2x, loss_reg, loss_lld, loss_bound, loss_2stage, loss_orth, self.lld_zx, self.lld_zy,  self.lld_cx, self.lld_cy, self.lld_zc, self.bound_zx, self.bound_zy,  self.bound_cx, self.bound_cy, self.bound_zc
        # return mip_loss, sp_loss, mip_loss_treatment, self.lld_zx, self.lld_zy, self.lld_cx, self.lld_cy, self.lld_zc


    # Fine tune
    # same as SASRec
    def finetune(self, src_loc, src_user, user_truncated_segment_index, src_quadkey, src_time, t_mat, g_mat, mat2t, pad_mask, attn_mask,
                trg_loc, trg_quadkey, key_pad_mask, mem_mask, ds, model_name, epoch_idx, state, is_lstm=True):
        #---------------------------------------------input-----------------------------------------------------------------
        #print(src_quadkey.max())
        self.args.model_name = model_name
        # (b, n, d)
        src_loc_emb = self.item_embeddings(src_loc)
        # (b, n * (1 + k), d)
        trg_loc_emb = self.item_embeddings(trg_loc)
        src_user_emb = self.emb_user(src_user)
        #src_time = self.time_embeddings()
        batch_size, seq_len = src_loc.shape[0], src_loc.shape[1]

        #---------------------------------------------lstm-----------------------------------------------------------------
        if self.args.model_name == 'lstm':
            h = self.h0_strategy.on_init(batch_size).cuda()#, self.device)
            for j in range(batch_size):
                if is_lstm:
                    hc = self.h0_strategy.on_reset(batch_size)
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = self.h0_strategy.on_reset(batch_size)
            src, h = self.model(src_loc_emb, h)
            #save the hidden state of sequence for rmsn
            if epoch_idx == (self.args.epoch_num-1) and state == 'train_rmsn':
                for i in range(batch_size):
                    if src_user[i][0].cpu().item() not in self.weight_lstm:
                        self.weight_lstm[src_user[i][0].cpu().item()] = {}
                    tmp = torch.ones([seq_len, self.features])
                    tmp[:] = src[i]
                    self.weight_lstm[src_user[i][0].cpu().item()][user_truncated_segment_index[i].cpu().item()] = tmp


            #in order to match target(pos and neg(1+k)), repeat src
            if self.training:
                # (b, n * (1 + k), 2 * d)
                src_repeat = src.repeat(1, trg_loc_emb.size(1)//src.size(1), 1)
            else:
                # (b, d)
                src_single = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
                # (b, 1 + k, d)
                src_repeat = src_single.unsqueeze(1).repeat(1, trg_loc_emb.size(1), 1)
            output = torch.sum(src_repeat * trg_loc_emb, dim=-1)
        #---------------------------------------------sasrec------------------------------------------------------------------
        elif self.args.model_name == 'sasrec':
            src = self.add_position_embedding(src_loc)
            src = self.self_attn(src, None, attn_mask, pad_mask)
            #in order to match target(pos and neg(1+k)), repeat src
            if self.training:
                # (b, n * (1 + k), 2 * d)
                src_repeat = src.repeat(1, trg_loc_emb.size(1)//src.size(1), 1)
            else:
                # (b, d)
                src_single = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
                # (b, 1 + k, d)
                src_repeat = src_single.unsqueeze(1).repeat(1, trg_loc_emb.size(1), 1)
            output = torch.sum(src_repeat * trg_loc_emb, dim=-1)

        #---------------------------------------------geosan--------------------------------------------------------------
        elif self.args.model_name == 'geosan':
            # (b, n, l, d)
            src_quadkey_emb = self.emb_quadkey(src_quadkey)
            # (b, n, d)
            src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
            # (b, n * (1 + k), d)
            trg_quadkey_emb = self.emb_quadkey(trg_quadkey)
            # (b, n * (1 + k), d)
            trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)
            # (b, n, 2 * d)
            src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)
            # (b, n * (1 + k), 2 * d)
            trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)
            
            src = self.inr_awa_attn_block(src, None, attn_mask, pad_mask)

            if self.training:
                # (b, n * (1 + k), 2 * d)
                src = src.repeat(1, trg.size(1)//src.size(1), 1)
                # (b, n * (1 + k), 2 * d)
                src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
            else:
                # (b, 2 * d)
                src = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
                # (b, 1 + k, 2 * d)
                src = src.unsqueeze(1).repeat(1, trg.size(1), 1)
                src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
            # (b, 1 + k)
            output = torch.sum(src * trg, dim=-1)
        #---------------------------------------------stisan-----------------------------------------------------------------
        elif self.args.model_name == 'stisan':
            # (b, n, l, d)
            src_quadkey_emb = self.emb_quadkey(src_quadkey)
            # (b, n, d)
            src_quadkey_emb = self.geo_encoder(src_quadkey_emb)
            # (b, n * (1 + k), d)
            trg_quadkey_emb = self.emb_quadkey(trg_quadkey)
            # (b, n * (1 + k), d)
            trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)
            # (b, n, 2 * d)
            src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)
            # (b, n * (1 + k), 2 * d)
            trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)
            # (b, n, 2 * d)
            src = self.tape(src, src_time, ds, self.device)
            # (b, n, n)
            for i in range(src.size(0)):
                mask = torch.gt(t_mat[i], self.k_t)
                t_mat[i] = t_mat[i].masked_fill(mask == True, self.k_t)
                t_mat[i] = t_mat[i].max() - t_mat[i]
                mask = torch.gt(t_mat[i], self.k_t)
                g_mat[i] = g_mat[i].masked_fill(mask == True, self.k_g)
                g_mat[i] = g_mat[i].max() - g_mat[i]
            # (b, n, n)
            r_mat = t_mat + g_mat
            # (b, n, 2 * d)
            src = self.inr_awa_attn_block(src, r_mat, attn_mask, pad_mask)

            if self.training:
                # (b, n * (1 + k), 2 * d)
                src = src.repeat(1, trg.size(1)//src.size(1), 1)
                # (b, n * (1 + k), 2 * d)
                src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
            else:
                # (b, 2 * d)
                src = src[torch.arange(len(ds)), torch.tensor(ds) - 1, :]
                # (b, 1 + k, 2 * d)
                src = src.unsqueeze(1).repeat(1, trg.size(1), 1)
                src = self.trg_awa_attn_decoder(src, trg, key_pad_mask, mem_mask)
            # (b, 1 + k)
            output = torch.sum(src * trg, dim=-1)
        return output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    '''GetNext'''
    def init_graph(self):
        print("start init_graph moudle")
        data_path = r'dataset/gowalla' 
        if os.path.exists(data_path):
            pass
        else:
            os.makedirs(data_path)
        # Build POI graph (built from train_df)
        print('Loading POI graph...')
        self.raw_A = build_global_POI_checkin_graph('./gowalla.txt', 'gowalla', self.args)
        self.X = np.random.randn(self.args.n_loc-1, self.features)#, dtype=np.float32)

        # Normalization
        print('Laplician matrix...')
        self.A = calculate_laplacian_matrix(self.raw_A, mat_type='hat_rw_normd_lap_mat')
        print("GCN...")
        self.X = torch.from_numpy(self.X).float().cuda()
        self.A = torch.from_numpy(self.A).float().cuda()
        self.gcn_nfeat = self.X.shape[1]
        self.pad_weight = torch.rand((self.features,)).unsqueeze(0).cuda()
        self.poi_embed_gcn = GCN(ninput=self.gcn_nfeat,
                          nhid=[32,64],
                          noutput=self.features,
                          dropout=0.3).cuda()
        print(self.A[0], self.X[0])
        self.loc_emb = self.poi_embed_gcn(self.X, self.A).cuda()
        self.loc_emb = torch.cat([self.pad_weight, self.loc_emb], dim=0)
        self.loc_emb = torch.cat([self.loc_emb, self.pad_weight], dim=0)
        self.item_embeddings.weight = nn.Parameter(self.loc_emb)#torch.nn.Embedding.from_pretrained(self.loc_emb)
        print("init graph embedding successfully")


    def adjust_pred_prob_by_graph(self, src_seq, y_pred_poi):
        #torch.set_printoptions(threshold=np.inf)
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = self.node_attn_model(self.X, self.A)
        pad = nn.ZeroPad2d(padding=(1,0,1,0)) #(left, right, top, bottom)
        attn_map = pad(attn_map)
        for i in range(y_pred_poi.size(0)):
            traj_i_input = src_seq[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j]-1, :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted