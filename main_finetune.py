from LBSNData import LBSNData, PretrainData
from near_location_query import Loc_Query_System
from utils import *
from loss_fn import *
from neg_sampler import *
from model import STiSAN
from models import S3RecModel
from trainer import *
from trainers import PretrainTrainer
from torch.utils.data import DataLoader, RandomSampler
from trainer_rmsn import *
import os
import torch
import argparse
from utils import *
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # Setting paths
    path_prefix = './'
    parser = argparse.ArgumentParser()
    # data management
    parser.add_argument('--data_name', default='brightkite', type=str, help='the dataset under ./data/<dataset.txt> to load')
    parser.add_argument('--gpu', default='0', type=str, help='the gpu to use')
    parser.add_argument('--train_bsz', default=500, type=int, help='amount of users to process in one pass (training batching)')#56
    parser.add_argument('--eval_bsz', default=1000, type=int, help='amount of users to process in one pass (validation batching)')
    parser.add_argument('--test_bsz', default=1000, type=int, help='amount of users to process in one pass (test batching)')

    parser.add_argument('--max_len', default=50, type=int, help='the max length of the sequence')
    parser.add_argument('--model_name', default='stisan', type=str, help='the name of the dataset')
    parser.add_argument('--state_rmsn', default=False, type=bool, help='whether to run rmsn' )
    parser.add_argument('--epoch_num',  default=100, type=int, help='the num of epoch')
    parser.add_argument('--cluster_nodes',  default=3, type=int, help='the num of clustered regions')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=500, help="number of pretrain batch_size")
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--load_model', default=False, type=bool, help='load pretrain model')
    parser.add_argument('--state_iv', default=False, type=bool, help='load pretrain iv model ')

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)



    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    #for caser
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')

    parser.add_argument('--datapara', default=False, type=bool, help='DataParallel' )
    args = parser.parse_args()
    set_seed(args.seed)
    # data_name = 'weeplaces'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Dataset: ", args.data_name)
    print("model_name: ", args.model_name)
    print("train_bsz: ", args.train_bsz)
    #torch.cuda.set_device('cpu')
    args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #args.device = torch.device('cpu')
    print("GPU number", torch.cuda.device_count(), args.device)
    print("paramters", args)
    if args.data_name == 'nyc':
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
    args_str = f'{args.model_name}-{args.data_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    # Data Process details
    min_loc_freq = 10
    min_user_freq = 20
    map_level = 15
    n_nearest = 2000
    #max_len = 100
    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        dataset = LBSNData(args.data_name, raw_data_path, min_loc_freq, min_user_freq, map_level, args)
        serialize(dataset, clean_data_path)
    args.loc2idx = dataset.loc2idx
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
    args.n_user = dataset.n_user
    args.n_loc = dataset.n_loc
    
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


    # print("Pretrain Data preparing..")
    # pretrain_dataset = PretrainData(args.data_name, raw_data_path, min_loc_freq, min_user_freq, map_level, args)

    print("Data Partition...")
    train_data, eval_data, test_data = dataset.data_partition(args, args.max_len, args.st_matrix)
    args.ex = dataset.su, dataset.sl, dataset.tu, dataset.tl

    if args.data_name == 'gowalla':
        temperature = 1.0
        #args.epoch_num = 35
        k_t_storage = 10.0
        k_g_storage = 15.0
    else:
        temperature = 100.0
        #args.epoch_num = 20
        k_t_storage = 5.0
        k_g_storage = 5.0
    if args.state_rmsn == True:
        args.epoch_num = 2
    #args.epoch_num = 50
    # Setting training details
    num_workers = 24
    n_nearest_locs = 2000
    #num_epoch = 35
    train_bsz = args.train_bsz
    eval_bsz = args.eval_bsz
    train_num_neg = 15
    eval_num_neg = 100
    test_num_neg = 100
    args.user_visited_locs = get_visited_locs(dataset)
    loss_fn = WeightedBCELoss(temperature)
    train_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'training', True)
    eval_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'evaluating', True)
    test_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'testing', True)

    # # Building Spatial-Temporal Relation Matrix
    # if os.path.exists(args.candidate_mat_path):
    #     with open(args.candidate_mat_path, 'rb') as file:
    #         args.mat2s = _pickle.load(file)
    # else:
    #     #mat2s = torch.zeros((100,100))#rs_mat2s(self, self.n_loc-1)
    #     mat2s = rs_mat2s(self, self.n_loc-1)
    #     joblib.dump(mat2s, args.candidate_mat_path, protocol=4)
    #     #torch.save(mat2s, args.candidate_mat_path)
    #     with open(args.candidate_mat_path, 'rb') as file:
    #         args.mat2s = _pickle.load(file)
    #Model details
    model = S3RecModel((dataset.n_loc+1),
                       dataset.n_user,
                       dataset.n_quadkey,
                       features=args.hidden_size,
                       exp_factor=4,
                       k_t=k_t_storage,
                       k_g=k_g_storage,
                       depth=4,
                       dropout=0.7,
                       device=args.device,
                       args=args)

    if args.datapara:
        model = torch.nn.DataParallel(model)
        # if isinstance(model, torch.nn.DataParallel):
        #     model = model.module
    
    skip = ["interest_tgcn.tgcn_cell.graph_conv1.emb_sl.weight", "interest_tgcn.tgcn_cell.graph_conv1.emb_su.weight", "interest_tgcn.tgcn_cell.graph_conv1.emb_tl.weight", "interest_tgcn.tgcn_cell.graph_conv1.emb_tu.weight", "interest_tgcn.tgcn_cell.graph_conv2.emb_sl.weight", "interest_tgcn.tgcn_cell.graph_conv2.emb_su.weight", "interest_tgcn.tgcn_cell.graph_conv2.emb_tl.weight", "interest_tgcn.tgcn_cell.graph_conv2.emb_tu.weight", "spatial_tgcn.tgcn_cell.graph_conv1.emb_sl.weight", "spatial_tgcn.tgcn_cell.graph_conv1.emb_su.weight", "spatial_tgcn.tgcn_cell.graph_conv1.emb_tl.weight", "spatial_tgcn.tgcn_cell.graph_conv1.emb_tu.weight", "spatial_tgcn.tgcn_cell.graph_conv2.emb_sl.weight", "spatial_tgcn.tgcn_cell.graph_conv2.emb_su.weight", "spatial_tgcn.tgcn_cell.graph_conv2.emb_tl.weight", "spatial_tgcn.tgcn_cell.graph_conv2.emb_tu.weight", "temporal_tgcn.tgcn_cell.graph_conv1.emb_sl.weight", "temporal_tgcn.tgcn_cell.graph_conv1.emb_su.weight", "temporal_tgcn.tgcn_cell.graph_conv1.emb_tl.weight", "temporal_tgcn.tgcn_cell.graph_conv1.emb_tu.weight", "temporal_tgcn.tgcn_cell.graph_conv2.emb_sl.weight", "temporal_tgcn.tgcn_cell.graph_conv2.emb_su.weight", "temporal_tgcn.tgcn_cell.graph_conv2.emb_tl.weight", "temporal_tgcn.tgcn_cell.graph_conv2.emb_tu.weight"]
    if args.load_model:
        if args.state_iv:
            ckp = f'{args.data_name}-pretrain_iv-epochs-{70}.pt'  #{args.output_dir}-
            checkpoint_path = os.path.join(args.output_dir, ckp)
            model.load_state_dict({k:v for k,v in torch.load(checkpoint_path).items() if k not in skip})
            #model.load_state_dict(torch.load(checkpoint_path))
        else:
            ckp = f'{args.data_name}-pretrain-epochs-{100}.pt'
            checkpoint_path = os.path.join(args.output_dir, ckp)
            model.load_state_dict({k:v for k,v in torch.load(checkpoint_path).items()})
            #model.load_state_dict(torch.load(checkpoint_path))
            print("load success!")
    model.cuda()#.to(args.device)

    # Starting training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train(model,
          args.max_len,
          train_data,
          train_sampler,
          args.train_bsz,
          train_num_neg,
          args.epoch_num,
          quadkey_processor,
          loc2quadkey,
          eval_data,
          eval_sampler,
          args.eval_bsz,
          eval_num_neg,
          test_data,
          test_sampler,
          args.test_bsz,
          test_num_neg,
          optimizer,
          loss_fn,
          args.device,
          num_workers,
          log_path,
          result_path,
          args)