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
from config import get_args
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
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Setting paths
    path_prefix = './'
    args = get_args()
    set_seed(args.seed)
    # data_name = 'weeplaces'
    print("Dataset: ", args.data_name)
    print("model_name: pretrain")
    print("pre_batch_size: ", args.pre_batch_size)
    #torch.cuda.set_device('cuda')
    #args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    #args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = None
    print("GPU number", torch.cuda.device_count(), args.device)
    print(args)
    if args.data_name == 'nyc':
        raw_data_path = path_prefix + args.data_name + '.csv'
    else:
        raw_data_path = path_prefix + args.data_name + '.txt'
    clean_data_path = path_prefix + args.data_name + '.data'
    args.loc_query_path = path_prefix + args.data_name + '_loc_query.pkl'
    matrix_path = path_prefix + args.data_name + '_st_matrix.data'
    matrix_path_onlyst = path_prefix + args.data_name + '_onlyst_matrix.data'
    matrix_path_decayst = path_prefix + args.data_name + '_decayst_matrix.data'
    log_path = path_prefix + args.data_name + '.txt'
    result_path = path_prefix + args.data_name + '.txt'
    args.candidate_mat_path = path_prefix + args.data_name + '_candidate_st_matrix.data'
    # save model args
    args_str = f'{args.model_name}-{args.data_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    args.coefs = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
                  'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
                  'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
                  'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}
    # Data Process details
    min_loc_freq = 10
    min_user_freq = 20
    map_level = 15
    n_nearest = 2000
    #max_len = 100

    torch.cuda.empty_cache()
    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir)

    # if args.model_name != "pretrain":
    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        dataset = LBSNData(args.data_name, raw_data_path, min_loc_freq, min_user_freq, map_level, args)
        serialize(dataset, clean_data_path)

    # print("Data Partition...")
    # train_data, eval_data, test_data = dataset.data_partition(args, args.max_len, args.st_matrix)
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))
    args.n_user = dataset.n_user
    args.n_loc = dataset.n_loc

    args.loc2idx = dataset.loc2idx
    # Searching nearest POIs
    args.quadkey_processor = dataset.QUADKEY
    loc2quadkey = dataset.loc2quadkey
    args.loc_query_sys = Loc_Query_System()
    if os.path.exists(args.loc_query_path):
        args.loc_query_sys.load(args.loc_query_path)
    else:
        args.loc_query_sys.build_tree(dataset)
        args.loc_query_sys.prefetch_n_nearest_locs(n_nearest)
        args.loc_query_sys.save(args.loc_query_path)
        args.loc_query_sys.load(args.loc_query_path)


    print("Building Spatial-Temporal Relation Matrix")
    # Building Spatial-Temporal Relation Matrix
    if os.path.exists(matrix_path):
        args.st_matrix = unserialize(matrix_path)
    else:
        dataset.spatial_temporal_matrix_building(matrix_path)
        args.st_matrix = unserialize(matrix_path)


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
    args.user_visited_locs = get_visited_locs(dataset)
    print("Pretrain Data preparing..")
    pretrain_dataset = PretrainData(args.data_name, raw_data_path, min_loc_freq, min_user_freq, map_level, args)


    # print("Data Partition...")
    # train_data, eval_data, test_data = dataset.data_partition(args, args.max_len, args.st_matrix)

    args.ex = pretrain_dataset.tu, pretrain_dataset.tl, pretrain_dataset.su, pretrain_dataset.sl
    if args.data_name == 'gowalla':
        temperature = 1.0
        args.epoch_num = 35
        k_t_storage = 10.0
        k_g_storage = 15.0
    else:
        temperature = 100.0
        args.epoch_num = 20
        k_t_storage = 5.0
        k_g_storage = 5.0
    if args.state_rmsn == True:
        args.epoch_num = 2
    args.epoch_num = 35
    # Setting training details
    num_workers = 24
    n_nearest_locs = 2000
    #num_epoch = 35
    train_bsz = args.train_bsz
    eval_bsz = args.eval_bsz
    train_num_neg = 15
    eval_num_neg = 100
    test_num_neg = 100

    loss_fn = WeightedBCELoss(temperature)
    train_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'training', True)
    eval_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'evaluating', True)
    test_sampler = KNNSampler(args.loc_query_sys, n_nearest_locs, args.user_visited_locs, 'testing', True)


    print("cx=max")
    args.cx = 'max'
    #s
    model_pretrain = STDE(dataset.n_loc+1,
                                dataset.n_user,
                                dataset.n_quadkey,
                                features=args.features,
                                exp_factor=4,
                                k_t=5.0,
                                k_g=5.0,
                                depth=4,
                                dropout=0.7, 
                                device=args.device,
                                args=args)
    # skip = ["emb_quadkey.lookup_table"]
    # ckp = f'{args.data_name}-pretrain_iv-epochs-{100}.pt'  #{args.output_dir}-
    # checkpoint_path = os.path.join('./gowalla_100/', ckp)
    # model_pretrain.load_state_dict(({k:v for k,v in torch.load(checkpoint_path).items() if k not in skip}), strict=False)
    # print("load model necessary")
    if args.datapara:
        model_pretrain = torch.nn.DataParallel(model_pretrain)
        # if isinstance(model_pretrain, torch.nn.DataParallel):
        #     model_pretrain = model_pretrain.module
    model_pretrain.cuda()#.to(args.device)

    trainer_pretrain = PretrainTrainer(model_pretrain, None, None, None, args)
    #Starting pretraining
    for epoch in range(args.pre_epochs):
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)
        trainer_pretrain.pretrain(epoch, pretrain_dataloader)

        if (epoch+1) % 10 == 0:
            if args.state_iv:
                ckp = f'{args.data_name}-pretrain_iv-epochs-{epoch+1}.pt'
                checkpoint_path = os.path.join(args.output_dir, ckp)
                torch.save(model_pretrain.state_dict(), checkpoint_path)

            else:
                ckp = f'{args.data_name}-pretrain-epochs-{epoch+1}.pt'
                checkpoint_path = os.path.join(args.output_dir, ckp)
                torch.save(model_pretrain.state_dict(), checkpoint_path)
    