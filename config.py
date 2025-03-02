import argparse
import json5
from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
        # data management
    parser.add_argument('--data_name', default='brightkite', type=str, help='the dataset under ./data/<dataset.txt> to load')
    parser.add_argument('--gpu', default='0', type=str, help='the gpu to use')
    parser.add_argument('--train_bsz', default=56, type=int, help='amount of users to process in one pass (training batching)')#56
    parser.add_argument('--eval_bsz', default=196, type=int, help='amount of users to process in one pass (validation batching)')
    parser.add_argument('--test_bsz', default=196, type=int, help='amount of users to process in one pass (test batching)')

    parser.add_argument('--max_len', default=50, type=int, help='the max length of the sequence')
    parser.add_argument('--model_name', default='pretrain', type=str, help='the name of the dataset')
    parser.add_argument('--pre_model_name', default='ctle', type=str, help='the name of the dataset')
    parser.add_argument('--state_rmsn', default=False, type=bool, help='whether to run rmsn')
    parser.add_argument('--state_iv', default=False, type=bool, help='whether to run iv model')
    parser.add_argument('--epoch_num',  default=100, type=int, help='the num of epoch')
    parser.add_argument('--cluster_nodes',  default=3, type=int, help='the num of clustered regions')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=150, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=200, help="number of pretrain batch_size")
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--load_model', default=False, type=bool, help='load pretrain model')
    parser.add_argument("--treatment_neg_num", type=int, default=30, help="number of treatment sampler for expored set")

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--features", type=int, default=64, help="hidden size of model")
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

    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--sp_weight", type=float, default=1.0, help="sp loss weight")
    parser.add_argument("--mip_treatment_weight", type=float, default=1.0, help="mip treatment loss weight")
    
    parser.add_argument("--cx2y_weight", type=float, default=0.002, help="cx2y loss weight")  #0.0003
    parser.add_argument("--zc2x_weight", type=float, default=0.002, help="zc2x loss weight")  #0.0003
    parser.add_argument("--reg_weight", type=float, default=30.0, help="reg loss weight")  #10.0
    parser.add_argument("--lld_weight", type=float, default=0.0005, help="lld loss weight")   #0.0001
    parser.add_argument("--bound_weight", type=float, default=0.1, help="bound loss weight")  #0.1
    parser.add_argument("--stage_weight", type=float, default=0.001, help="stage treatment loss weight") #0.0001
    parser.add_argument("--orth_weight", type=float, default=5.0, help="orth loss weight") #1.0

    parser.add_argument("--zx_weight", type=float, default=0.2, help="zx loss weight")
    parser.add_argument("--zy_weight", type=float, default=0.2, help="zy loss weight")
    parser.add_argument("--cx_weight", type=float, default=0.2, help="cx loss weight")
    parser.add_argument("--cy_weight", type=float, default=0.2, help="cy loss weight")
    parser.add_argument("--zc_weight", type=float, default=0.2, help="zc loss weight")


    parser.add_argument("--sigma", type=float, default=0.1, help="sigma for RBF kernal in mi_net")

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
    
    parser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')
    parser.add_argument(
        '-i', '--id',
        metavar='I',
        default='',
        help='The commit id)')
    parser.add_argument(
        '-t', '--ts',
        metavar='T',
        default='',
        help='The time stamp)')
    parser.add_argument(
        '-d', '--dir',
        metavar='D',
        default='',
        help='The output directory)')
    args = parser.parse_args()
    return args


def get_config_from_json(json_file):
    # parse the configurations from the configs json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json5.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config