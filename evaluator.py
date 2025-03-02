from batch_generater import cf_eval_quadkey
from utils import *
from torch.utils.data import DataLoader
from collections import Counter


def evaluate(model, max_len, eval_data, eval_sampler, eval_batch_size, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers, args):
    model.eval()
    loader = DataLoader(eval_data, batch_size=eval_batch_size, num_workers=num_workers,
                        collate_fn=lambda e: cf_eval_quadkey(e, eval_data, max_len, eval_sampler, quadkey_processor, loc2quadkey, eval_num_neg, args))
    cnt = Counter()
    array = np.zeros(1 + eval_num_neg)
    with torch.no_grad():
        for _, (src_locs_, src_users_, src_quadkeys_, src_times_, t_mat_, g_mat_, mat2t_, trg_locs_, trg_quadkeys_, data_size) in enumerate(loader):
            src_loc = src_locs_.cuda()#to(device)
            src_user = src_users_.cuda()#to(device)
            src_quadkey = src_quadkeys_.cuda()#to(device)
            src_time = src_times_.cuda()#to(device)
            t_mat = t_mat_.cuda()#to(device)
            g_mat = g_mat_.cuda()#to(device)
            mat2t = mat2t_.cuda()#to(device)
            trg_loc = trg_locs_.cuda()#to(device)
            trg_quadkey = trg_quadkeys_.cuda()#to(device)
            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(data_size, max_len, device)
            mem_mask = None
            key_pad_mask = None
            data_size = torch.tensor(data_size).cuda()
            output = model.finetune(src_loc, src_user, _, src_quadkey, src_time, t_mat, g_mat, mat2t, pad_mask, attn_mask,
                                    trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size, args.model_name, _, 'eval', is_lstm=True)
            idx = output.sort(descending=True, dim=1)[1]
            order = idx.topk(k=1, dim=1, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    Hit_Rate = array.cumsum()
    NDCG = 1 / np.log2(np.arange(0, eval_num_neg + 1) + 2)
    NDCG = NDCG * array
    NDCG = NDCG.cumsum() / Hit_Rate.max()
    Hit_Rate = Hit_Rate / Hit_Rate.max()

    return Hit_Rate, NDCG