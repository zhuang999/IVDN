import time

import torch
from einops import rearrange
from tqdm import tqdm
from evaluator import evaluate
from batch_generater import cf_train_quadkey
from utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(model, max_len, train_data, train_sampler, train_bsz, train_num_neg, num_epoch, quadkey_processor, loc2quadkey,
          eval_data, eval_sampler, eval_bsz, eval_num_neg, test_data, test_sampler, test_bsz, test_num_neg, optimizer, loss_fn, device, num_workers, log_path, result_path, args):
    pre_avg_loss = 1e6
    writer = SummaryWriter(log_dir='logs', flush_secs=30)
    for epoch_idx in range(num_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_data,
                                 sampler=LadderSampler(train_data, train_bsz),
                                 num_workers=num_workers, batch_size=train_bsz,
                                 collate_fn=lambda e: cf_train_quadkey(
                                     e,
                                     train_data,
                                     max_len,
                                     train_sampler,
                                     quadkey_processor,
                                     loc2quadkey,
                                     train_num_neg,
                                     args))
        print("=====epoch {:>2d}=====".format(epoch_idx))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True, ncols=70)
        model.train()
        for batch_idx, (src_locs_, src_users_, user_truncated_segment_index_, src_quadkeys_, src_times_, t_mat_, g_mat_, mat2t_, trg_locs_, trg_quadkeys_, data_size) in batch_iterator:
            src_loc = src_locs_.cuda()#.to(device)
            src_user = src_users_.cuda()#to(device)
            user_truncated_segment_index = user_truncated_segment_index_.cuda()#to(device)
            src_quadkey = src_quadkeys_.cuda()#to(device)
            src_time = src_times_.cuda()#to(device)
            t_mat = t_mat_.cuda()#to(device)
            g_mat = g_mat_.cuda()#to(device)
            mat2t = None
            data_size = torch.tensor(data_size).cuda()
            trg_loc = trg_locs_.cuda()#to(device)
            trg_quadkey = trg_quadkeys_.cuda()#to(device)
            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(data_size, max_len, device)
            mem_mask = get_mem_mask(data_size, max_len, train_num_neg, device)
            key_pad_mask = get_key_pad_mask(data_size, max_len, train_num_neg, device)
            optimizer.zero_grad()
            output = model.finetune(src_loc, src_user, user_truncated_segment_index, src_quadkey, src_time, t_mat, g_mat, mat2t, pad_mask, attn_mask,
                                    trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size, args.model_name, epoch_idx, 'train')
            output = rearrange(rearrange(output, 'b (k n) -> b k n', k=1 + train_num_neg), 'b k n -> b n k')
            pos_scores, neg_scores = output.split([1, train_num_neg], -1)
            loss = loss_fn(pos_scores, neg_scores)
            keep = [torch.ones(e, dtype=torch.float32).cuda() for e in data_size]#to(device)
            keep = fix_length(keep, 1, max_len, dtype="exclude padding term")

            loss = torch.sum(loss * keep) / torch.sum(torch.tensor(data_size).cuda())#.to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")
            #writer.add_scalar("Loss/train", loss, global_step=epoch_idx)
        epoch_time = time.time() - start_time
        cur_avg_loss = running_loss / processed_batch
        delta_decay = pre_avg_loss - cur_avg_loss
        f = open(log_path, 'a+')
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("avg. loss decay: {:.4f}".format(delta_decay))
        print("epoch={:d}, loss={:.4f}, delta={:.4f}".format(epoch_idx + 1, cur_avg_loss, delta_decay), file=f)
        writer.add_scalar("Loss_batch/train", cur_avg_loss, global_step=epoch_idx)
        f.close()
        pre_avg_loss = cur_avg_loss
        if (epoch_idx + 1) % 5 == 0:
            print("=====evaluation under sampled metric (100 nearest un-visited locations)=====")
            hr_1, ndcg_1 = evaluate(model, max_len, test_data, test_sampler, eval_bsz, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers, args)
            print("Hit@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr_1[0], hr_1[4], ndcg_1[4], hr_1[9], ndcg_1[9]))
            writer.add_scalars(args.data_name+"/metric", {'Hit@1':hr_1[0],
                                                          'Hit@5':hr_1[4],
                                                          'NDCG@5':ndcg_1[4],
                                                          'Hit@10':hr_1[9],
                                                          'NDCG@10':ndcg_1[9]}, global_step=epoch_idx)
            
            # print("=====test under sampled metric (100 nearest un-visited locations)=====")
            # hr_1, ndcg_1 = evaluate(model, max_len, test_data, test_sampler, eval_bsz, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers, args)
            # print("Hit@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr_1[0], hr_1[4], ndcg_1[4], hr_1[9], ndcg_1[9]))
            # writer.add_scalars(args.data_name+"/metric", {'Hit@1':hr_1[0],
            #                                               'Hit@5':hr_1[4],
            #                                               'NDCG@5':ndcg_1[4],
            #                                               'Hit@10':hr_1[9],
            #                                               'NDCG@10':ndcg_1[9]}, global_step=epoch_idx)

    print("training completed!")
    print("")
    print("=====test under sampled metric (100 nearest un-visited locations)=====")
    hr, ndcg = evaluate(model, max_len, test_data, test_sampler, test_bsz, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers, args)
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr[4], ndcg[4], hr[9], ndcg[9]))


    f = open(result_path, 'a+')
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr[4], ndcg[4], hr[9], ndcg[9]), file=f)
    writer.add_scalars(args.data_name+"/metric", {'Hit@1':hr[0],
                                                  'Hit@5':hr[4],
                                                  'NDCG@5':ndcg[4],
                                                  'Hit@10':hr[9],
                                                  'NDCG@10':ndcg[9]}, global_step=epoch_idx+1)
    writer.close()
    f.close()

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))