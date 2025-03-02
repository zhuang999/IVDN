import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.device = args.device

        self.model = model

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.cuda()#.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.args = args

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'MIP-{self.args.mip_weight}-' \
               f'SP-{self.args.sp_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        mip_loss_avg = 0.0
        sp_loss_avg = 0.0
        mip_treatment_loss_avg = 0.0
        cx2y_loss_avg = 0.0
        zc2x_loss_avg = 0.0
        reg_loss_avg = 0.0
        lld_loss_avg = 0.0
        bound_loss_avg = 0.0
        stage_loss_avg = 0.0
        orth_loss_avg = 0.0
        lld_zx_avg = 0.0
        lld_zy_avg = 0.0
        lld_cx_avg = 0.0
        lld_cy_avg = 0.0
        lld_zc_avg = 0.0

        bound_zx_avg = 0.0
        bound_zy_avg = 0.0
        bound_cx_avg = 0.0
        bound_cy_avg = 0.0
        bound_zc_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.cuda() for t in batch) #to(self.device)
            masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment, target_items, neg_treatment, t_mat, s_mat = batch #neg_treatment, treatment_neg_trg,

            mip_loss, sp_loss, mip_treatment_loss, loss_cx2y, loss_zc2x, loss_reg, loss_lld, loss_bound, loss_2stage, loss_orth, loss_lld_zx, loss_lld_zy,  loss_lld_cx, loss_lld_cy, loss_lld_zc, loss_bound_zx, loss_bound_zy,  loss_bound_cx, loss_bound_cy, loss_bound_zc = self.model.pretrain(masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment, neg_treatment, target_items, t_mat, s_mat)
            joint_loss = self.args.mip_weight * mip_loss + self.args.sp_weight * sp_loss + self.args.mip_treatment_weight * mip_treatment_loss + self.args.cx2y_weight * loss_cx2y + self.args.zc2x_weight *  loss_zc2x +  self.args.reg_weight * loss_reg + self.args.lld_weight *  loss_lld + self.args.bound_weight * loss_bound + self.args.stage_weight * loss_2stage + self.args.orth_weight * loss_orth #self.args.mip_weight * mip_loss +
            if self.args.datapara:
                joint_loss = torch.sum(joint_loss)
            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            mip_loss_avg += torch.sum(mip_loss).item()
            sp_loss_avg += torch.sum(sp_loss).item()
            mip_treatment_loss_avg += torch.sum(mip_treatment_loss).item()
            cx2y_loss_avg += torch.sum(loss_cx2y).item()
            zc2x_loss_avg += torch.sum(loss_zc2x).item()
            reg_loss_avg += torch.sum(loss_reg).item()
            lld_loss_avg += torch.sum(loss_lld).item()
            bound_loss_avg += torch.sum(loss_bound).item()
            stage_loss_avg += torch.sum(loss_2stage).item()
            orth_loss_avg += torch.sum(loss_orth).item()

            lld_zx_avg += torch.sum(loss_lld_zx).item()
            lld_zy_avg += torch.sum(loss_lld_zy).item()
            lld_cx_avg += torch.sum(loss_lld_cx).item()
            lld_cy_avg += torch.sum(loss_lld_cy).item()
            lld_zc_avg += torch.sum(loss_lld_zc).item()

            bound_zx_avg += torch.sum(loss_bound_zx).item()
            bound_zy_avg += torch.sum(loss_bound_zy).item()
            bound_cx_avg += torch.sum(loss_bound_cx).item()
            bound_cy_avg += torch.sum(loss_bound_cy).item()
            bound_zc_avg += torch.sum(loss_bound_zc).item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "mip_loss_avg": '{:.4f}'.format(mip_loss_avg /num),
            "sp_loss_avg": '{:.4f}'.format(sp_loss_avg / num),
            "mip_treatment_loss_avg": '{:.4f}'.format(mip_treatment_loss_avg /num),

            "cx2y_loss_avg": '{:.4f}'.format(cx2y_loss_avg / num),
            "zc2x_loss_avg": '{:.4f}'.format(zc2x_loss_avg /num),
            "reg_loss_avg": '{:.4f}'.format(reg_loss_avg / num),
            "lld_loss_avg": '{:.4f}'.format(lld_loss_avg /num),
            "bound_loss_avg": '{:.4f}'.format(bound_loss_avg / num),
            "stage_loss_avg": '{:.4f}'.format(stage_loss_avg / num),
            "orth_loss_avg": '{:.4f}'.format(orth_loss_avg / num),

            "lld_zx_avg": '{:.4f}'.format(lld_zx_avg / num),
            "lld_zy_avg": '{:.4f}'.format(lld_zy_avg /num),
            "lld_cx_avg": '{:.4f}'.format(lld_cx_avg / num),
            "lld_cy_avg": '{:.4f}'.format(lld_cy_avg /num),
            "lld_zc_avg": '{:.4f}'.format(lld_zc_avg / num),

            "bound_zx_avg": '{:.4f}'.format(bound_zx_avg / num),
            "bound_zy_avg": '{:.4f}'.format(bound_zy_avg /num),
            "bound_cx_avg": '{:.4f}'.format(bound_cx_avg / num),
            "bound_cy_avg": '{:.4f}'.format(bound_cy_avg /num),
            "bound_zc_avg": '{:.4f}'.format(bound_zc_avg / num)

        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.cuda() for t in batch) #to(self.device)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.cuda() for t in batch)#to(self.device)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.cuda() for t in batch)#to(self.device)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
