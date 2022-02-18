import warnings

warnings.filterwarnings("ignore")

import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
from time import time
import os
import joblib
from sklearn import metrics
import numpy as np

import config
from Utils import Logging


def calc_acc(y_true, y_pred):  # y_pred：{0,1} 必须事先通过阈值转变为0,1
    return metrics.accuracy_score(y_true, y_pred)


def calc_auc(y_true, y_pred):  # y_pred：[0,1]之间任何数
    return metrics.roc_auc_score(y_true, y_pred)


class HeteroClassifier(nn.Module):
    def __init__(self, layer_size, pretrain=True):
        super().__init__()
        if pretrain:
            self.wordemb_file = pd.read_csv("../data/word.csv", sep='\t', header=0)
            self.wordemb = torch.FloatTensor([list(eval(emb)) for emb in self.wordemb_file['embedding']])
        else:
            self.wordemb_file = pd.read_csv("../data/word.csv", sep='\t', header=0)
            self.wordemb = nn.Parameter(torch.FloatTensor([list(eval(emb)) for emb in self.wordemb_file['embedding']]))

        # self.rgcn = RGCN(in_feats=50, out_feats=32)
        self.gat = dglnn.GATConv(50, 32, num_heads=1)
        self.trans = nn.Linear(512, 50)
        self.Discriminator = nn.Sequential()
        for j in range(len(layer_size) - 1):
            self.Discriminator.add_module("Linear_layer_%d" % j, nn.Linear(layer_size[j], layer_size[j + 1]))
            if j == len(layer_size) - 2:
                self.Discriminator.add_module("Sigmoid_layer_%d" % j, nn.Sigmoid())
            else:
                self.Discriminator.add_module("Relu_layer_%d" % j, nn.ReLU())

        self.bceloss = nn.BCELoss()

    def forward(self, g_list, graph_click, train=True):
        g_emb_list = []
        for bhg in g_list:
            for i in range(len(bhg)):
                if bhg[i].nodes['d_word'].data['word_id'].shape[0] > 0:
                    bhg[i].nodes['d_word'].data['hv'] = self.wordemb[bhg[i].nodes['d_word'].data['word_id']]
                else:
                    bhg[i].nodes['d_word'].data['hv'] = torch.ones(0, 50)
                if bhg[i].nodes['s_word'].data['word_id'].shape[0] > 0:
                    bhg[i].nodes['s_word'].data['hv'] = self.wordemb[bhg[i].nodes['s_word'].data['word_id']]
                else:
                    bhg[i].nodes['s_word'].data['hv'] = torch.ones(0, 50)
                if bhg[i].nodes['img'].data['img_emb'].shape[0] > 0:
                    bhg[i].nodes['img'].data['hv'] = self.trans(bhg[i].nodes['img'].data['img_emb'])
                else:
                    bhg[i].nodes['img'].data['hv'] = torch.ones(0, 50)
                bhg[i] = dgl.to_homogeneous(bhg[i], ndata=['hv'])
                bhg[i] = dgl.add_self_loop(bhg[i])

            g = dgl.batch(bhg)
            g.ndata['h'] = self.gat(g, g.ndata['hv'])
            with g.local_scope():
                g_emb = dgl.mean_nodes(g, 'h')
                # print("g_emb:", g_emb.shape)
                g_emb = g_emb.reshape(-1, 32)  # [b,1,32]
                # print(g_emb.shape)
                g_emb_list.append(g_emb)

        anchor = g_emb_list[0]
        if train:
            session_memory = anchor
            session_click_memory = graph_click

            pos = g_emb_list[1]
            s1 = self.Discriminator(torch.cat([anchor, pos], 1))

            col_num, emb_dim, row_num = session_memory.shape[0],session_memory.shape[1],anchor.shape[0]
            bt1 = anchor.repeat(1, col_num).reshape(-1, emb_dim)
            bt2 = session_memory.repeat(row_num, 1)
            s2_ = self.Discriminator(torch.cat([bt1, bt2], 1)).reshape(-1, col_num)

            click = []
            for c1 in graph_click:
                for c2 in session_click_memory:
                    if len(c1 & c2) == 0:
                        click.append(1)
                    else:
                        click.append(0)
            click_reverse = torch.tensor(click).reshape(-1, col_num)

            res = torch.mul(s2_, click_reverse)
            s2 = torch.max(res, 1).values.reshape(-1, 1)

            loss = self.bceloss(s1, torch.ones((s1.shape[0], 1))) + \
                   self.bceloss(s2, torch.zeros((s2.shape[0], 1)))
            yp1 = s1.squeeze().detach().numpy()
            yp2 = s2.squeeze().detach().numpy()

            acc = calc_acc(np.ones_like(yp1), np.where(yp1 > 0.5, 1, 0)) + calc_acc(np.zeros_like(yp2),
                                                                                    np.where(yp2 > 0.5, 1, 0))
            return loss, acc / 2.
        else:
            return anchor

    def get_neg(self, anchor):
        idx = torch.randperm(anchor.shape[0])
        neg = anchor[idx, :].view(anchor.size())
        return neg


def batcher():
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        # return batch_trees
        return batch

    return batcher_dev


def batcher2():
    def batcher_dev(batch):
        b, bn1, c = [], [], []
        for session_id in batch:
            b.append(session_graph[session_id])
            bn1.append(session_graph_pos[session_id])
            c.append(session_click[session_id])
        # return dgl.batch(b), dgl.batch(bn1), c
        return b, bn1, c

    return batcher_dev


# 训练
if __name__ == "__main__":
    test = False
    save_path = config.save_path+"_double_gat"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log = Logging(save_path)

    # load data
    if test:
        categorys = ['beauty']
    else:
        categorys = ['beauty', 'clothes', 'digital']

    for category in categorys:
        t0 = time()
        path = "../data/{}/".format(category)
        path1 = "../data_view1/{}/".format(category)
        session_graph = joblib.load(path1+"graphs_train.bin")  # dict{session_id:graph}
        session_graph_pos = joblib.load(path1+"graphs_pos2.bin")   # dict{session_id:graph_pos2}
        session_click = joblib.load(path+"session_click.bin")   # dict{session_id:click_items}
        log.record("load {} data done --- time {:.2f}, glist_len {}".format(category, time() - t0, len(session_graph)))

        t1 = time()
        model = HeteroClassifier(layer_size=config.layer_size, pretrain=True)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        log.record("start train --- time {:.2f}".format(time() - t1))

        model.train()
        for epoch in range(21):
            t2 = time()
            epoch_loss = 0.
            epoch_acc = []
            train_loader = DataLoader(dataset=list(session_graph.keys()), batch_size=config.batch_size, collate_fn=batcher2(), shuffle=True, drop_last=False)
            for batch, batch1, c in train_loader:
                loss, acc = model([batch, batch1], c, True)
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
                epoch_loss += loss.item()
                epoch_acc.append(acc)
                if test:break
            log.record("epoch {},loss {:.4f},acc {:.4f},time {:.2f}".format(epoch, epoch_loss, sum(epoch_acc) / len(epoch_acc), time() - t2))

            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), "{}/model_epoch{}.pth".format(save_path, epoch))

            if epoch == 20:
                tr_loader = DataLoader(dataset=list(session_graph.values()), batch_size=1024 * 8, collate_fn=batcher(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb = []
                    for batch_g in tr_loader:
                        semb = model([batch_g],None, False)
                        all_semb.append(semb)
                        if test:break
                all_semb = torch.cat(all_semb, 0)
                joblib.dump(all_semb, "{}/{}_train_session_emb{}.bin".format(save_path, category, epoch))

                session_test = joblib.load("../data_view1/{}/graphs_test.bin".format(category))
                te_loader = DataLoader(dataset=list(session_test.values()), batch_size=1024 * 8, collate_fn=batcher(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb = []
                    for batch_g in te_loader:
                        semb = model([batch_g],None, False)
                        all_semb.append(semb)
                        if test:break
                all_semb = torch.cat(all_semb, 0)
                joblib.dump(all_semb, "{}/{}_test_session_emb{}.bin".format(save_path, category, epoch))

            if epoch == 20:
                tg = time()
                path = "../data_view1/{}/".format(category)
                data = joblib.load(path+"graph_qid.bin")
                gs,qid = data[0],data[1]
                g_loader = DataLoader(dataset=gs, batch_size=1024 * 8, collate_fn=batcher(), shuffle=False, drop_last=False)

                model.eval()
                session_emb = []
                with torch.no_grad():
                    for batch_g in g_loader:
                        session_emb.append(model([batch_g],None, False))
                        if test:break
                slist = torch.cat(session_emb, 0)
                joblib.dump([slist.numpy().tolist(), qid], "{}/{}_query_emb_{}.bin".format(save_path, category, epoch))
                log.record("{} query finished, time {:.2f}".format(category, time() - tg))
