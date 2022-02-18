import warnings

warnings.filterwarnings("ignore")

import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
from time import time
import os
import joblib
from sklearn import metrics

import config
from Utils import Logging


class VIEW2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        rel_names = ['side', 'upd']

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')

        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, layer_size, pretrain=True):
        super().__init__()
        if pretrain:
            self.wordemb_file = pd.read_csv("../data/word.csv", sep='\t', header=0)
            self.wordemb = torch.FloatTensor([list(eval(emb)) for emb in self.wordemb_file['embedding']])
        else:
            self.wordemb_file = pd.read_csv("../data/word.csv", sep='\t', header=0)
            self.wordemb = nn.Parameter(torch.FloatTensor([list(eval(emb)) for emb in self.wordemb_file['embedding']]))

        self.view2 = VIEW2(in_feats=50, hid_feats=32, out_feats=32)
        self.trans = nn.Linear(512, 50)

        self.Discriminator = nn.Sequential()
        for j in range(len(layer_size) - 1):
            self.Discriminator.add_module("Linear_layer_%d" % j, nn.Linear(layer_size[j], layer_size[j + 1]))
            if j == len(layer_size) - 2:
                self.Discriminator.add_module("Sigmoid_layer_%d" % j, nn.Sigmoid())
            else:
                self.Discriminator.add_module("Relu_layer_%d" % j, nn.ReLU())

        self.bceloss = nn.BCELoss()

        self.item_memory = None
        self.itemc_memory = None

    def forward(self, g_list, graph_click, train=True):
        g_emb_list = []
        for g in g_list:
            h = {'word': self.wordemb[g.nodes['word'].data['word_id']],
                 'img': self.trans(g.nodes['img'].data['img_emb']),
                 'query': torch.zeros(g.num_nodes('query'), 50)}
            h = self.view2(g, h)
            with g.local_scope():
                g.ndata['h'] = h
                g_emb = dgl.mean_nodes(g, 'h', ntype='query')
                g_emb_list.append(g_emb)

        anchor = g_emb_list[0]

        if train:
            session_memory = anchor
            session_click_memory = graph_click

            pos = g_emb_list[1]
            s1 = self.Discriminator(torch.cat([anchor, pos], 1))

            col_num, emb_dim, row_num = session_memory.shape[0], session_memory.shape[1], anchor.shape[0]
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

            loss2 = self.bceloss(s1, torch.ones((s1.shape[0], 1))) + \
                    self.bceloss(s2, torch.zeros((s2.shape[0], 1)))

            return loss2
        else:
            return anchor

    def get_neg(self, anchor):
        idx = torch.randperm(anchor.shape[0])
        neg = anchor[idx, :].view(anchor.size())
        return neg


def batcher():
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return batch_trees

    return batcher_dev


def batcher2():
    def batcher_dev(batch):
        b, bn1, c = [], [], []
        for session_id in batch:
            b.append(session_graph[session_id])
            bn1.append(session_graph_pos[session_id])
            c.append(session_click[session_id])
        return dgl.batch(b), dgl.batch(bn1), c

    return batcher_dev


# 训练
if __name__ == "__main__":
    test = False
    save_path = config.save_path + "_double_pos2neg2view2"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log = Logging(save_path)

    # load data
    if test:
        categorys = ['furniture']
    else:
        categorys = ['beauty', 'clothes', 'digital']

    for category in categorys:
        t0 = time()
        path = "../data/{}/".format(category)
        path2 = "../data_view2/{}/".format(category)
        session_graph = joblib.load(path2 + "graphs_train.bin")  # dict{session_id:graph}
        session_graph_pos = joblib.load(path2 + "graphs_pos2.bin")  # dict{session_id:graph_pos2}
        session_click = joblib.load(path + "session_click.bin")  # dict{session_id:click_items}
        log.record("load {} data done --- time {:.2f}, glist_len {}".format(category, time() - t0, len(session_graph)))

        t1 = time()
        model = HeteroClassifier(layer_size=config.layer_size, pretrain=True)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        log.record("start train --- time {:.2f}".format(time() - t1))

        model.train()
        for epoch in range(21):
            t2 = time()
            epoch_loss = 0.
            train_loader = DataLoader(dataset=list(session_graph.keys()), batch_size=config.batch_size, collate_fn=batcher2(), shuffle=True, drop_last=False)
            for batch, batch1, c in train_loader:
                loss = model([batch, batch1], c, True)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                if test: break
            log.record("epoch {},loss {:.4f}, time {:.2f}".format(epoch, epoch_loss,time() - t2))

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), "{}/model_epoch{}.pth".format(save_path, epoch))

            if epoch == 20:
                tr_loader = DataLoader(dataset=list(session_graph.values()), batch_size=1024 * 8,collate_fn=batcher(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb = []
                    for batch_g in tr_loader:
                        semb = model([batch_g], None, False)
                        all_semb.append(semb)
                        if test: break
                all_semb = torch.cat(all_semb, 0)
                joblib.dump(all_semb, "{}/{}_train_session_emb{}_view2.bin".format(save_path, category, epoch))

                session_test = joblib.load("../data_view2/{}/graphs_test.bin".format(category))
                te_loader = DataLoader(dataset=list(session_test.values()), batch_size=1024 * 8, collate_fn=batcher(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb = []
                    for batch_g in te_loader:
                        semb = model([batch_g], None, False)
                        all_semb.append(semb)
                        if test: break
                all_semb = torch.cat(all_semb, 0)
                joblib.dump(all_semb, "{}/{}_test_session_emb{}_view2.bin".format(save_path, category, epoch))

            if epoch == 20:
                tg = time()
                path = "../data_view2/{}/".format(category)
                data = joblib.load(path + "graph_qid.bin")
                gs, qid = data[0], data[1]
                g_loader = DataLoader(dataset=gs, batch_size=1024 * 8, collate_fn=batcher(), shuffle=False,drop_last=False)

                model.eval()
                session_emb = []
                with torch.no_grad():
                    for batch_g in g_loader:
                        session_emb.append(model([batch_g], None, False))
                        if test: break
                slist = torch.cat(session_emb, 0)
                joblib.dump([slist.numpy().tolist(), qid],
                            "{}/{}_query_emb_{}_view2.bin".format(save_path, category, epoch))
                log.record("{} query finished, time {:.2f}".format(category, time() - tg))
