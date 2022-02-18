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

import config
from Utils import Logging


class VIEW1(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        rel_names = ['concur', 'upd', 'side']

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, out_feats, num_heads=1) for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        return h


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

        self.view1 = VIEW1(in_feats=50, out_feats=32)
        self.view2 = VIEW2(in_feats=50, hid_feats=32, out_feats=32)
        self.trans = nn.Linear(512, 50)

        self.Discriminator1 = nn.Sequential()
        for j in range(len(layer_size) - 1):
            self.Discriminator1.add_module("Linear_layer_%d" % j, nn.Linear(layer_size[j], layer_size[j + 1]))
            if j == len(layer_size) - 2:
                self.Discriminator1.add_module("Sigmoid_layer_%d" % j, nn.Sigmoid())
            else:
                self.Discriminator1.add_module("Relu_layer_%d" % j, nn.ReLU())

        self.Discriminator2 = nn.Sequential()
        for j in range(len(layer_size) - 1):
            self.Discriminator2.add_module("Linear_layer_%d" % j, nn.Linear(layer_size[j], layer_size[j + 1]))
            if j == len(layer_size) - 2:
                self.Discriminator2.add_module("Sigmoid_layer_%d" % j, nn.Sigmoid())
            else:
                self.Discriminator2.add_module("Relu_layer_%d" % j, nn.ReLU())

        self.bceloss = nn.BCELoss()

    def forward(self, g_list1, g_list2, train=True):
        if train:
            loss1 = self.get_view1(g_list1, train)
            loss2 = self.get_view2(g_list2, train)
            return loss1, loss2
        else:
            g_emb1 = self.get_view1(g_list1, train)
            g_emb2 = self.get_view2(g_list2, train)
            return g_emb1, g_emb2

    def get_view1(self, g_list, train=True):
        g_emb_list = []
        for g in g_list:
            x_src = {'d_word': self.wordemb[g.nodes['d_word'].data['word_id']],
                     's_word': self.wordemb[g.nodes['s_word'].data['word_id']],
                     'img': self.trans(g.nodes['img'].data['img_emb'])}
            x_dst = {'d_word': self.wordemb[g.nodes['d_word'].data['word_id']],
                     's_word': self.wordemb[g.nodes['s_word'].data['word_id']]}
            h = self.view1(g, (x_src,x_dst))
            with g.local_scope():
                g.ndata['h'] = h
                g_emb = 0
                for ntype in ['d_word', 's_word']:  # g.ntypes
                    g_emb = g_emb + dgl.mean_nodes(g, 'h', ntype=ntype)
                g_emb = g_emb / 2.0
                g_emb = g_emb.reshape(-1, 32)
                g_emb_list.append(g_emb)

        anchor = g_emb_list[0]

        if train:
            pos = g_emb_list[1]
            s1 = self.Discriminator1(torch.cat([anchor, pos], 1))

            neg = self.get_neg(anchor)
            s2 = self.Discriminator1(torch.cat([anchor, neg], 1))

            loss1 = self.bceloss(s1, torch.ones((s1.shape[0], 1))) + self.bceloss(s2, torch.zeros((s2.shape[0], 1)))

            return loss1
        else:
            return anchor

    def get_view2(self, g_list, train=True):
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
            pos = g_emb_list[1]
            s1 = self.Discriminator2(torch.cat([anchor, pos], 1))

            neg = self.get_neg(anchor)
            s2 = self.Discriminator1(torch.cat([anchor, neg], 1))

            loss2 = self.bceloss(s1, torch.ones((s1.shape[0], 1))) + \
                    self.bceloss(s2, torch.zeros((s2.shape[0], 1)))

            return loss2
        else:
            return anchor

    def get_neg(self, anchor):
        idx = torch.randperm(anchor.shape[0])
        neg = anchor[idx, :].view(anchor.size())
        return neg


def batcher1():
    def batcher_dev(batch):
        v1b1, v2b1= [], []
        for session_id in batch:
            v1b1.append(session_graph1[session_id])
            v2b1.append(session_graph2[session_id])
        return dgl.batch(v1b1),dgl.batch(v2b1)

    return batcher_dev


def batcher2():
    def batcher_dev(batch):
        v1b1, v1b2, v2b1, v2b2 = [], [], [], []
        for session_id in batch:
            v1b1.append(session_graph1[session_id])
            v1b2.append(session_graph1_pos[session_id])
            v2b1.append(session_graph2[session_id])
            v2b2.append(session_graph2_pos[session_id])
        return dgl.batch(v1b1), dgl.batch(v1b2), dgl.batch(v2b1), dgl.batch(v2b2)

    return batcher_dev


# 训练
if __name__ == "__main__":
    test = False
    save_path = config.save_path + "_double_pos2negview12"
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
        path1 = "../data_view1/{}/".format(category)
        path2 = "../data_view2/{}/".format(category)
        session_graph1 = joblib.load(path1 + "graphs_train.bin")     # dict{session_id:graph}
        session_graph1_pos = joblib.load(path1 + "graphs_pos2.bin")  # dict{session_id:graph_pos2}
        session_graph2 = joblib.load(path2 + "graphs_train.bin")     # dict{session_id:graph}
        session_graph2_pos = joblib.load(path2 + "graphs_pos2.bin")  # dict{session_id:graph_pos2}
        log.record("load {} data done --- time {:.2f}".format(category, time() - t0))

        t1 = time()
        model = HeteroClassifier(layer_size=config.layer_size, pretrain=True)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        log.record("start train --- time {:.2f}".format(time() - t1))

        model.train()
        for epoch in range(21):
            t2 = time()
            epoch_loss1 = 0.
            epoch_loss2 = 0.
            train_loader = DataLoader(dataset=list(session_graph1.keys()), batch_size=config.batch_size, collate_fn=batcher2(), shuffle=True, drop_last=False)
            for v1b1, v1b2, v2b1, v2b2 in train_loader:
                loss1, loss2 = model([v1b1, v1b2], [v2b1, v2b2], True)
                loss = loss1+loss2
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                if test: break
            log.record("epoch {},loss {:.4f}+{:.4f},time {:.2f}".format(epoch, epoch_loss1, epoch_loss2, time() - t2))

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), "{}/{}_model_epoch{}.pth".format(save_path, category, epoch))

            if epoch == 20:
                tr_loader = DataLoader(dataset=list(session_graph1.keys()), batch_size=1024*8, collate_fn=batcher1(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb1 = []
                    all_semb2 = []
                    for v1b1, v2b1 in tr_loader:
                        semb1,semb2 = model([v1b1], [v2b1], False)
                        all_semb1.append(semb1)
                        all_semb2.append(semb2)
                        if test: break
                all_semb1 = torch.cat(all_semb1, 0)
                all_semb2 = torch.cat(all_semb2, 0)
                joblib.dump(all_semb1, "{}/{}_train_session_emb{}_view1.bin".format(save_path, category, epoch))
                joblib.dump(all_semb2, "{}/{}_train_session_emb{}_view2.bin".format(save_path, category, epoch))

                session_graph1 = joblib.load("../data_view1/{}/graphs_test.bin".format(category))
                session_graph2 = joblib.load("../data_view2/{}/graphs_test.bin".format(category))
                te_loader = DataLoader(dataset=list(session_graph1.keys()), batch_size=1024*8, collate_fn=batcher1(), shuffle=False, drop_last=False)
                model.eval()
                with torch.no_grad():
                    all_semb1 = []
                    all_semb2 = []
                    for v1b1, v2b1 in te_loader:
                        semb1,semb2 = model([v1b1], [v2b1], False)
                        all_semb1.append(semb1)
                        all_semb2.append(semb2)
                        if test: break
                all_semb1 = torch.cat(all_semb1, 0)
                all_semb2 = torch.cat(all_semb2, 0)
                joblib.dump(all_semb1, "{}/{}_test_session_emb{}_view1.bin".format(save_path, category, epoch))
                joblib.dump(all_semb2, "{}/{}_test_session_emb{}_view2.bin".format(save_path, category, epoch))


            if epoch == 20:
                tg = time()
                data1 = joblib.load("../data_view1/{}/graph_qid.bin".format(category))
                data2 = joblib.load("../data_view2/{}/graph_qid.bin".format(category))
                gs1, qid = data1[0], data1[1]
                session_graph1 = dict(zip(qid,gs1))
                gs2, qid = data2[0], data2[1]
                session_graph2 = dict(zip(qid,gs2))
                g_loader = DataLoader(dataset=qid, batch_size=1024*8, collate_fn=batcher1(), shuffle=False, drop_last=False)

                model.eval()
                with torch.no_grad():
                    all_semb1 = []
                    all_semb2 = []
                    for v1b1, v2b1 in g_loader:
                        semb1,semb2 = model([v1b1], [v2b1], False)
                        all_semb1.append(semb1)
                        all_semb2.append(semb2)
                        if test: break
                all_semb1 = torch.cat(all_semb1, 0)
                all_semb2 = torch.cat(all_semb2, 0)
                joblib.dump([all_semb1.numpy().tolist(), qid], "{}/{}_query_emb_{}_view1.bin".format(save_path, category, epoch))
                joblib.dump([all_semb2.numpy().tolist(), qid], "{}/{}_query_emb_{}_view2.bin".format(save_path, category, epoch))
                joblib.dump([torch.cat([all_semb1,all_semb2],1).numpy().tolist(), qid], "{}/{}_query_emb_{}_view12.bin".format(save_path, category, epoch))
                log.record("{} query finished, time {:.2f}".format(category, time() - tg))
