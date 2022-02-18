import pandas as pd
import itertools
import dgl
import torch
import joblib
import random
import os


def format(x):
    try:
        return float(x) / 127.
    except ValueError:
        x = 0.0
        return x


def session2graph(path1, path2, f):
    # 1 read
    session = pd.read_csv(path1 + "session.csv", sep='\t', header=0)  # session_id,query_id,split
    query = pd.read_csv(path1 + "query2.csv", sep='\t', header=0, index_col=0)  # query_id,word_id,query_type,image_emb
    valid_qids = set(pd.read_csv(path1 + "query2.csv", sep='\t', header=0, usecols=[0])['query_id'])
    # 2 deal
    gs = {}
    if (f == "train") or (f == 'pos2') or (f == 'pos1'):
        trte = "train"
    else:
        trte = 'test'
    for sid, se, sp in zip(session['session_id'], session['query_id'], session['split']):  # session
        if sp == trte:
            dcd = set()
            scs = set()
            sud = set()
            sus = set()
            dus = set()
            dud = set()
            iss = set()
            all_word_kgb = set()
            all_word_sim = set()
            all_img_sim = set()
            query_img = {}
            flag = '0'
            last_query = None

            s = se.split(',')
            new_s = []
            for qid in s:
                if qid not in valid_qids: continue
                new_s.append(qid)
            if len(new_s) < 2: continue

            # method1、打乱顺序
            if f == 'pos1':
                random.shuffle(new_s)
            # method2、随机删除一个query
            if f == 'pos2':
                index = random.randrange(len(new_s))
                new_s.pop(index)

            for qid in new_s:  # query
                q = query.loc[qid]
                cur_query = eval(q['word_id'])

                # concur | self_loop
                cl = set(itertools.permutations(cur_query, 2)) | set(zip(cur_query, cur_query))
                if flag != '0': ul = set(itertools.product(last_query, cur_query))

                if q['query_type'] == 'keyword_query':
                    all_word_kgb = all_word_kgb | set(cur_query)
                    dcd = dcd | cl

                    if flag == 'd':  # last is direct query
                        dud = dud | ul
                    if flag == 's':  # last is sim query
                        sud = sud | ul

                    flag = 'd'
                elif q['query_type'] == 'goods_query':
                    all_word_sim = all_word_sim | set(cur_query)
                    all_img_sim.add(qid)
                    scs = scs | cl
                    iss = iss | set(zip([qid] * len(cur_query), cur_query))
                    img_emb = q['image_emb'].split(",")
                    query_img[qid] = [format(x) for x in img_emb]

                    if flag == 'd':
                        dus = dus | ul
                    if flag == 's':
                        sus = sus | ul

                    flag = 's'

                last_query = cur_query

            reindex_word_kgb = dict(zip(all_word_kgb, range(len(all_word_kgb))))
            reindex_word_sim = dict(zip(all_word_sim, range(len(all_word_sim))))
            reindex_img_sim = dict(zip(all_img_sim, range(len(all_img_sim))))

            # graph
            dcd_n = [[], []]
            for i in dcd:
                dcd_n[0].append(reindex_word_kgb[i[0]])
                dcd_n[1].append(reindex_word_kgb[i[1]])
            scs_n = [[], []]
            for i in scs:
                scs_n[0].append(reindex_word_sim[i[0]])
                scs_n[1].append(reindex_word_sim[i[1]])
            sud_n = [[], []]
            for i in sud:
                sud_n[0].append(reindex_word_sim[i[0]])
                sud_n[1].append(reindex_word_kgb[i[1]])
            sus_n = [[], []]
            for i in sus:
                sus_n[0].append(reindex_word_sim[i[0]])
                sus_n[1].append(reindex_word_sim[i[1]])
            dus_n = [[], []]
            for i in dus:
                dus_n[0].append(reindex_word_kgb[i[0]])
                dus_n[1].append(reindex_word_sim[i[1]])
            dud_n = [[], []]
            for i in dud:
                dud_n[0].append(reindex_word_kgb[i[0]])
                dud_n[1].append(reindex_word_kgb[i[1]])
            iss_n = [[], []]
            for i in iss:
                iss_n[0].append(reindex_img_sim[i[0]])
                iss_n[1].append(reindex_word_sim[i[1]])

            hg = dgl.heterograph(
                {('d_word', 'concur', 'd_word'): (torch.LongTensor(dcd_n[0]), torch.LongTensor(dcd_n[1])),
                 ('s_word', 'concur', 's_word'): (torch.LongTensor(scs_n[0]), torch.LongTensor(scs_n[1])),
                 ('s_word', 'upd', 'd_word'): (torch.LongTensor(sud_n[0]), torch.LongTensor(sud_n[1])),
                 ('s_word', 'upd', 's_word'): (torch.LongTensor(sus_n[0]), torch.LongTensor(sus_n[1])),
                 ('d_word', 'upd', 's_word'): (torch.LongTensor(dus_n[0]), torch.LongTensor(dus_n[1])),
                 ('d_word', 'upd', 'd_word'): (torch.LongTensor(dud_n[0]), torch.LongTensor(dud_n[1])),
                 ('img', 'side', 's_word'): (torch.LongTensor(iss_n[0]), torch.LongTensor(iss_n[1]))},
                num_nodes_dict={'d_word': len(reindex_word_kgb),
                                's_word': len(reindex_word_sim),
                                'img': len(reindex_img_sim)})

            # 按vlaue[0,1,2,...]排好序的key[qid,qid,qid,...]
            hg.nodes['d_word'].data['word_id'] = torch.LongTensor(list(reindex_word_kgb.keys()))
            hg.nodes['s_word'].data['word_id'] = torch.LongTensor(list(reindex_word_sim.keys()))
            hg.nodes['img'].data['img_emb'] = torch.FloatTensor(
                [query_img[qid] for qid in list(reindex_img_sim.keys())])
            gs[sid] = hg

    if f == 'train':
        joblib.dump(gs, path2 + "graphs_train.bin")
    elif f == 'test':
        joblib.dump(gs, path2 + "graphs_test.bin")
    elif f == 'pos2':
        joblib.dump(gs, path2 + "graphs_pos2.bin")
    elif f == 'pos1':
        joblib.dump(gs, path2+"graphs_pos1.bin")


if __name__ == "__main__":
    for category in ['beauty', 'clothes', 'digital']:
        path1 = "../data/{}/".format(category)
        path2 = "../data_view2/{}/".format(category)
        if not os.path.exists(path2):
            os.makedirs(path2)
        for flag in ['train', 'test', 'pos2']:
            session2graph(path1, path2, flag)
            print("finish {} {}".format(category, flag))
