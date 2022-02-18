import pandas as pd
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


def session2graph(path1,path2,f):
    # 1 read
    session = pd.read_csv(path1 + "session.csv", sep='\t', header=0)  # session_id,query_id,split
    print(len(session))
    query = pd.read_csv(path1 + "query2.csv", sep='\t', header=0, index_col=0)
    valid_qids = set(pd.read_csv(path1 + "query2.csv", sep='\t', header=0, usecols=[0])['query_id'])
    # 2 deal
    gs = {}
    if (f == "train") or (f == 'pos2') or (f == 'pos1'):
        trte = "train"
    else:
        trte = 'test'

    for sid, se, sp in zip(session['session_id'], session['session'], session['split']):  # session
        if sp == trte:
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

            wq = [[], []]
            qq = [[], []]
            all_word = set()
            last_center_q = None
            center_q = 0

            for qid in new_s:  # query
                q = query.loc[qid]
                cur_query = eval(q['word_id'])

                wq[0].extend(cur_query)
                wq[1].extend([center_q] * len(cur_query))
                all_word |= set(cur_query)

                if last_center_q != None:
                    qq[0].append(last_center_q)
                    qq[1].append(center_q)
                last_center_q = center_q
                center_q += 1

            reindex_word = dict(zip(all_word, range(len(all_word))))

            # graph
            wq[0] = [reindex_word[w] for w in wq[0]]
            qq[0].extend(list(range(center_q)))
            qq[1].extend(list(range(center_q)))

            hg = dgl.heterograph(
                {('word', 'side', 'query'): (torch.LongTensor(wq[0]), torch.LongTensor(wq[1])),
                 ('query', 'upd', 'query'): (torch.LongTensor(qq[0]), torch.LongTensor(qq[1]))},
                num_nodes_dict={'word': len(reindex_word),
                                'query': center_q})

            hg.nodes['word'].data['word_id'] = torch.LongTensor(list(reindex_word.keys()))
            hg.nodes['query'].data['query_num'] = torch.LongTensor([center_q]*center_q)
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
        path2 = "../data2_view2/{}/".format(category)
        if not os.path.exists(path2):
            os.makedirs(path2)
        for flag in ['train', 'test', 'pos2']:
            session2graph(path1, path2, flag)
            print("finish {} {}".format(category, flag))
