import joblib
import pandas as pd
import itertools
import dgl
import torch
import os


def session2graphs(path1, path2):
    # 1 read
    session = pd.read_csv(path1+"session.csv", sep='\t', header=None, usecols=[0, 1, 3], names=['session_id', 'session', 'split'])
    query = pd.read_csv(path1+"query2.csv", sep='\t', header=0, index_col=0)
    valid_qids = set(pd.read_csv(path1+"query2.csv", sep='\t', header=0)['qid'])
    print(query.head())
    # 2 deal
    gs = []
    qids = []
    for se in session['session']:  # session
        dcd = set()
        scs = set()
        sud = set()
        sus = set()
        dus = set()
        dud = set()
        all_word = set()
        flag = '0'
        last_query = None

        s = se.split(',')
        for qid in s:  # query
            if qid not in valid_qids: continue

            q = query.loc[qid]
            cur_query = eval(q['word_id'])
            all_word = all_word | set(cur_query)

            cl = set(itertools.permutations(cur_query, 2))
            if flag != '0': ul = set(itertools.product(last_query, cur_query))

            if q['query_type'] == 'keyword_query':
                dcd = dcd | cl

                if flag == 'd':  # last is direct query
                    dud = dud | ul
                if flag == 's':  # last is sim query
                    dus = dus | ul

                flag = 'd'
            elif q['query_type'] == 'goods_query':
                scs = scs | cl

                if flag == 'd':
                    sud = sud | ul
                if flag == 's':
                    sus = sus | ul

                flag = 's'

            last_query = cur_query

            # 为每个query_id构建以改query结尾的session图
            reindex_word = dict(zip(all_word, range(len(all_word))))
            dcd_n = [[], []]
            for i in dcd:
                dcd_n[0].append(reindex_word[i[0]])
                dcd_n[1].append(reindex_word[i[1]])
            scs_n = [[], []]
            for i in scs:
                scs_n[0].append(reindex_word[i[0]])
                scs_n[1].append(reindex_word[i[1]])
            sud_n = [[], []]
            for i in sud:
                sud_n[0].append(reindex_word[i[0]])
                sud_n[1].append(reindex_word[i[1]])
            sus_n = [[], []]
            for i in sus:
                sus_n[0].append(reindex_word[i[0]])
                sus_n[1].append(reindex_word[i[1]])
            dus_n = [[], []]
            for i in dus:
                dus_n[0].append(reindex_word[i[0]])
                dus_n[1].append(reindex_word[i[1]])
            dud_n = [[], []]
            for i in dud:
                dud_n[0].append(reindex_word[i[0]])
                dud_n[1].append(reindex_word[i[1]])

            hg = dgl.heterograph({('d_word', 'concur', 'd_word'): (torch.LongTensor(dcd_n[0]), torch.LongTensor(dcd_n[1])),
                                  ('s_word', 'concur', 's_word'): (torch.LongTensor(scs_n[0]), torch.LongTensor(scs_n[1])),
                                  ('s_word', 'upd', 'd_word'): (torch.LongTensor(sud_n[0]), torch.LongTensor(sud_n[1])),
                                  ('s_word', 'upd', 's_word'): (torch.LongTensor(sus_n[0]), torch.LongTensor(sus_n[1])),
                                  ('d_word', 'upd', 's_word'): (torch.LongTensor(dus_n[0]), torch.LongTensor(dus_n[1])),
                                  ('d_word', 'upd', 'd_word'): (torch.LongTensor(dud_n[0]), torch.LongTensor(dud_n[1]))},
                                 num_nodes_dict={'d_word': len(reindex_word), 's_word': len(reindex_word)})
            hg.nodes['d_word'].data['word_id'] = torch.LongTensor(list(reindex_word.keys()))
            hg.nodes['s_word'].data['word_id'] = torch.LongTensor(list(reindex_word.keys()))
            qids.append(qid)
            gs.append(hg)

    joblib.dump([gs, qids], path2+"graph_qid.bin")


if __name__ == "__main__":
    for category in ['beauty', 'clothes', 'digital']:
        path1 = "../data/{}/".format(category)
        path2 = "../data2_view1/{}/".format(category)
        if not os.path.exists(path2):
            os.makedirs(path2)
        session2graphs(path1,path2)



