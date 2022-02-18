import pandas as pd
import dgl
import torch
import joblib
import os

def format(x):
    try:
        return float(x) / 127.
    except ValueError:
        x = 0.0
        return x


def session2graph(path1,path2):
    # 1 read
    session = pd.read_csv(path1 + "session.csv", sep='\t', header=0)  # session_id,query_id,split
    print(len(session))
    query = pd.read_csv(path1 + "query2.csv", sep='\t', header=0, index_col=0)
    valid_qids = set(pd.read_csv(path1 + "query2.csv", sep='\t', header=0, usecols=[0])['query_id'])
    # 2 deal
    gs = []
    qids = []
    for sid, se, sp in zip(session['session_id'], session['session'], session['split']):  # session
        wq = [[], []]
        iq = [[], []]
        qq = [[], []]
        query_img = {}
        last_center_q = None
        center_q = 0
        s = se.split(',')

        for qid in s:  # query
            if qid not in valid_qids: continue
            q = query.loc[qid]
            cur_query = eval(q['word_id'])

            wq[0].extend(cur_query)
            wq[1].extend([center_q] * len(cur_query))
            if q['query_type'] == 'goods_query':
                iq[0].append(qid)
                iq[1].append(center_q)

                img_emb = q['image_emb'].split(",")
                query_img[qid] = [format(x) for x in img_emb]
            if last_center_q != None:
                qq[0].append(last_center_q)
                qq[1].append(center_q)
            last_center_q = center_q
            center_q += 1

            reindex_word = dict(zip(list(set(wq[0])), range(len(set(wq[0])))))
            reindex_img = dict(zip(list(set(iq[0])), range(len(set(iq[0])))))

            # graph
            qq[0].extend(list(range(center_q)))
            qq[1].extend(list(range(center_q)))

            hg = dgl.heterograph(
                {('word', 'side', 'query'): (torch.LongTensor([reindex_word[w] for w in wq[0]]), torch.LongTensor(wq[1])),
                 ('img', 'side', 'query'): (torch.LongTensor([reindex_img[i] for i in iq[0]]), torch.LongTensor(iq[1])),
                 ('query', 'upd', 'query'): (torch.LongTensor(qq[0]), torch.LongTensor(qq[1]))},
                num_nodes_dict={'word': len(reindex_word),
                                'img': len(reindex_img),
                                'query': center_q})

            hg.nodes['word'].data['word_id'] = torch.LongTensor(list(reindex_word.keys()))
            hg.nodes['img'].data['img_emb'] = torch.FloatTensor([query_img[qid] for qid in list(reindex_img.keys())])
            hg.nodes['query'].data['query_num'] = torch.LongTensor([center_q]*center_q)
            gs.append(hg)
            qids.append(qid)

    joblib.dump([gs, qids], path2 + "graph_qid.bin")



if __name__ == "__main__":
    for category in ['beauty', 'clothes', 'digital']:
        path1 = "../data/{}/".format(category)
        path2 = "../data_view2/{}/".format(category)
        if not os.path.exists(path2):
            os.makedirs(path2)
        session2graph(path1,path2)
        print("finish ", category)
