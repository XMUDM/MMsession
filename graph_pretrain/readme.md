# MM_Model

This repository holds the codes for [**Understanding Multi-modal Multi-query E-commerce Search**](https://arxiv.org/abs/xxxx.xxxxx).

## Environmental:

- Python==3.8.12
- Pytorch==1.10.1
- DGL==0.6.1


## Datasets:
url: https://www.aliyundrive.com/s/8XdkgFBGq5d

There are three categories, namely clothes, beauty and digital. Each category contains the following files.

  |  FileName   | Content  
  |  ----  | ----  | 
  | session.csv   | session_id,query_id,split
  | query2.csv   | query_id,word_id,query_type,image_emb 
  | word.csv   | word_id,embedding 
  | session_click.csv   | session_id,item_id


## Process:

- run `python session_click.py`, change data format.

- run `python session2graphx_xxx.py`, and output file will saved in `data12_view12/{category}/`.

  |  DataType   | graph_train.bin、graph_test.bin、graph_pos2.bin  | graph_qid.bin
  |  ----  | ----  | ----  |
  | text   | session2graph_view1.py | session2graphs_view1.py
  | text   | session2graph_view2.py | session2graphs_view2.py
  | text+img  | session2graph_double_view1.py |  session2graphs_double_view1.py
  | text+img  | session2graph_double_view2.py |  session2graphs_double_view2.py



## PreTrain:

  |  python xxx.py      | GraphType  | DataType | Pos | Neg | View
  |  ----  | ----  | ----  | ----  |----  |----  |
  | smodel_double_gat.py   | Isomorphism Graph | text+img | random mask query | hard negative sample | view1+view2
  | smodel_text_pos2neg2view12.py  | Heterogeneous Graph |  text | random mask query | hard negative sample | view1+view2
  | smodel_double_pos2neg2view12.py  | Heterogeneous Graph |  text+img | random mask query | hard negative sample | view1+view2
  | smodel_double_pos2negview12.py  | Heterogeneous Graph |  text+img | random mask query | random negative sample | view1+view2
  | smodel_double_pos2neg2view1.py  | Heterogeneous Graph |  text+img | random mask query | hard negative sample | view1
  | smodel_double_pos2neg2view2.py  | Heterogeneous Graph |  text+img | random mask query | hard negative sample | view2

run `python smodel_xxx.py`, and output file will saved in `save2/`.
