data url: https://www.aliyundrive.com/s/YCfuiBwotgP

there are three datasets named:clothes, beauty, digital. we selected 50000 sessions randomly from every dataset and organized their queries and clicks as three tables.

1. query_table(sessionid,query_id,query_emb,query_type,start_time,end_time,user_id,category,date,split)

2. click_table(sessionid,query_id,click_type,user_id,item_id,title_emb,time_stamp,is_buy,category,date,split)

3. pv_table(query_id,item_list)

open ranking model:
1. Attentive Long Short-Term Preference Modeling for Personalized Product Search

code url:https://github.com/guoyang9/ALSTP

2. Multi-modal Preference Modeling for Product Search

code url:https://github.com/guoyang9/TranSearch

3. A Hybrid Framework for Session Context Modeling

code url:https://github.com/xuanyuan14/HSCM-master

use our data to run ALSTP and TranSearch:
regard every record in click_table as an amazon_type record. 
choose(user_id,item_id,query_id,time_stamp,query_emb,title_emb as review_emb,split,item_list as ranking_list) from our three tables as input_table for raw ALSTP and TranSearch.
After pretrain graph model, add graph_emb as a new column to input_table join by query_id and then we can get a graph_input_table for ALSTP_graph and Transearch_graph.