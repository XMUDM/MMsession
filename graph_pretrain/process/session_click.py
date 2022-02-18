import pandas as pd
import joblib
from collections import defaultdict


def p_session_click(path):
    session_clicks = pd.read_csv(path + "session_click.csv", sep='\t', header=0)  # session_id, item_id
    sc_dic = defaultdict(set)
    for sid, si in zip(session_clicks['session_id'], session_clicks['item_id']):
        sc_dic[sid].add(si)
    joblib.dump(sc_dic,path + "session_click.bin")


if __name__ == "__main__":
    for category in ['beauty', 'clothes', 'digital']:
        path = "../data/{}/".format(category)
        p_session_click(path)
