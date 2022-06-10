# -*- coding: utf-8 -*-
"""

"""

import joblib
import numpy as np
import gc
from time import time

def sort_by_values_len(dict, state):
    dict_len = {key: len(value) for key, value in dict.items()}
    import operator
    if state:
        sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=False)
    else:
        sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=True)
    cou = 0
    sorted_dict = {}
    for item in sorted_key_list:
        sorted_dict[cou] = dict[item[0]]
        cou += 1
    return sorted_dict


if __name__ == '__main__':

    st = time()

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="email")

    args = parser.parse_args()
    data = args.data
    data_path = "./data/" + data + "/"

    seeds = list(np.arange(50))
    seed_path = data_path + "seeds_degree.list"
    joblib.dump(seeds, seed_path)

    et = time() - st

    print("Time of Degree: ", et)


