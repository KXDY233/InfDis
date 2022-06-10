# -*- coding: utf-8 -*-
"""
betweenness centratity
"""



import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from time import time
from random import random
from numba import jit
import cProfile
import pstats
import io
import joblib
import networkx as nx
import pickle
import numpy.matlib
import numpy as np
import gc
import math
from os import path
from timeit import default_timer as timer
from pprint import pprint
from typing import Dict, Optional



if __name__ == '__main__':


    st = time()

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="email")

    args = parser.parse_args()
    data = args.data
    data_path = "./data/" + data + "/"

    subset_dict_path = data_path + "subset.dict"
    subset_dict = joblib.load(subset_dict_path)
    
    nodes = []
    pair = []
    mm = max(subset_dict.keys())
    for key in list(subset_dict):
        nodes.append(key)
        for item in subset_dict[key]:
            pair.append([key, int(item) + int(mm) + 1])
            
    graph = nx.Graph()
    graph.add_edges_from(pair)



    betweenness = nx.betweenness_centrality(graph, k= 100, normalized=True)
    L = sorted(betweenness.items(),key=lambda item:item[1],reverse=True)
    
    count = 0
    seeds = []
    for a,b in L:
        if a<mm+1:
            seeds.append(a)
            count += 1
        if count>=50:
            break
    
    seed_path = data_path + "seeds_betweeness.list"
    joblib.dump(seeds, seed_path)

    et = time()-st
    print("Time of Between: ", et)