# -*- coding: utf-8 -*-
"""

"""

import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import time
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
import argparse

def largest_valued_key(dic: Dict[str, set]) -> str:
    """Find the key with the largest value."""
    biggest_size = -1
    biggest_key = None
    for key, value in dic.items():
        length = len(value)
        if length > biggest_size:
            biggest_size = length
            biggest_key = key
    assert isinstance(biggest_key, str)
    return biggest_key


def largest_valued_key_simple(dic: Dict[str, int]) -> str:
    """Find the key with the largest value."""
    biggest_size = -1
    biggest_key = None
    for key, value in dic.items():
        length = value
        if length > biggest_size:
            biggest_size = length
            biggest_key = key
    assert isinstance(biggest_key, str)
    return biggest_key



def dict2listset(dict):
    values = []
    for key in list(dict.keys()):
        values.append(set(dict[key]))
    return values


def sort_by_values_len(dict, state):
    dict_len= {key: len(value) for key, value in dict.items()}
    import operator
    if state:
        sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse = False)
    else:
        sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse = True)
    cou = 0
    sorted_dict = {}
    for item in sorted_key_list:
        sorted_dict[cou] = dict[item[0]]
        cou+=1
    return sorted_dict

def get_nei(target, unionset, len_list):
    nei = []
    for i in range(len_list):
        comm = target & unionset[i]
        if comm:
            nei.append(i)
    return nei


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    return (a_set & b_set)


def largest_valued_dict_key(dic):
    """Find the key with the largest value."""
    biggest_size = -1
    biggest_key = None
    for key, value in dic.items():
        length = value
        if length > biggest_size:
            biggest_size = length
            biggest_key = key
    return biggest_key

def get_hedge_cover(list, dict):
    uni = []
    for item in list:
        uni = uni + dict[item]
    return len(set(uni))


def influence_spread_computation_IC_new(dict, seeds, neighbor_dict, unionset, earlystopping, rounds, p):
    influence = []
    len_list = len(dict.keys())
    for i in range(rounds):
        node_list = []
        active_hedges = []
        node_list.append(seeds)
        active_hedges = active_hedges + seeds
        
        for count in range(earlystopping):
            current_nodes = node_list.pop(0)
            tlist = []
            for node in current_nodes:
                try:
                    neighbors = neighbor_dict[node]
                except:
                    neighbors = get_nei(set(dict[node]), unionset, len_list)
                neighbors = set(neighbors) - set(active_hedges)
                for neighbor in neighbors:
                    te = len(unionset[node] & unionset[neighbor])
                    if (random() <= 1 - ((1-p)**te)):
                        tlist.append(neighbor)
            if tlist == []:
                break
            # else:
            #     print(tlist)
            node_list.append(tlist) 
            active_hedges = active_hedges + tlist
        influence.append(get_hedge_cover(active_hedges, dict))
        # influence = influence + get_hedge_cover(active_hedges, dict)
    return influence


def influence_spread_computation_IC_Mu(dict, seeds, neighbor_dict, unionset, earlystopping, num_mcmc, p):
    pool = multiprocessing.Pool()
    n=multiprocessing.cpu_count()
    results = []
    sub = int(num_mcmc / n)
    for i in range(n):
        result = pool.apply_async(influence_spread_computation_IC_new, args=(dict, seeds, neighbor_dict, unionset, earlystopping, sub, p,))
        results.append(result)
    pool.close()
    pool.join()
    influence_list = []
    for result in results:
        influence_list = influence_list + result.get()
    influence = sum(influence_list) / len(influence_list)
    pool.terminate()
    return influence, influence_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="email")
    parser.add_argument('--method', type=str, default="InfDis")
    parser.add_argument('--num_mcmc', type=int, default=100)
    parser.add_argument('--earlystopping', type=int, default=10)
    parser.add_argument('--probability', type=float, default=0.01)


    args = parser.parse_args()
    print(args.data)
    print(args.method)

    data = args.data
    data_path = "./data/" + data + "/"
    method = args.method
    pro = args.probability
    earlystopping = args.earlystopping
    num_mcmc = args.num_mcmc

    if method == "InfDis":
        seeds_path = data_path + "seeds_" + str(pro)
    elif method == "Between":
        seeds_path = data_path + 'seeds_betweeness.list'
    elif method == "Degree":
        seeds_path = data_path + 'seeds_degree.list'
    elif method == "HyperRank":
        seeds_path = data_path + 'seeds_pagerank.list'
    elif method == "Greedy":
        seeds_path = data_path + 'seeds_greedy_0.05.list'

    subset_dict_path = data_path + "subset.dict"
    subset_dict = joblib.load(subset_dict_path)

    seeds = list(joblib.load(seeds_path))
    unionset_path = data_path + "unionset.set"
    unionset = joblib.load(unionset_path)
    
    nei_path = data_path + "neighbors.dict"
    neighbor_dict = joblib.load(nei_path)
    
    len_list = len(subset_dict.keys())

    result = []
    for nt in [1, 10, 20, 30, 40, 50]:
        tseeds = seeds[0:nt]
        influence, influence_list = influence_spread_computation_IC_Mu(subset_dict, tseeds, neighbor_dict, unionset, earlystopping, num_mcmc, pro)
        avg_influence = np.mean(influence_list)
        std_influence = np.std(influence_list)
        result.append((avg_influence,std_influence))
    outfile = "./result/" + data + "_" + method + ".txt"
    results = np.array(result)
    np.savetxt(outfile, results, fmt='%.3f', delimiter='\t')

    
