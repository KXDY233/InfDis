# -*- coding: utf-8 -*-
"""

"""

import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import time
import joblib

import numpy as np
import gc
import math
from os import path
from timeit import default_timer as timer
from pprint import pprint
from typing import Dict, Optional


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



def get_nei_mu(targets, unionset, len_list):
    nei_mu = {}
    for i in tqdm(range(len(targets))):
        result = get_nei(targets[i], unionset, len_list)
        nei_mu[i] = result
    return nei_mu

def get_fe(id, dict, de, neighbor_dict, seeds, p):
    try:
        nei = neighbor_dict[id]
    except:
        nei = get_nei(set(dict[id]), unionset, len_list)
    nei = set(nei) - set(seeds)

    neighborhood = []
    for seed in seeds:
        neighborhood = neighborhood + dict[seed]
    neighborhood = neighborhood + dict[id]
    neighborhood = set(neighborhood)

    fe = 0

    tlist = []
    for item in nei:
        tu = len(setdict[item] & neighborhood)
        if tu == 1:
            tlist.append(de[item])
        else:
            fe = fe + (1-math.pow(1-p, tu))*(de[item]-tu)
    fe = fe + np.sum((np.array(tlist) - 1)*p)
    return fe


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

def get_te(id, seeds, dict):
    ground = []
    for item in seeds:
        ground += dict[item]
    return len(setdict[id]&set(ground))

def influence_discount(k, dict):
    seeds = []
    te = {}
    fe = {}
    yta = {}
    for key in list(dict.keys()):
        te[key] = 0
        fe[key] = 0
        for item in neighbor_dict[key]:
            if item != key:
                tu = len(set(dict[item]) & set(dict[key]))
                fe[key] += (de[item]-tu)*(1-(1-p)**tu)
        yta[key] = de[key] + fe[key]
    
    for i in tqdm(range(k)):
        seed = largest_valued_dict_key(yta)
        yta[seed] = -1
        seeds.append(seed)
        try:
            nei = neighbor_dict[seed]            
        except:
            nei = get_nei(set(dict[seed]), unionset, len_list)
 
        nei = set(nei) - set(seeds)
        # nei = set(nei)  & set(sizemax2)

        count = 0
        for item in nei:
            fe[item] = get_fe(item, dict, de, neighbor_dict, seeds, p)
            te[item] = get_te(item, seeds, dict)

            if te[item] == 1:
                yta[item] = (1-p) * (de[item] - 1 + fe[item])
            else:
                yta[item] = math.pow(1-p, te[item]) * (de[item] - te[item] + fe[item])
            count = count + 1
            if count%1000==0:
                print("number of iterations: ", count)

    return seeds



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="email")

    args = parser.parse_args()
    data = args.data
    data_path = "./data/" + data + "/"


    subset_path = data_path + "subset.dict"
    subset_dict = joblib.load(subset_path)


    dict_2 = {}
    dict_max2 = {}
    sizemax2 = []
    setdict = {}
    for key in list(subset_dict.keys()):
        setdict[key] = set(subset_dict[key])
        if len(subset_dict[key]) <= 2:
            dict_2[key] = subset_dict[key]
        else:
            dict_max2[key] = subset_dict[key]
            sizemax2.append(key)

    bigvalues = dict2listset(sort_by_values_len(dict_max2, 0))
    unionset = dict2listset(subset_dict)
    setsize2 = dict2listset(dict_2)
    

    setsize2 = set(subset_dict.keys()) - set(sizemax2)
    
    len_list = len(unionset)


    nei_path = data_path + "neighbors.dict"
    neighbor_dict = joblib.load(nei_path)
    seeds = []

    
    de = {}
    for key in list(subset_dict.keys()):
        de[key] = len(subset_dict[key])
    

    p_list = [0.01, 0.03, 0.05, 0.07, 0.1]
    for pro in p_list:
        p = pro
        k=20

        seeds = influence_discount(k, subset_dict)


        seed_path = data_path + "seeds_"+ str(p)
        joblib.dump(seeds, seed_path)

