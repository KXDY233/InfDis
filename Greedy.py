# coding=utf-8
import pickle, joblib
from tqdm import tqdm
from random import random
import multiprocessing
from time import time
import numpy as np


def get_nei(target, unionset, len_list):
    nei = []
    for i in range(len_list):
        comm = target & unionset[i]
        if comm:
            nei.append(i)
    return nei


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
                    if (random() <= 1 - ((1 - p) ** te)):
                        tlist.append(neighbor)
            if tlist == []:
                break
            node_list.append(tlist)
            active_hedges = active_hedges + tlist
        influence.append(get_hedge_cover(active_hedges, dict))
        # influence = influence + get_hedge_cover(active_hedges, dict)
    return np.mean(np.array(influence)), influence



def this_node_influence(S, u_list, subset_dict, num_MC, earlystopping, neighbor_dict, unionset, pro):
    influence_List = []
    for u in u_list:
        S_union_u = S | {u}  # S Union {u}
        influence_bigger, inf_list = influence_spread_computation_IC_new(subset_dict, list(S_union_u), neighbor_dict, unionset, earlystopping, num_MC, pro)
        influence_List.append((u, influence_bigger, inf_list))
    return influence_List

def simgreedy(subset_dict, num_seeds, num_MC,earlystopping, neighbor_dict, unionset, pro):

    universe = set(subset_dict.keys()) 
    S = set() 
    influence_estimate = []

    for i in tqdm(range(num_seeds)):

        search_list = universe - S  #
        search_list = list(search_list)

        pool = multiprocessing.Pool()
        n = multiprocessing.cpu_count()

        each_pool_size = int(np.ceil(len(search_list) / n))  
        actual_process = int(np.ceil(len(search_list) / each_pool_size))  
        chunck_range = [np.array([0, each_pool_size - 1]) + each_pool_size * a for a in range(actual_process)] 


        results = []
        for i in range(actual_process):
            u_index = chunck_range[i]
            u_list = search_list[u_index[0]: min(u_index[1], len(search_list) - 1) + 1]

            result = pool.apply_async(this_node_influence, args=(S, u_list, subset_dict, num_MC, earlystopping, neighbor_dict, unionset, pro, ))
            results.append(result)

        pool.close()
        pool.join()
        influence_list = []  
        influence_array = []
        edge_lists = []
        for result in results:
            items = result.get()  # [(u, influence)]
            for (u, influence, influence_record) in items:
                edge_lists.append(u)
                influence_list.append(influence)
                influence_array.append(influence_record)

        pool.terminate()

        max_index = np.array(influence_list).argmax() 
        max_influence_record = influence_array[max_index]
        max_u = edge_lists[max_index] 

        S.add(max_u)
        influence_estimate.append(max_influence_record) 


    return S, influence_estimate




def dict2listset(dict):
    values = []
    for key in list(dict.keys()):
        values.append(set(dict[key]))
    return values


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="w3cemail")
    parser.add_argument('--num_mcmc', type=int, default=12)
    parser.add_argument('--earlystopping', type=int, default=10)
    parser.add_argument('--probability', type=float, default=0.05)

    args = parser.parse_args()
    data = args.data
    earlystopping = args.earlystopping
    num_mcmc = args.num_mcmc

    data_path = "./data/" + data + "/"

    subset_dict_path = data_path + "subset.dict"
    subset_dict = joblib.load(subset_dict_path)

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

    unionset = dict2listset(subset_dict)
    setsize2 = set(subset_dict.keys()) - set(sizemax2)

    len_list = len(subset_dict.keys())

    unionset_path = data_path + "unionset.set"
    unionset = joblib.load(unionset_path)

    nei_path = data_path + "neighbors.dict"
    neighbor_dict = joblib.load(nei_path)



    result = []

    result_vary_p = []
    # p_list = [0.001, 0.003, 0.005, 0.007, 0.01]
    p_list = [0.01]
    for pro in p_list:
        print ('################ p = ', pro)

        if pro == 0.01:
            seeds_num = 50
        else:
            seeds_num = 20

        st = time()
        seeds, influence_list = simgreedy(subset_dict=subset_dict, num_seeds=seeds_num, num_MC=num_mcmc, earlystopping = earlystopping, neighbor_dict = neighbor_dict, unionset=unionset, pro=pro)

        # print(list(seeds))
        et = time()-st
        print ("Time of Greedy: ", et)
        influence_array = np.array(influence_list)

        result = []
        # 10, 20, 30, 40, 50
        for num in [1, 10, 20, 30, 40, 50]:
            avg_influence = np.mean(influence_list[num-1])
            std_influence = np.std(influence_list[num-1])
            result.append((avg_influence, std_influence))

        seed_path = data_path + "seeds_greedy_" + str(pro)+ ".list"
        joblib.dump(seeds, seed_path)

        result_path = "./result/" + data + "_Greedy_" + + str(pro) + ".txt"
        np.savetxt(result_path, result, fmt='%.3f', delimiter='\t')


