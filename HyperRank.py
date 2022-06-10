# -*- coding: utf-8 -*-
"""

"""

from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle, joblib
from functools import reduce
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix

def load_graph(datapath):
    graph = nx.read_gpickle(datapath)
    return graph


def load_subsets(datapath):
    with open(datapath, 'rb') as f:
        hyperedge_dict = pickle.load(f)
    return hyperedge_dict


def hyperedge_weights(H):
    num_hyperedge = H.shape[1]
    print('number of hyperedge is ', num_hyperedge)
    de = H.sum(axis = 0)
    de_inverse = 1/de
    offsets = np.array([0])
    D_e_inverse = dia_matrix((de_inverse, offsets), shape=(num_hyperedge,num_hyperedge)).tocsr()
    return D_e_inverse


def node_weights(H):
    num_node = H.shape[0]
    dv = H.sum(axis=1).reshape(1,-1) # [1,num node * 1]
    dv_inverse = 1/dv
    offsets = np.array([0])
    D_v_inverse = dia_matrix((dv_inverse, offsets), shape=(num_node, num_node)).tocsr()
    return D_v_inverse




def transition_matrix(edge_list):
    H, H_T = make_sparse_H(edge_list)
    D_e_inverse = hyperedge_weights(H)
    D_v_inverse = node_weights(H)

    temp = D_e_inverse.dot(H_T)
    temp1 = temp.dot(D_v_inverse)
    P = temp1.dot(H)
    return P.transpose()


def pagerank(edge_list, ntimes):
    P_T = transition_matrix(edge_list)
    num_hedge = len(edge_list)
    v = np.ones(num_hedge, dtype=float).reshape(-1, 1) / num_hedge
    e = np.ones(num_hedge, dtype=float).reshape(-1, 1) / num_hedge
    alpha = 0.85
    for i in tqdm(range(ntimes)):
        v = alpha * P_T.dot(v) + (1 - alpha) * e
    return v


def reindex(subset_dict):


    items = subset_dict.items()
    nodes = [a[1] for a in items]
    edges = [a[0] for a in items]

    unique_nodes = sum(nodes, []) 
    unique_nodes = list(set(unique_nodes))
    num_node = len(unique_nodes)
    node2index = dict(zip(unique_nodes, list(np.arange(num_node))))  # {node:index}
    index2node = dict(zip(list(np.arange(num_node)), unique_nodes)) # {index:node}


    unique_edges = list(set(edges))
    assert len(unique_edges) == len(edges) 
    reindexed_edge_list = []
    edge2index = {}
    index2edge = {}
    cnt_edge = 0
    for j in range(len(edges)):
        eindex = cnt_edge
        index2edge[eindex] = edges[j]
        cnt_edge += 1
        nindex = [node2index[a] for a in nodes[j]]
        reindexed_edge_list.append((eindex, nindex))
    return  reindexed_edge_list, index2node, index2edge



def make_sparse_H(edge_list):
    size_col = len(edge_list)
    nodes = [a[1] for a in edge_list]
    nodes = sum(nodes, []) 
    size_row = len(set(nodes))
    print(size_row, size_col)

    indptr = [0] 
    indices = []  
    total_item = 0
    data = []
    for (edge, nodes) in edge_list:
        num_nodes = len(nodes)
        total_item += num_nodes
        indptr.append(total_item)
        data.extend(list(np.ones(num_nodes)))
        indices.extend(nodes)

    assert len(data) == indptr[-1]
    assert len(data) == len(indices)
    csr = csr_matrix((data, indices, indptr), shape=(size_col, size_row))


    return csr.transpose(),csr 

if __name__ == '__main__':

    st = time()

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=str, default="email")

    args = parser.parse_args()
    data = args.data
    data_path = "./data/" + data + "/"

    subset_path = data_path + "subset.dict"
    subset_dict = joblib.load(subset_path)

    reindexed_edge_list, index2node, index2edge = reindex(subset_dict) 
    ntimes = 1000
    hyperedge_importance = pagerank(reindexed_edge_list, ntimes).squeeze()
    hyperedge_rank = np.argsort(-1 * hyperedge_importance)
    seeds = hyperedge_rank[:50]
    real_hyperedge = [index2edge[a] for a in seeds]

    seed_path = data_path + "seeds_pagerank.list"
    joblib.dump(real_hyperedge, seed_path)

    et = time() - st

    print("Time of HyperRank: ", et)


