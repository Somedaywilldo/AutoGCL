import numpy as np
from IPython import embed
import copy
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx

import torch
import torch.nn as nn

from torch_geometric.utils import subgraph, k_hop_subgraph

def NodeDrop(data, aug_ratio):
    data = copy.deepcopy(data)
    x = data.x
    edge_index = data.edge_index
    
    drop_num = int(data.num_nodes * aug_ratio)
    keep_num = data.num_nodes - drop_num
    
    keep_idx = torch.randperm(data.num_nodes)[:keep_num]    
    edge_index, _ = subgraph(keep_idx, edge_index)

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[keep_idx] = False
    x[drop_idx] = 0
    
    data.x = x
    data.edge_index = edge_index
    return data

def EdgePerturb(data, aug_ratio):
    data = copy.deepcopy(data)
    
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index

    unif = torch.ones(2, node_num)
    add_edge_idx = unif.multinomial(permute_num, replacement=True).to(data.x.device)

    unif = torch.ones(edge_num)
    keep_edge_idx = unif.multinomial((edge_num - permute_num), replacement=True)

    edge_index = torch.cat((edge_index[:, keep_edge_idx], add_edge_idx), dim=1)
    data.edge_index = edge_index
    return data

# same function as GraphCL but in torch_geometric
def Subgraph(data, aug_ratio):
    data = copy.deepcopy(data)
    
    # return data
    x = data.x
    edge_index = data.edge_index

    sub_num = int(data.num_nodes * aug_ratio)
    idx_sub = torch.randint(0, data.num_nodes, (1, )).to(edge_index.device)
    last_idx = idx_sub

    keep_idx = None
    diff = None

    # print("sub_num:", sub_num)
    for k in range(1, sub_num):
        keep_idx, _, _, _ = k_hop_subgraph(last_idx, 1, edge_index)
        # print("subgraph: {}, keep_idx size: {}".format(k, keep_idx.shape[0]) )
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = torch.cat((last_idx, keep_idx)).to(edge_index.device)
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = torch.randperm(diff.shape[0])[:diff_keep_num].to(edge_index.device)
    final_keep_idx = torch.cat((last_idx, diff_keep_idx))
    
    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0
        
    edge_index, _ = subgraph(final_keep_idx, edge_index)
    
    data.x = x
    data.edge_index = edge_index
    return data

def AttrMask(data, aug_ratio):
    data = copy.deepcopy(data)
    
    mask_num = int(data.num_nodes * aug_ratio)
    unif = torch.ones(data.num_nodes)
    mask_idx = unif.multinomial(mask_num, replacement=True)

    token = data.x.mean(dim=0)
    data.x[mask_idx] = token
    return data

class Augmentor(nn.Module):
    def __init__(self, aug_ratio, preset=-1):
        super().__init__()
        self.aug_ratio = aug_ratio
        self.aug = preset
    
    def forward(self, data):
        # if self.aug == -1:
        self.aug = np.random.randint(4)
        # self.aug = 2
        # self.aug = 3
        # ri = 0
        # ri = 3
        # self.aug = 0
        if self.aug == 0:
            # print("node drop")
            data = NodeDrop(data, self.aug_ratio)
        elif self.aug == 1:
            # print("subgraph")
            data = Subgraph(data, self.aug_ratio)
        elif self.aug == 2:
            # print("edge perturb")
            data = EdgePerturb(data, self.aug_ratio)
        elif self.aug == 3:
            # print("attr mask")
            data = AttrMask(data, self.aug_ratio)
        else:
            print('sample augmentation error')
            assert False
        return data


'''
# you can make use of this function if you want to visualize the augmented graph
def vis_graph(data, view1, view2):
    # fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
    plt.figure(figsize=(16, 4))
    
    edge_index = data.edge_index.detach().cpu().numpy()
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')
    
    plt.subplot(1, 3, 1)
    pos = nx.fruchterman_reingold_layout(G)
    
    nx.draw(G, with_labels=True, node_size=100, node_color="skyblue", pos=pos)
    plt.title("Original")
    plt.subplot(1, 3, 2)
    edge_index = view1.edge_index.detach().cpu().numpy()
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')
    # pos = nx.fruchterman_reingold_layout(G)
    nx.draw(G, with_labels=True, node_size=100, node_color="skyblue", pos=pos)
    plt.title("View1")
    
    plt.subplot(1, 3, 3)
    edge_index = view2.edge_index.detach().cpu().numpy()
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')
    # pos = nx.fruchterman_reingold_layout(G)
    nx.draw(G, with_labels=True, node_size=100, node_color="skyblue", pos=pos)
    plt.title("View2")
'''