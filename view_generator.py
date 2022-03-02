import copy
import argparse
import random
import numpy as np
import os
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, subgraph

from IPython import embed

class GIN_Classifier(torch.nn.Module):
    def __init__(self, dataset, dim):
        super().__init__()

        num_features = dataset.num_features

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
    def forward_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x

class GIN_VEncoder(torch.nn.Module):
    def __init__(self, dataset, dim):
        super().__init__()

        num_features = dataset.num_features

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        nn_mu = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn_logstd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        
        self.conv_mu = GINConv(nn_mu)
        self.conv_logstd = GINConv(nn_logstd)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        # x = global_add_pool(x, batch)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd

class GIN_NodeWeightEncoder(torch.nn.Module):
    def __init__(self, dataset, dim, add_mask=False):
        super().__init__()

        num_features = dataset.num_features
        # num_features = dataset_num_features

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        # nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv2 = GINConv(nn2)
        # self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)

        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = None
        if add_mask == True:
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 3))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(3)
        else:
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 2))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(2)
    
    def forward(self, data):        
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        # x = F.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        return x

class ViewGenerator(VGAE):
    def __init__(self, dataset, dim, encoder, add_mask=False):
        self.add_mask = add_mask
        encoder = encoder(dataset, dim, self.add_mask)
        super().__init__(encoder=encoder)

    def sample_view(self, data):
        data = copy.deepcopy(data)
        edge_index = data.edge_index
        z = self.encode(data)
        # pre_recovered = self.decoder.forward_all(z) 
        # exp_num = pre_recovered.sum()
        
        recovered = self.decoder.forward_all(z)
        exp_num = recovered.sum()
        recovered = self.decoder.forward_all(z) * (data.num_edges / float(exp_num)) 
        edge_selected = torch.bernoulli(recovered)
        edge_selected = edge_selected.bool()
        
        edge_index = edge_selected.nonzero(as_tuple=False).T
        # print(edge_selected)
        edge_index = to_undirected(edge_index)
        edge_index = add_self_loops(edge_index)[0]
        data.edge_index = edge_index
        return z, recovered, data

    def sample_partial_view(self, data):
        data = copy.deepcopy(data)
        z = self.encode(data)
        edge_index = data.edge_index

        neg_edge_index = negative_sampling(edge_index)
        joint_edge_index = torch.cat((edge_index, neg_edge_index), dim=1)
        # joint_edge_index = to_undirected(joint_edge_index)
        joint_edge_index = remove_self_loops(joint_edge_index)[0]
        # joint_edge_index = add_self_loops(joint_edge_index)[0]

        wanted_num_edges = data.num_edges // 2
        edge_weights = self.decoder.forward(z, joint_edge_index)
        exp_num_edges = edge_weights.sum()
        edge_weights *=  wanted_num_edges / exp_num_edges
        
        edge_selected = torch.bernoulli(edge_weights)
        edge_selected = edge_selected.bool()

        edge_index = joint_edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]

        data.edge_index = edge_index
        return z, None, data
    
    def sample_partial_view_recon(self, data, neg_edge_index):
        data = copy.deepcopy(data)
        z = self.encode(data)
        # return z, None, None
        
        edge_index = data.edge_index
        
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(edge_index)
        
        joint_edge_index = torch.cat((edge_index, neg_edge_index), dim=1)
        # joint_edge_index = edge_index
        # joint_edge_index = to_undirected(joint_edge_index)

        # wanted_num_edges = data.num_edges // 2
        edge_weights = self.decoder.forward(z, joint_edge_index)
        edge_selected = torch.bernoulli(edge_weights)
        edge_selected = edge_selected.bool()

        edge_index = joint_edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)

        # edge_index = add_self_loops(edge_index)[0]
        # print("final edges:", edge_index.shape[1])
        data.edge_index = edge_index
        return z, neg_edge_index, data

    def sample_subgraph_view(self, data):
        data = copy.deepcopy(data)
        z = self.encode(data)
        edge_index = data.edge_index

        recovered_all = self.decoder.forward_all(z)
        recovered = self.decode(z, edge_index)
        edge_selected = torch.bernoulli(recovered)
        edge_selected = edge_selected.bool()
        edge_index = edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)

        edge_index = add_self_loops(edge_index, num_nodes = data.num_nodes)[0]

        data.edge_index = edge_index
        return z, recovered_all, data
    
    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad
        
        p = self.encoder(data)
        sample = F.gumbel_softmax(p, hard=True)

        real_sample = sample[:,0]
        attr_mask_sample = None
        if self.add_mask == True:
            attr_mask_sample = sample[:,2]
            keep_sample = real_sample + attr_mask_sample
        else:
            keep_sample = real_sample
        
        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1,)
        edge_index, edge_attr = subgraph(keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        if self.add_mask == True:
            attr_mask_idx = attr_mask_sample.bool()
            token = data.x.detach().mean()
            x[attr_mask_idx] = token

        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr
        
        return keep_sample, data

def get_adj(data):
    data = copy.deepcopy(data)
    edge_index = data.edge_index.cpu().detach()
    adj = torch.zeros(data.num_nodes, data.num_nodes)
    adj_all_edge_index = add_self_loops(edge_index)[0]
    adj[adj_all_edge_index[0], adj_all_edge_index[1]] = 1
    return adj

def set_seed(seed):
    args.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dataset', type=str, default='', help='batch size')

    args = parser.parse_args("")
    return args

