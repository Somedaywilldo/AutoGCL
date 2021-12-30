import torch 
import argparse
import random

from torch_geometric.utils import dropout_adj, degree, to_undirected
from torch_geometric.nn import global_add_pool
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted
from pGRACE.functional import degree_drop_weights
from pGRACE.functional import evc_drop_weights, pr_drop_weights
from pGRACE.functional import feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.utils import get_base_model, get_activation
from pGRACE.utils import compute_pr, eigenvector_centrality

def get_drop_and_feature_weights(data, param, device, gca_args):
    drop_weights = None
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if gca_args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if gca_args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if gca_args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)
    
    return drop_weights, feature_weights

def get_gca_model(dataset, param, gca_args):
    torch_seed = gca_args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    device = torch.device(gca_args.device)
    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    return model, encoder

def get_gca_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_proj_hidden', type=int, default=32)
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--base_model', type=str, default='GCNConv')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.1)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--drop_scheme', type=str, default='degree')

    gca_args = parser.parse_args("")

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': gca_args.num_hidden,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # # add hyper-parameters into parser
    param_keys = default_param.keys()
    # for key in param_keys:
    #     parser.add_argument(f'--{key}', type=type(default_param[key]), ngca_args='?')
    

    # parse param
    # sp = SimpleParam(default=default_param)
    # param = sp(source=gca_args.param)
    param = default_param

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(gca_args, key) is not None:
            param[key] = getattr(gca_args, key)

    # embed()
    # exit()
    
    return gca_args, param

def train_gca(data_loader, param, model, optimizer, device, gca_args):
    loss_all = 0
    total_graphs = 0

    for data in data_loader:
        data = data.to(device)

        drop_weights, feature_weights = get_drop_and_feature_weights(data, param, device, gca_args)

        model.train()
        optimizer.zero_grad()

        def drop_edge(idx: int):
            # global drop_weights
            if param['drop_scheme'] == 'uniform':
                return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
            elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        # embed()
        loss = model.loss(z1, z2, batch_size=1024 if gca_args.dataset == 'Coauthor-Phy' else None)
        
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        # print("loss:", loss.item())
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

class GCA_Classifier(torch.nn.Module):
    def __init__(self, dataset, hidden):
        super().__init__()
        self.fc1 = Linear(hidden, hidden)
        self.fc2 = Linear(hidden, dataset.num_classes)

    def forward(self, x, batch):
        out = global_add_pool(x, batch)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=-1)
        return out

def train_gca_cls(data_loader, model, gca_cls, gca_cls_optimizer, device):
    loss_all = 0
    total_graphs = 0

    model.train()
    gca_cls.train()

    for data in data_loader:
        gca_cls_optimizer.zero_grad()

        data = data.to(device)
        z = model(data.x, data.edge_index)
        output = gca_cls(z, data.batch)

        loss = F.nll_loss(output, data.y)
        loss.backward()
        # print("cls loss:", loss.item())
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        gca_cls_optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

@torch.no_grad()
def eval_gca_acc(model, gca_cls, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        gca_cls.eval()
    
    correct = 0
    for data in loader:
        data = data.to(device)
        z = model(data.x, data.edge_index)
        output = gca_cls(z, data.batch)

        with torch.no_grad():
            pred = output.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)
