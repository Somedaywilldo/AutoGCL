import argparse
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import time
import shutil
import glob
import logging
import sys

from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from IPython import embed

from chem_splitters import scaffold_split, random_split, random_scaffold_split
from chem_model import GNN_CL
from chem_loader import MoleculeDataset_aug
from chem_loader import MoleculeDataset

from IPython import embed

# embed()

sys.path.append(os.path.abspath(os.path.join('..')))
from view_generator import ViewGenerator, GIN_NodeWeightEncoder

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset_root', type=str, default = 'dataset', help='root directory of dataset')
    parser.add_argument('--dataset', type=str, default = 'chembl_filtered', help='root directory of dataset. For now, only classification.')
    # chembl_filtered
    parser.add_argument('--exp', type=str, default = 'chem', help='')
    parser.add_argument('--save', type=str, default = '', help='')

    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)        
        os.mkdir(os.path.join(path, 'model'))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class graphcl(nn.Module):

    def __init__(self, gnn, dataset):
        super().__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        # self.cls_head = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(300, dataset.num_classes))
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.cls_head = nn.Sequential(nn.Linear(300, 1))

    def forward_cl(self, data, sample):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(data, sample)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        x = self.cls_head(x)
        return x

    # def loss_cl(self, x1, x2):
    #     T = 0.1
    #     batch_size, _ = x1.size()
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = - torch.log(loss).mean()
    #     return loss

def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss

def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        view1, view2 = batch
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(view1.x, view1.edge_index, view1.edge_attr, view1.batch)
        x2 = model.forward_cl(view2.x, view2.edge_index, view2.edge_attr, view2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())
        # print()

    return train_acc_accum/(step+1), train_loss_accum/(step+1)

def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def train_node_view_cl(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device, args, logger):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0
    total_graphs = 0

    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    with tqdm(data_loader) as t:
        for data in data_loader:
            optimizer.zero_grad()
            view_optimizer.zero_grad()

            data = data.to(device)        

            sample1, view1 = view_gen1(data, True)
            sample2, view2 = view_gen2(data, True)

            out1 = model.forward_cl(view1, sample1)
            out2 = model.forward_cl(view2, sample2)

            loss = loss_cl(out1, out2)
            loss.backward()

            optimizer.step()
            view_optimizer.step()

            train_loss_accum += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

            batch_loss = loss.detach().cpu().item()
            postfix_str = 'batch_loss: {:.04f}'.format(batch_loss)
            t.set_postfix_str(postfix_str)
            t.update()

    logger.info("Exp Dir: {}".format(args.save) )
    return train_loss_accum / total_graphs

def train_node_view_cl_with_sim(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device, args, logger):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0
    total_graphs = 0
    
    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    with tqdm(data_loader) as t:
        for data in data_loader:
            optimizer.zero_grad()
            view_optimizer.zero_grad()

            data = data.to(device)        

            sample1, view1 = view_gen1(data, True)
            sample2, view2 = view_gen2(data, True)

            sim_loss = F.mse_loss(sample1, sample2)
            sim_loss = (1 - sim_loss)

            sample = torch.ones(sample1.shape).to(device)
            # embed()
            # exit()

            input_pair_list = [(sample, data), (sample1, view1), (sample2, view2)]
            # input_list = [view1, view2]
            input_pair1, input_pair2 = random.choices(input_pair_list, k=2)

            out1 = model.forward_cl(input_pair1[1], input_pair1[0])
            out2 = model.forward_cl(input_pair2[1], input_pair2[0])

            cl_loss = loss_cl(out1, out2)
            
            loss = cl_loss + sim_loss
            loss.backward()
            # embed()
            # exit()
            optimizer.step()
            view_optimizer.step()

            train_loss_accum += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

            batch_loss = loss.detach().cpu().item()
            # postfix_str = 'batch_loss: {:.04f}'.format(batch_loss)
            postfix_str = 'sim_loss: {:.04f}, cl_loss: {:.04f}'.format(sim_loss, cl_loss)

            t.set_postfix_str(postfix_str)
            t.update()

    logger.info("Exp Dir: {}".format(args.save) )
    return train_loss_accum / total_graphs


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    args.save = '{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('transfer_exp', args.exp, args.save)
    create_exp_dir(args.save, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #set up dataset
    dataset = MoleculeDataset(os.path.join(args.dataset_root, args.dataset), dataset=args.dataset)
    logger.info(dataset)
    
    dataset = dataset.shuffle()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=True)

    #set up model
    gnn = GNN_CL(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    # embed()
    # exit()

    model = graphcl(gnn, dataset)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    logger.info(optimizer)

    view_gen1 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
    view_gen2 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
    view_gen1 = view_gen1.to(device)
    view_gen2 = view_gen2.to(device)

    view_optimizer = optim.Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=args.lr
                                , weight_decay=args.decay)


    for epoch in range(1, args.epochs+1):    
        # train_acc, train_loss = train(args, model, device, dataset, optimizer)
        # train_loss = train_node_view_cl(view_gen1, view_gen2, view_optimizer, 
        #                                 model, optimizer, train_loader, device, args, logger)

        train_loss = train_node_view_cl_with_sim(view_gen1, view_gen2, view_optimizer, 
                                        model, optimizer, train_loader, device, args, logger)

        logger.info('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))

        if epoch % 5 == 0:
            model_name = "cl_model_{}.pth".format(epoch)
            model_path = os.path.join(args.save, 'model', model_name)
            torch.save(gnn.state_dict(), model_path)

            model_name = "cl_view_gen1_{}.pth".format(epoch)
            model_path = os.path.join(args.save, 'model', model_name)
            torch.save(view_gen1.state_dict(), model_path)

            model_name = "cl_view_gen2_{}.pth".format(epoch)
            model_path = os.path.join(args.save, 'model', model_name)
            torch.save(view_gen2.state_dict(), model_path)
 