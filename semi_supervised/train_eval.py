import logging
import sys
import time
import copy
from sklearn.model_selection import StratifiedKFold
import random
import argparse
import os

import torch
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from gca import get_gca_param, get_gca_model, train_gca
from gca import GCA_Classifier, train_gca_cls, eval_gca_acc

sys.path.append(os.path.abspath(os.path.join('..')))
from view_generator import ViewGenerator, GIN_NodeWeightEncoder, GIN_Classifier
from augs import Augmentor
from utils import print_weights

from IPython import embed

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

def get_snapshot(view_gen1, view_gen2, model):
    snapshot = {
        'view_gen1': copy.deepcopy(view_gen1.state_dict()),
        'view_gen2': copy.deepcopy(view_gen2.state_dict()),
        'model': copy.deepcopy(model.state_dict())
    }
    return snapshot

def load_snapshot(snapshot, view_gen1, view_gen2, model):
    view_gen1.load_state_dict(snapshot['view_gen1'])
    view_gen2.load_state_dict(snapshot['view_gen2'])
    model.load_state_dict(snapshot['model'])

def benchmark_exp(device, logger, dataset, model_func, 
                 folds, epochs, batch_size,
                 lr, lr_decay_factor, lr_decay_step_size, weight_decay, 
                 epoch_select, with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        semi_dataset = dataset[semi_idx]
        # val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)
        
        # # Change to GIN model
        # model = GIN_Classifier(dataset, 128)
        # model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Pre-training Classifier...")
        # best_model = None
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, 
                                                                train_acc, test_acc))

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                    # logger.info("lr:" + str(param_group['lr']))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def aug_only_exp(device, logger, dataset, model_func, 
                 folds, epochs, batch_size,
                 lr, lr_decay_factor, lr_decay_step_size, weight_decay, 
                 epoch_select, with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        view_gen1 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen2 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        view_optimizer = Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=lr
                                , weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Direct Augmentation...")
        best_test_acc = 0
        # best_model = None

        for epoch in range(1, epochs + 1):
            train_loss, sim_loss, cls_loss = train_cls_with_node_weight_view_gen(view_gen1, view_gen2, view_optimizer, model, optimizer, semi_loader, device)
            
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Sim Loss: {:.4f}, Cls Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, 
                                                                sim_loss, cls_loss,
                                                                train_acc, test_acc))
            
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                    # logger.info("lr:" + str(param_group['lr']))
        
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def cl_gca_exp(device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    gca_args, param = get_gca_param()

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model, encoder = get_gca_model(dataset, param, gca_args)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )

        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None

        for epoch in range(1, 30 + 1):
            cl_loss = train_gca(train_loader, param, model, optimizer, device, gca_args)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, cl_loss))

        gca_cls = GCA_Classifier(dataset, param["num_hidden"])
        gca_cls.to(device)
        gca_cls_optimizer = Adam([
                                    {"params": gca_cls.parameters()},
                                    {"params": model.parameters()}
                                ], lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            cls_loss = train_gca_cls(semi_loader, model, gca_cls, gca_cls_optimizer, device)
            
            train_acc = eval_gca_acc(model, gca_cls, semi_loader, device, with_eval_mode)
            test_acc = eval_gca_acc(model, gca_cls, test_loader, device, with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            logger.info('Epoch: {:03d}, Cls Loss: {:.4f}, '
                        'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, cls_loss, 
                                                                    train_acc, test_acc))
            
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))

    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

# GraphCL Reproduced
def graph_cl_exp(
            device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None, aug_ratio=0.2):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        augmentor = Augmentor(aug_ratio)
        
        logger.info("*" * 50)
        logger.info("Training Contrastive learning...")
        for epoch in range(1, epochs + 1):
            train_loss = train_graph_cl(augmentor, model, optimizer, train_loader, device)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, train_loss))
        
        logger.info("*" * 50)
        logger.info("Pretraining Classifier...")
        best_model = None
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train_cls(model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, 
                                                                train_acc, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        best_model = None

        logger.info("*" * 50)
        logger.info("Fold: {}, Best Test Acc: {:.4f}".format(fold, best_test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
    
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    # logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    
    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

# GraphCL reproduced with only the augmentations
def graph_cl_aug_only_exp(
            device,
            logger,
            dataset,
            model_func,
            folds,
            epochs,
            batch_size,
            lr,
            lr_decay_factor,
            lr_decay_step_size,
            weight_decay,
            epoch_select,
            with_eval_mode=True,
            semi_split=None,
            aug_ratio=0.2):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    # print("developing CL...")
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        # train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        t_start = time.perf_counter()

        augmentor = Augmentor(aug_ratio)
        
        logger.info("*" * 50)
        logger.info("Training Classifier with GraphCL Augs...")
        best_model = None
        best_test_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = train_graph_cl_aug_semi(augmentor, model, optimizer, semi_loader, device)
            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            logger.info('Epoch: {:03d}, Train Loss: {:.4f}, '
                    'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss, 
                                                                train_acc, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        best_model = None

        logger.info("*" * 50)
        logger.info("Fold: {}, Best Test Acc: {:.4f}".format(fold, best_test_acc))

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    # logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def naive_cl_exp(device, logger, dataset, model_func, folds, epochs, batch_size,
            lr, lr_decay_factor, lr_decay_step_size, weight_decay, epoch_select,
            with_eval_mode=True, semi_split=None):
    
    assert epoch_select in ['val_max', 'test_max'], epoch_select
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        view_gen1 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen2 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        view_optimizer = Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=lr
                                , weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None

        for epoch in range(1, epochs + 1):
            cl_loss = train_cl_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model, 
                                                    optimizer, train_loader, device)
            logger.info('Epoch: {:03d}, CL Loss: {:.4f}'.format(epoch, cl_loss))
                                                
        for epoch in range(1, epochs + 1):
            train_view_loss, sim_loss, cls_loss = train_node_weight_view_gen_and_cls(
                                                    view_gen1, view_gen2, 
                                                    view_optimizer,
                                                    model, optimizer,
                                                    semi_loader, device)

            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Sim Loss: {:.4f}, '
                    'Cls Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_view_loss, sim_loss,
                                                                cls_loss, train_acc, test_acc))
            
            if epoch % lr_decay_step_size == 0:
                for param_group in view_optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))

    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def joint_cl_exp(
            device,
            logger,
            dataset,
            model_func,
            folds,
            epochs,
            batch_size,
            lr,
            lr_decay_factor,
            lr_decay_step_size,
            weight_decay,
            epoch_select,
            with_eval_mode=True,
            semi_split=None,
            add_mask=False):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    # print("developing CL...")
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx, semi_idx) in enumerate(
            zip(*cl_k_fold(dataset, folds, epoch_select, semi_split))):
        
        logger.info("*" * 10)
        logger.info("Fold: %d" % fold)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]
        semi_dataset = dataset[semi_idx]

        semi_loader = DataLoader(semi_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size * 4, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        logger.info("Train size: %d" % len(train_dataset))
        logger.info("Semi size: %d" % len(semi_dataset))
        # logger.info("Val size: %d" % len(val_dataset))
        logger.info("Test size: %d" % len(test_dataset))

        model = model_func(dataset)
        model.to(device)

        view_gen1 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder, add_mask)
        view_gen2 = ViewGenerator(dataset, 128, GIN_NodeWeightEncoder, add_mask)
        view_gen1 = view_gen1.to(device)
        view_gen2 = view_gen2.to(device)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        view_optimizer = Adam([ {'params': view_gen1.parameters()},
                                {'params': view_gen2.parameters()} ], lr=lr
                                , weight_decay=weight_decay)

        t_start = time.perf_counter()

        logger.info("*" * 50)
        logger.info("Training View Generator and Classifier...")
        best_test_acc = 0
        after_train_view_snapshot = None
                                    
        for epoch in range(1, epochs + 1):
            train_view_loss, sim_loss, cls_loss, cl_loss = train_node_weight_view_gen_and_cls(
                                                    view_gen1, view_gen2, 
                                                    view_optimizer,
                                                    model, optimizer,
                                                    semi_loader, device)

            train_acc = eval_acc(model, semi_loader, device, with_eval_mode)
            test_acc = eval_acc(model, test_loader, device, with_eval_mode)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            logger.info('Epoch: {:03d}, Train View Loss: {:.4f}, Sim Loss: {:.4f}, '
                    'Cls Loss: {:.4f}, CL Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
                                                                epoch, train_view_loss, sim_loss,
                                                                cls_loss, cl_loss, train_acc, test_acc))
            
            if epoch % lr_decay_step_size == 0:
                for param_group in view_optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
    
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # logger.info("All Test Acc: {}".format(test_acc))

    best_epoch = test_acc.max(dim=1)[1]
    test_acc = test_acc.max(dim=1)[0]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    logger.info("Test Acc: {}".format(test_acc))
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean

def train_node_weight_view_gen_and_cls(view_gen1, view_gen2, view_optimizer, 
                                        model, optimizer, loader, device):
    view_gen1.train()
    view_gen2.train()
    model.train()

    loss_all = 0
    sim_loss_all = 0
    cls_loss_all = 0
    cl_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        
        data = data.to(device)
        # output = model(data)

        sample1, view1 = view_gen1(data, True)
        sample2, view2 = view_gen2(data, True)

        sim_loss = F.mse_loss(sample1, sample2)
        sim_loss = (1 - sim_loss)

        output = model(data)
        output1 = model(view1)
        output2 = model(view2)        

        loss0 = F.nll_loss(output, data.y)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)

        cls_loss = (loss0 + loss1 + loss2) / 3

        cl_loss = loss_cl(output1, output2)

        loss = sim_loss + cls_loss + cl_loss
        loss.backward()

        loss_all += loss.item() * data.num_graphs
        sim_loss_all += sim_loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        cl_loss_all += cl_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
    
    loss_all /= total_graphs
    sim_loss_all /= total_graphs
    cls_loss_all /= total_graphs
    cl_loss_all /= total_graphs

    return loss_all, sim_loss_all, cls_loss_all, cl_loss_all

def train_cl_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()
        
        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        input_list = [data, view1, view2]
        # input_list = [view1, view2]
        input1, input2 = random.choices(input_list, k=2)

        # embed()
        # exit()
        
        output1 = model.forward_cl(input1)
        output2 = model.forward_cl(input2)

        cl_loss = loss_cl(output1, output2)

        loss = cl_loss 
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs

    return loss_all

def train_graph_cl(augmentor, model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()    
        
        data = data.to(device)
        view1 = augmentor(data)
        view2 = augmentor(data)

        output1 = model.forward_cl(view1)
        output2 = model.forward_cl(view2)

        cl_loss = loss_cl(output1, output2)
        loss = cl_loss
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

def train_graph_cl_aug_semi(augmentor, model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()    
        data = data.to(device)
        view = augmentor(data)
        output = model(view)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

def train_cl(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()    
        
        data = data.to(device)
        # z1, rec1, view1 = view_gen1(data)
        # z2, rec2, view2 = view_gen2(data)

        # output1 = model.forward_cl(view1)
        # output2 = model.forward_cl(view2)
        # raw cl
        output1 = model.forward_cl(data)
        output2 = output1

        cl_loss = loss_cl(output1, output2)
        loss = cl_loss
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

def train_cls_with_fix_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    cls_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()
        
        data = data.to(device)
        _, _, view1 = view_gen1(data)
        _, _, view2 = view_gen2(data)

        output1 = model(view1)
        output2 = model(view2)

        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        
        cls_loss = (loss1 + loss2) / 2
        loss = cls_loss 
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all

def train_cls_with_node_weight_view_gen(view_gen1, view_gen2, view_optimizer, model, optimizer, loader, device):
    view_gen1.train()
    view_gen2.train()
    model.train()

    loss_all = 0
    sim_loss_all = 0
    cls_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        view_optimizer.zero_grad()
        optimizer.zero_grad()
        data = data.to(device)
        # output = model(data)

        sample1, view1 = view_gen1(data, True)
        sample2, view2 = view_gen2(data, True)

        sim_loss = F.l1_loss(sample1, sample2)
        sim_loss = torch.exp(1 - sim_loss)
        
        # output = model(data)
        output1 = model(view1)
        output2 = model(view2)        

        output = model(data)
        loss0 = F.nll_loss(output, data.y)

        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        # loss1 = F.mse_loss(output1, output)
        # loss2 = F.mse_loss(output2, output)

        cls_loss = (loss0 + loss1 + loss2) / 3
        # cls_loss = sim_loss
        # loss = cls_loss 
        loss = sim_loss + cls_loss
        loss.backward()

        # embed()
        # exit()
        loss_all += loss.item() * data.num_graphs
        sim_loss_all += sim_loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
        optimizer.step()
    
    loss_all /= total_graphs
    sim_loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all, sim_loss_all, cls_loss_all
    # return loss_all, rec_loss_all, cls_loss_all

def train_node_weight_view_gen_with_fix_cls(view_gen1, view_gen2, view_optimizer, model, loader, device):
    view_gen1.train()
    view_gen2.train()
    model.eval()
    # model.train()

    loss_all = 0
    sim_loss_all = 0
    cls_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        view_optimizer.zero_grad()
        data = data.to(device)
        # output = model(data)

        sample1, view1 = view_gen1(data, True)
        sample2, view2 = view_gen2(data, True)

        sim_loss = F.l1_loss(sample1, sample2)
        sim_loss = torch.exp(1 - sim_loss)
        
        # output = model(data)
        output1 = model(view1)
        output2 = model(view2)        

        # loss1 = F.nll_loss(output, data.y)
        # output = model(data)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        # loss1 = F.mse_loss(output1, output)
        # loss2 = F.mse_loss(output2, output)

        cls_loss = (loss1 + loss2) / 2
        # cls_loss = sim_loss
        
        # loss = cls_loss 
        loss = sim_loss + cls_loss
        loss.backward()

        # embed()
        # exit()
        
        loss_all += loss.item() * data.num_graphs
        sim_loss_all += sim_loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs

        total_graphs += data.num_graphs
        view_optimizer.step()
    
    loss_all /= total_graphs
    sim_loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all, sim_loss_all, cls_loss_all
    # return loss_all, rec_loss_all, cls_loss_all

def train_cls_with_fix_node_weight_view_gen(view_gen1, view_gen2, model, optimizer, loader, device):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    cls_loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()
        
        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        output1 = model(view1)
        output2 = model(view2)

        # output = model(data)
        loss1 = F.nll_loss(output1, data.y)
        loss2 = F.nll_loss(output2, data.y)
        
        # loss1 = F.mse_loss(output1, output)
        # loss2 = F.mse_loss(output2, output)

        cls_loss = (loss1 + loss2) / 2
        # cls_loss = loss1

        loss = cls_loss 
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        cls_loss_all += cls_loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    cls_loss_all /= total_graphs

    return loss_all

def cl_k_fold(dataset, folds, epoch_select, semi_split):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []

    semi_indices = []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    dataset_size = len(dataset)
    semi_size = int(dataset_size * semi_split / 100)

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indice = torch.nonzero(train_mask, as_tuple=False).view(-1)
        train_indices.append(train_indice)
   
        # semi split
        train_size = train_indice.shape[0]
        select_idx = torch.randperm(train_size)[:semi_size]
        semi_indice = train_indice[select_idx]
        semi_indices.append(semi_indice)
    # embed()
    # exit()
    return train_indices, test_indices, val_indices, semi_indices

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        # embed()
        # exit()
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train_cls(model, optimizer, loader, device):
    model.train()

    loss_all = 0
    total_graphs = 0
    
    for data in loader:
        optimizer.zero_grad()    
        data = data.to(device)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
        
        optimizer.step()
    
    loss_all /= total_graphs
    return loss_all

@torch.no_grad()
def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def eval_acc_with_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        _, _, view1 = view_gen1(data)
        _, _, view2 = view_gen2(data)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()

        correct += (correct1 + correct2) / 2

    return correct / len(loader.dataset)

@torch.no_grad()
def eval_acc_with_node_weight_view_gen(view_gen1, view_gen2, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
        view_gen1.eval()
        view_gen2.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        _, view1 = view_gen1(data, False)
        _, view2 = view_gen2(data, False)

        with torch.no_grad():
            pred1 = model(view1).max(1)[1]
            pred2 = model(view2).max(1)[1]

        correct1 = pred1.eq(data.y.view(-1)).sum().item()
        correct2 = pred2.eq(data.y.view(-1)).sum().item()

        correct += (correct1 + correct2) / 2

    return correct / len(loader.dataset)

def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
