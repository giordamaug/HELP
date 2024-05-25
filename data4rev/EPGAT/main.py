#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
from pprint import pprint
sys.path.append('.')

from utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import *
from gat import *

modelname = 'GAT'

def get_snapshot_name(name, expr_path, ortho_path, subloc_path, no_ppi, weights):
    snapshot_name = f'{name}'
    snapshot_name += f'_expr' if expr_path is not None else ''
    snapshot_name += f'_ortho' if ortho_path is not None else ''
    snapshot_name += f'_subl' if subloc_path is not None else ''
    snapshot_name += f'_ppi' if not no_ppi else ''
    snapshot_name += f'_w' if weights else ''
    return snapshot_name

def main(name, label_path, ppi_path=None,
         expr_path=None, ortho_path=None, subloc_path=None, no_ppi=False, 
         weights=False, seed=0, train_mode=False, n_epochs=1000, 
         savedir='.', predsavedir='.', verbose=False):

    set_seed(seed)

    snapshot_name = get_snapshot_name(name, expr_path, ortho_path, subloc_path, no_ppi, weights)

    savepath = os.path.join(savedir, snapshot_name)

    # Getting the data ----------------------------------
    (edge_index, edge_weights), X, (train_idx, train_y), \
        (val_idx, val_y), (test_idx, test_y), genes = data(label_path, ppi_path, expr_path, ortho_path, subloc_path, no_ppi=no_ppi, weights=weights, verbose=verbose)
    if verbose: print('Fetched data')

    # Train the model -----------------------------------
    if train_mode:
        if verbose: print('\nTraining the model')
        gat_params = gat_human
        model = train(gat_params, X, edge_index, edge_weights,
                        train_y, train_idx, val_y, val_idx, n_epochs=n_epochs, savepath=savepath)
    # ---------------------------------------------------

    # Load trained model --------------------------------
    if verbose: print(f'\nLoading the model from: {savepath}')
    snapshot = torch.load(savepath)
    model = GAT(in_feats=X.shape[1], **snapshot['model_params'])
    model.load_state_dict(snapshot['model_state_dict'])
    if verbose: print('Model loaded. Val AUC: {}'.format(snapshot['auc']))
    # ---------------------------------------------------

    # Test the model ------------------------------------
    preds, auc, score, ba, mcc, sens, specs = test(model, X, edge_index, (test_idx, test_y))
    preds = np.concatenate(
        [genes[test_idx].reshape((-1, 1)), preds[test_idx]], axis=1)
    save_preds(modelname, preds, predsavedir, snapshot_name, seed=seed)
    if verbose: 
        print('Test AUC:', auc)
        print('Test Accuracy:', score)
        print('Test BA:', ba)
        print('Test Sens.', sens)
        print('Test Sp.', specs)
        print('Test MCC:', mcc)
    # ---------------------------------------------------

    return preds, auc, score, ba, mcc, sens, specs

path = "../../data"
ipath = "./data"
n_runs = 10
n_epochs = 1000
seed=0
name ='brain'
label_path = os.path.join(path, 'Brain_HELP_2.csv')
ppi_path = os.path.join(path, 'Brain_PPI.csv')
expr_path=None #os.path.join(ipath, 'GTEX_expr_kidney.csv'),
ortho_path=None #os.path.join(ipath, 'Orthologs_kidney.csv'),
subloc_path=os.path.join(ipath, 'Sublocs_kidney.csv') 
no_ppi=False 
weights=False
train_mode = True
hypersearch = True
snapshot_name = get_snapshot_name(name, expr_path, ortho_path, subloc_path, no_ppi, weights)

if hypersearch:
    seed = np.random.randint(1000) + 10
    datasets = []
    for i in range(3):
        set_seed(seed+i)
        datasets += [data(label_path, ppi_path, expr_path, ortho_path, subloc_path, no_ppi=no_ppi, weights=weights, verbose=True) for i in range(3)]
    hyper_search(name, './studies', datasets)
elif n_runs:
    print(f'Training on {n_runs} runs')
    m = np.array([main(name, label_path, 
        ppi_path=ppi_path, expr_path=expr_path, ortho_path=ortho_path, subloc_path=subloc_path, 
        no_ppi=no_ppi, weights=weights, train_mode=train_mode, n_epochs=n_epochs,
        savedir='models', predsavedir='results',seed=i)[1:] for i in range(n_runs)])
    measures = np.ravel(np.column_stack((np.mean(m, axis=0),np.std(m, axis=0))))
    save_results(os.path.join('results', f'{modelname}_{snapshot_name}_r{n_runs}.csv'), *measures)
else:
    print('Training a single run with seed', seed)
    main(name, label_path, 
        ppi_path=ppi_path, expr_path=expr_path, ortho_path=ortho_path, subloc_path=subloc_path, 
        no_ppi=no_ppi, weights=weights, train_mode=train_mode, n_epochs=n_epochs,
        savedir='models', predsavedir='results',seed=seed)
