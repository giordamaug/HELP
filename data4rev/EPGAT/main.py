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
import argparse

parser = argparse.ArgumentParser(description='PLOS COMPBIO EPGAT')
parser.add_argument('-n', "--name", dest='name', metavar='<name>', type=str, help='name of experiment', required=True)
parser.add_argument('-l', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename', required=True)
parser.add_argument('-e', "--exprfile", dest='exprfile', metavar='<exprfile>', type=str, default=None, help='expression filename', required=False)
parser.add_argument('-o', "--orthofile", dest='orthofile', metavar='<orthofile>', type=str, default=None, help='ortho filename', required=False)
parser.add_argument('-s', "--sublocfile", dest='sublocfile', metavar='<sublocfile>', type=str, default=None, help='sublocalization filename', required=False)
parser.add_argument('-p', "--ppifile", dest='ppifile', metavar='<ppifile>', type=str, default=None, help='ppifile filename', required=False)
parser.add_argument('-np', "--noppi", action='store_true', default=False, help='disable PPI usage', required=False)
parser.add_argument('-w', "--weights", action='store_true', default=False, help='use weights in PPI', required=False)
parser.add_argument('-v', "--verbose", action='store_true', default=False, help='enable verbosity', required=False)
parser.add_argument('-hy', "--hypersearch", action='store_true', default=False, help='enable optuna hyper-search', required=False)
parser.add_argument('-t', "--trainmode", action='store_true', default=False, help='enable training mode', required=False)
parser.add_argument('-r', "--nruns", dest='nruns', metavar='<nruns>', type=int, help='n. of runs in epxeriments (default: 10)' , default=10, required=False)
parser.add_argument('-m', "--measure", dest='measure', metavar='<measure>', type=str, help='measure for optuna (default: auc, choices: auc, ba, mcc, sens, spec)' , default='auc', required=False)
args.measure
args = parser.parse_args()

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

n_runs = args.nruns
n_epochs = 1000
seed=0
name =args.name
label_path = args.labelfile
ppi_path = args.ppifile
expr_path=args.exprfile
ortho_path=args.orthofile
subloc_path=args.sublocfile 
no_ppi=args.noppi 
weights=args.weights
train_mode = args.trainmode
hypersearch = args.hypersearch
snapshot_name = get_snapshot_name(name, expr_path, ortho_path, subloc_path, no_ppi, weights)
metrics = {'auc': 1, 'ba': 2, 'mcc': 3, 'sens': 4 : 'spec': 5}
if hypersearch:
    seed = np.random.randint(1000) + 10
    datasets = []
    for i in range(3):
        set_seed(seed+i)
        datasets += [data(label_path, ppi_path, expr_path, ortho_path, subloc_path, no_ppi=no_ppi, weights=weights, verbose=True) for i in range(3)]
    hyper_search(name, './studies', datasets, metrics[args.measure])
elif n_runs:
    print(f'Training on {n_runs} runs')
    m = np.array([main(name, label_path, 
        ppi_path=ppi_path, expr_path=expr_path, ortho_path=ortho_path, subloc_path=subloc_path, 
        no_ppi=no_ppi, weights=weights, train_mode=train_mode, n_epochs=n_epochs,
        savedir='models', predsavedir='results',seed=i, verbose=args.verbose)[1:] for i in range(n_runs)])
    measures = np.ravel(np.column_stack((np.mean(m, axis=0),np.std(m, axis=0))))
    save_results(os.path.join('results', f'{modelname}_{snapshot_name}_r{n_runs}.csv'), *measures)
else:
    print('Training a single run with seed', seed)
    main(name, label_path, 
        ppi_path=ppi_path, expr_path=expr_path, ortho_path=ortho_path, subloc_path=subloc_path, 
        no_ppi=no_ppi, weights=weights, train_mode=train_mode, n_epochs=n_epochs,
        savedir='models', predsavedir='results',seed=seed, verbose=args.verbose)
