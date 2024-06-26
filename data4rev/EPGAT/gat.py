import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import *
import optuna

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gat_human = {
    'lr': 0.005,
    'weight_decay': 5e-4,
    'h_feats': [8, 1],
    'heads': [8, 1],
    'dropout': 0.4,
    'negative_slope': 0.2}

GATparams = {'KIDNEY': gat_human,
              # best params from hyper-search (not used)
              #{'linear_layer': None, #256, 
              #   'n_layers': 1, 
              #   'h_feats': [16, 1], #'h_feat_0': 16, 'h_feat_1': 8, 
              #   'heads': [2, 1], #'head_0': 2, 'head_1': 4, 'head_2': 8, 
              #   'lr': 0.0010633766921485904, 
              #   'weight_decay': 3.905559247728033e-05, 
              #   'dropout': 0.3655805198848601}
              # AUC: 0.886±0.020
              # Acc.: 0.810v0.052
              # BA:  0.798±0.034971  
              # Sens.: 0.812±0.062 
              # Spec.: 0.783±0.104   
              # MCC: 0.373±0.046 
              # deafult gat_human (used)
              # AUC: 0.902±0.007
              # Acc.: 0.834±0.028
              # BA:  0.824±0.012  
              # Sens.: 0.813±0.045
              # Spec.: 0.835±0.033  
              # MCC: 0.414±0.026
             'LUNG': gat_human,
                 # best params from hyper-search (not used)
                 #{'linear_layer': None, #128, 
                 # 'n_layers': 1, 
                 # 'h_feats': [64, 1], #'h_feat_0': 64, 
                 # 'heads': [4, 1], #'head_0': 4, 'head_1': 8, 
                 # 'lr': 0.0018142943583547278, 
                 # 'weight_decay': 0.00026208428025768776, 
                 # 'dropout': 0.45691838072317914},
                 # AUC: 0.908±0.018
                 # Acc: 0.810±0.041
                 # BA: 0.832±0.013
                 # Sens. : 0.858±0.044
                 # Spec. : 0.807±0.047
                 # MCC: 0.405±0.040
                 # deafult gat_human (used)
                 # AUC: 0.913±0.009  
                 # Acc: 0.843±0.032  
                 # BA: 0.832±0.014  
                 # Sens. : 0.819±0.051   
                 # Spec. : 0.845±0.037
                 # MCC: 0.430±0.031   
             'BRAIN':
                # best params from hyper-search (used)
                {'linear_layer': None, 
                 'n_layers': 2, 
                 'h_feats': [32, 8, 1], #'h_feat_0': 32, 'h_feat_1': 8, 
                 'heads': [8, 4, 1], #'head_0': 8, 'head_1': 4, 'head_2': 8, 
                 'lr': 0.0005752973665699493, 
                 'weight_decay': 2.4757999496124907e-05, 
                 'dropout': 0.13721629534853402}, 
                 # AUC: 0.908±0.012
                 # Acc: 0.857±0.022
                 # BA: 0.833±0.008
                 # Sens. : 0.806±0.027
                 # Spec. : 0.861±0.026
                 # MCC: 0.445±0.030
             'HUMAN': 
                # best params from hyper-search (used)
                 {'linear_layer': None, 
                  'n_layers': 1, # best BA with one layer
                  'h_feats': [64, 1], #'h_feat_0': 64, 'h_feat_1': 64, 
                  'heads': [4, 1], #'head_0': 4, 'head_1': 4, 'head_2': 8, 
                  'lr': 0.0023087204557795176, 
                  'weight_decay': 0.0001263490604135409, 
                  'dropout': 0.3390586753625013}
                 # AUC: 0.880±0.017
                 # Acc: 0.784±0.043
                 # BA: 0.798±0.020
                 # Sens. : 0.815±0.063
                 # Spec. : 0.781±0.050
                 # MCC: 0.355±0.030
             }

def specificity_myscore(ground_truth, predictions):
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(ground_truth):        
        if m==predictions[l] and m==1:
            tp+=1
        if m==predictions[l] and m==0:
            tn+=1
        if m!=predictions[l] and m==1:
            fn+=1
        if m!=predictions[l] and m==0:
            fp+=1
    return tn/(tn+fp)


class Loss():
    def __init__(self, y, idx):
        self.y = y
        idx = np.array(idx)

        self.y_pos = y[y == 1]
        self.y_neg = y[y == 0]

        self.pos = idx[y.cpu() == 1]
        self.neg = idx[y.cpu() == 0]

    def __call__(self, out):
        loss_p = F.binary_cross_entropy_with_logits(
            out[self.pos].squeeze(), self.y_pos)
        loss_n = F.binary_cross_entropy_with_logits(
            out[self.neg].squeeze(), self.y_neg)
        loss = loss_p + loss_n
        return loss

def hyper_search(name, savepath, data, metric_pos=1):

    def objective(trial):
        linear_layer = trial.suggest_categorical(
            f'linear_layer', [None, 64, 128, 256])
        linear_layer = None

        n_layers = trial.suggest_int('n_layers', 1, 3)
        h_feats = [trial.suggest_categorical(
            f'h_feat_{i}', [8, 16, 32, 64]) for i in range(n_layers)]
        h_feats += [1]

        heads = [trial.suggest_categorical(
            f'head_{i}', [1, 2, 4, 8]) for i in range(n_layers+1)]

        params = {
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
            'h_feats': h_feats,
            'heads': heads,
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
            'negative_slope': 0.2}

        try:
            final_metric = [] 
            for d in data:
                (edge_index, edge_weights), X, (train_idx, train_y), \
                    (val_idx, val_y), (test_idx, test_y), _ = d

                model = train(params, X, edge_index, edge_weights,
                              train_y, train_idx, val_y, val_idx)

                measure = test(model, X, edge_index, (test_idx, test_y)) # preds, auc, ba, mcc, sens, spec
                final_metric.append(measure[metric_pos])

            return np.mean(final_metric)
        except:
            return -1

    study = optuna.create_study(
        study_name=f'gat_{name}',
        direction='maximize',
        load_if_exists=True,
        storage=f'sqlite:///{savepath}/gat_{name}.db')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print('Best Params:', best_params)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(savepath, f'gat_{name}_hypersearch.csv'))
    print(df.head())
    
def train(params, X, A, edge_weights, train_y, train_idx, val_y, val_idx, save_best_only=True, n_epochs=1000, savepath='',):

    model = GAT(in_feats=X.shape[1], **params)
    model.to(DEVICE)
    X = X.to(DEVICE)
    A = A.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_y = val_y.to(DEVICE)
    if edge_weights is not None:
        edge_weights = edge_weights.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fnc = Loss(train_y, train_idx)
    val_loss_fnc = Loss(val_y, val_idx)

    iterable = tqdm(range(n_epochs))
    for i in iterable:
        model.train()
        logits = model(X, A, edge_attr=edge_weights)

        optimizer.zero_grad()
        loss = loss_fnc(logits)
        loss.backward()
        optimizer.step()

        logits = logits.detach()
        val_loss = val_loss_fnc(logits)
        train_auc = evalAUC(None, 0, 0, train_y, 0, logits[train_idx])
        val_auc = evalAUC(None, 0, 0, val_y, 0, logits[val_idx])

        tqdm.set_description(iterable, desc='Loss: %.4f ; Val Loss %.4f ; Train AUC %.4f. Validation AUC: %.4f' % (
            loss, val_loss, train_auc, val_auc))

    score = evalAUC(model, X, A, val_y, val_idx)
    print(f'Last validation AUC: {val_auc}')

    if savepath:
        save = {
            'auc': score,
            'model_params': params,
            'model_state_dict': model.state_dict()
        }
        torch.save(save, savepath)

    return model


def test(model, X, A, test_ds=None):
    model.to(DEVICE).eval()
    X = X.to(DEVICE)
    A = A.to(DEVICE)

    with torch.no_grad():
        logits = model(X, A)
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()

    if test_ds is not None:
        test_idx, test_y = test_ds
        test_y = test_y.cpu().numpy()
        auc = metrics.roc_auc_score(test_y, probs[test_idx])
        preds = (probs[test_idx] > 0.5) * 1
        score = metrics.accuracy_score(test_y, preds)
        ba = metrics.balanced_accuracy_score(test_y, preds)
        mcc = metrics.matthews_corrcoef(test_y, preds)
        sens = metrics.recall_score(test_y, preds)
        specs = specificity_myscore(test_y, preds)
        return probs, auc, score, ba, mcc, sens, specs
    return probs, None, None, None, None, None, 

class GAT(nn.Module):
    def __init__(self, in_feats=1,
                 h_feats=[8, 8, 1],
                 heads=[8, 8, 4],
                 dropout=0.6,
                 negative_slope=0.2,
                 linear_layer=None,
                 **kwargs):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.linear_layer = linear_layer
        if self.linear_layer is not None:
            print('Applying linear')
            self.linear = nn.Linear(in_feats, linear_layer)

        in_feats = in_feats if linear_layer is None else linear_layer
        for i, h_feat in enumerate(h_feats):
            last = i + 1 == len(h_feats)
            self.layers.append(GATConv(in_feats, h_feat,
                                       heads=heads[i],
                                       dropout=dropout,
                                       concat=False if last else True))
            in_feats = h_feat * heads[i]

    def forward(self, X, A, edge_attr=None, return_alphas=False):
        if self.linear_layer is not None:
            X = self.linear(X)
            #X = F.relu(X)

        alphas = []
        for layer in self.layers[:-1]:
            if return_alphas:
                X, alpha, _ = layer(
                    X, A, edge_attr=edge_attr, return_alpha=True)
                alphas.append(alpha)
            else:
                X = layer(X, A, edge_attr=edge_attr)
            X = F.relu(X)
            X = F.dropout(X, self.dropout)

        if return_alphas:
            X, alpha, edge_index = self.layers[-1](
                X, A, edge_attr=edge_attr, return_alpha=True)
            alphas.append(alpha)
            return X, alphas, edge_index

        X = self.layers[-1](X, A, edge_attr=edge_attr)
        return X