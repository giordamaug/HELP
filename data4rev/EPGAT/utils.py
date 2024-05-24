import os
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.sparse as sparse
from os.path import join
from tqdm import tqdm
import os
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse


def evalAUC(model, X, A, y, mask, logits=None):
    assert(model is not None or logits is not None)
    if model is not None:
        model.eval()
        with torch.no_grad():
            logits = model(X, A)
            logits = logits[mask]
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()
    y = y.cpu().numpy()
    auc = metrics.roc_auc_score(y, probs)
    return auc


def evaluate(model, X, A, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, A)
        logits = logits[mask]
        preds = (torch.sigmoid(logits) > 0.5).to(torch.float32)[:, 0]
        correct = torch.sum(preds == y)
        return correct.item() * 1.0 / len(y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_preds(modelname, preds, savepath, prefix, seed):
    name = f'{modelname}_{prefix}__s{seed}.csv'
    name = name.lower()
    path = os.path.join(savepath, name)
    df = pd.DataFrame(preds, columns=['gene', 'prediction'])
    df.to_csv(path)
    print('Saved the predictions to:', path)

def save_results(df_path, *measures):
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame([], columns=["auc_mean", "auc_std", "acc_mean", "acc_std", "ba_mean", "ba_std", 
                                       "mcc_mean", "mcc_std", "sens_mean", "sens_std", "spec_mean", "spec_std"])
    df.loc[len(df)] = [*measures]
    df.to_csv(df_path, index=False)
    print(df.tail())

def dim_reduction_cor(X, y, k=20):
    cors = np.zeros((X.shape[1]))

    # calculate the correlation with y for each feature
    for i in range(X.shape[1]):
        cor = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(cor):
            cors[i] = cor

    features = np.zeros_like(cors).astype(bool)
    features[np.argsort(-cors)[:k]] = True

    return features, cors


def data(label_path, ppi_path, expr_path, ortho_path, subloc_path, 
         string_thr=0, 
         seed=42,
         key = 'combined_score',
         source = 'A', target='B',
         labelname = 'label',
         no_ppi=False,
         weights=False,
         verbose=False):

    if verbose: print(f'PPI: {os.path.basename(ppi_path)}.')

    edges = pd.read_csv(ppi_path)
    if weights:
        if key in edges.columns:
            edges = edges[edges.loc[:, key] > string_thr].reset_index()
            edge_weights = edges[key] # / 1000
            if verbose: print('Filtered String network with thresh:', string_thr)
        else:
            edge_weights = pd.DataFrame(np.ones(len(edges)), columns=[key], index=edges.index)
    else:
        edge_weights = None
    edges = edges[[source, target]]

    edges = edges.dropna()
    index, edges = edges.index, edges.values
    ppi_genes = np.union1d(edges[:, 0], edges[:, 1])
    if weights:
        edge_weights = edge_weights.iloc[index.values].values

    labels = pd.read_csv(label_path, index_col=0)

    # filter labels not in the PPI network
    if verbose: print('Number of labels before filtering:', len(labels))
    labels = labels.loc[np.intersect1d(labels.index, ppi_genes)].copy()
    if verbose: print('Number of labels after filtering:', len(labels))

    genes = np.union1d(labels.index, ppi_genes)
    if verbose: print('Total number of genes:', len(genes))

    X = np.zeros((len(genes), 0))
    X = pd.DataFrame(X, index=genes)

    if ortho_path is not None:
        orths = pd.read_csv(ortho_path, index_col=0)
        columns = [f'ortholog_{i}' for i in range(orths.shape[1])]
        orths.columns = columns
        X = X.join(orths, how="left")
        if verbose: print('Orthologs dataset shape:', orths.shape)

    if expr_path is not None:
        expression = pd.read_csv(expr_path, index_col=0)
        columns = [f'expression_{i}' for i in range(expression.shape[1])]
        expression.columns = columns
        X = X.join(expression, how="left")
        if verbose: print('Gene expression dataset shape:', expression.shape)

    if subloc_path is not None:
        subloc = pd.read_csv(subloc_path, index_col=0)
        columns = [f'subloc_{i}' for i in range(subloc.shape[1])]
        subloc.columns = columns
        X = X.join(subloc, how="left")
        if verbose: print('Subcellular Localizations dataset shape:', subloc.shape)

    X = X.fillna(0)

    if pd.api.types.is_string_dtype(labels[labelname]):
        labels = labels.apply(LabelEncoder().fit_transform)
    train_ds, test_ds = train_test_split(
        labels, test_size=0.2, random_state=seed, stratify=labels)

    if verbose: 
        print(f'Num nodes {len(genes)} ; num edges {len(edges)}')
        print(f'X.shape: {None if X is None else X.shape}.')
        print(f'Train labels. Num: {len(train_ds)} ; Num pos: {train_ds[labelname].sum()}')
        print(f'Test labels. Num: {len(test_ds)} ; Num pos: {test_ds[labelname].sum()}')
    #print(X.tail())

    # return (edges, edge_weights), X, train_ds, test_ds, genes
    N = len(X)
    mapping = dict(zip(genes, range(N)))
    # ---------------------------------------------------

    # Preprocessing -------------------------------------
    # Remove self loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    if weights:
        edge_weights = edge_weights[mask]

    # Removes repeated connections
    df = pd.DataFrame({source: edges[:, 0], target: edges[:, 1]})
    df = df.drop_duplicates()
    edges = df.values
    indexes = df.index.values
    if weights:
        edge_weights = edge_weights[indexes]
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    edge_index = np.vectorize(mapping.__getitem__)(edges)
    if no_ppi:
        edges = np.ones((N, 2), dtype=int)
        edges[:, 0] = range(N)
        edges[:, 1] = range(N)

    degrees = np.zeros((N, 1))
    nodes, counts = np.unique(edge_index, return_counts=True)
    degrees[nodes, 0] = counts

    if X is None or not X.shape[1]:
        X = np.random.random((N, 50))

    if X.shape[1] < 50:
        X = np.concatenate([X, np.random.random((N, 50))], axis=1)

    X = np.concatenate([X, degrees.reshape((-1, 1))], 1)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    train, val = train_test_split(train_ds, test_size=0.05,
                     stratify=train_ds, random_state=seed)

    train_idx = [mapping[t] for t in train.index]
    val_idx = [mapping[v] for v in val.index]
    test_idx = [mapping[v] for v in test_ds.index]

    # number of feature to select -------------------------------------
    k = 300

    red_idx = np.concatenate([train_idx, test_idx, val_idx], 0)
    red_y = np.concatenate([train[labelname], test_ds[labelname], val[labelname]], 0)
    feats, cors = dim_reduction_cor(X[red_idx], red_y.astype(np.float32), k=k)
    X = X[:, feats]

    # Torch -------------------------------------------------
    edge_index = torch.from_numpy(edge_index.T)
    edge_index = edge_index.to(torch.long).contiguous()

    X = torch.from_numpy(X).to(torch.float32)
    train_y = torch.tensor(train[labelname], dtype=torch.float32)
    val_y = torch.tensor(val[labelname], dtype=torch.float32)
    test_y = torch.tensor(test_ds[labelname], dtype=torch.float32)

    # ---------------------------------------------------
    if verbose: 
        print('Not using PPI' if no_ppi else 'Using PPI')
        print(f'\nNumber of edges in graph: {len(edges)}' + '... only self loops on nodes' if no_ppi else '')
        print(f'Number of features: {X.shape[1]}')
        print(f'Number of nodes in graph: {len(X)}\n')
        print('Using Edge Weights' if edge_weights is not None and not no_ppi else 'Not using edge weights')

    return (edge_index, edge_weights), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), genes


