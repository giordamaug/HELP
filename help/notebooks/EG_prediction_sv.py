import pandas as pd
import numpy as np
import random
from help.preprocess.loaders import feature_assemble_df
import os,sys
import argparse
from joblib import Parallel, delayed
from help.models.prediction import predict_cv
import tabulate
from ast import literal_eval
from tabulate import tabulate
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PLOS COMPBIO')
parser.add_argument('-i', "--inputfile", dest='inputfile', metavar='<inputfile>', nargs="+", type=str, help='input attribute filename list', required=True)
parser.add_argument('-c', "--chunks", dest='chunks', metavar='<chunks>', nargs="+", type=int, help='no of chunks for attribute filename list', required=False)
parser.add_argument('-X', "--excludelabels", dest='excludelabels', metavar='<excludelabels>', nargs="+", default=[], help='labels to exclude (default NaN, values any list)', required=False)
parser.add_argument('-L', "--labelname", dest='labelname', metavar='<labelname>',  type=str, help='label name (default label)', default='label', required=False)
parser.add_argument('-l', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename', required=True)
parser.add_argument('-A', "--aliases", dest='aliases', default="{}", metavar='<aliases>', help='the dictionary for label renaming (es: {"oldlabel1": "newlabel1", ..., "oldlabelN": "newlabelN"})', required=False)
parser.add_argument('-b', "--seed", dest='seed', metavar='<seed>', type=int, help='random seed (default: 1)' , default='1', required=False)
parser.add_argument('-r', "--repeat", dest='repeat', metavar='<repeat>', type=int, help='n. of iteration (default: 10)' , default=10, required=False)
parser.add_argument('-f', "--folds", dest='folds', metavar='<folds>', type=int, help='n. of cv folds (default: 5)' , default=5, required=False)
parser.add_argument('-j', "--jobs", dest='jobs', metavar='<jobs>', type=int, help='n. of parallel jobs (default: -1)' , default=-1, required=False)
parser.add_argument('-B', "--batch", action='store_true', help='enable batch mode (no output)', required=False)
parser.add_argument('-v', "--voters", dest='voters', metavar='<voters>', type=int, help='n. of voter predictors (default: 1 - one classifier)' , default=1, required=False)
parser.add_argument('-ba', "--balanced", action='store_true', default=False, help='enable balancing in classifier (default disabled)', required=False)
parser.add_argument('-fx', "--fixna", action='store_true', default=False, help='enable fixing NaN (default disabled)', required=False)
parser.add_argument('-n', "--normalize", dest='normalize', metavar='<normalize>',  type=str, help='normalization mode (default None)', choices=['max', 'std'], required=False)
parser.add_argument('-o', "--outfile", dest='outfile', metavar='<outfile>', help='output file for performance measures sumup', type=str, required=False)
parser.add_argument('-s', "--scorefile", dest='scorefile', metavar='<scorefile>', type=str, help='output file reporting all measurements', required=False)
parser.add_argument('-p', "--predfile", dest='predfile', metavar='<predfile>', type=str, help='output file reporting predictions', required=False)
args = parser.parse_args()

## redefine print function
import time
def vprint(string):
  if args.batch:
    def fun(string):
       pass
    fun
  else:
    __builtins__.print(string)

if args.batch:
   verbose = False
else:
   verbose = True
print = vprint   
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

set_seed(1)
label_file = args.labelfile
label_name = args.labelname
features = []
if args.chunks is not None:                  # load files by chunks
   assert len(args.chunks) == len(args.inputfile), "Chunk list must be same size of input list!" 
   for attrfile,nchunks in zip(args.inputfile,args.chunks):
      fixna = args.fixna
      normalization = False if args.normalize is None else args.normalize
      if 'Emb' in os.path.basename(attrfile):
         features += [{'fname': attrfile, 'fixna' : False, 'normalize': None, 'nchunks': nchunks}]  # no normalization for embedding
      else:
         features += [{'fname': attrfile, 'fixna' : fixna, 'normalize': normalization, 'nchunks': nchunks}]
else:                                         # no kile is are 
   for attrfile in args.inputfile:
      fixna = args.fixna
      normalization = False if args.normalize is None else args.normalize
      if 'Emb' in os.path.basename(attrfile):
         features += [{'fname': attrfile, 'fixna' : False, 'normalize': None}]  # no normalization for embedding
      else:
         features += [{'fname': attrfile, 'fixna' : fixna, 'normalize': normalization}]
      
df_lab = pd.read_csv(label_file, index_col=0)
# get label aliases
label_aliases = literal_eval(args.aliases)
for key,newkey in label_aliases.items():
    if key in np.unique(df_lab[label_name].values):
        print(f'- replacing label {key} with {newkey}')
        df_lab = df_lab.replace(key, newkey)
# exclude labels
print(f'- removing label {args.excludelabels}')
df_lab = df_lab[df_lab[label_name].isin(args.excludelabels) == False]
df_X, df_y = feature_assemble_df(df_lab, features=features, subsample=False, seed=1, saveflag=False, verbose=verbose)
print(f'Working with {args.voters} classifiers...')

def predict_cv_sv(df_X, df_y, n_voters=1, n_splits=5, colname='label', balanced=False, seed=42, verbose=False):
   # find the majority class: will be split among voters
   minlab = df_y[colname].value_counts().nsmallest(1).index[0]
   maxlab = df_y[colname].value_counts().nlargest(1).index[0]
   if verbose: print(f"Majority {maxlab} {df_y[colname].value_counts()[maxlab]}, minority {minlab} {df_y[colname].value_counts()[minlab]}")
   df_y_ne = df_y[df_y['label']==maxlab]
   #df_y_ne = df_y_ne.sample(frac=1, random_state=seed)
   df_y_e = df_y[df_y['label']!=maxlab]
   splits = np.array_split(df_y_ne, n_voters)
   predictions_ne = pd.DataFrame()
   predictions_e = pd.DataFrame(index=df_y_e.index)
   d=np.empty((len(df_y_e.index),),object)
   d[...]=[list() for _ in range(len(df_y_e.index))]
   predictions_e['probabilities'] = d
   predictions_e['label'] = np.array([0 for idx in df_y_e.index])
   predictions_e['prediction'] = np.array([np.nan for idx in df_y_e.index])
   for df_index_ne in splits:
      df_x = pd.concat([df_X.loc[df_index_ne.index], df_X.loc[df_y_e.index]])
      df_yy = pd.concat([df_y.loc[df_index_ne.index], df_y.loc[df_y_e.index]])
      _, _, preds = predict_cv(df_x, df_yy, n_splits=n_splits, method='LGBM', balanced=balanced, verbose=verbose, seed=seed)
      predictions_ne = pd.concat([predictions_ne, preds.loc[df_index_ne.index]])
      r = np.empty((len(df_y_e.index),),object)
      r[...]=[predictions_e.loc[idx]['probabilities'] + [preds.loc[idx]['probabilities']]  for idx in df_y_e.index]
      predictions_e['probabilities'] = r
   predictions_e['prediction'] = predictions_e['probabilities'].map(lambda x: 0 if sum(x)/n_voters > 0.5 else 1)
   predictions_e['probabilities'] = predictions_e['probabilities'].map(lambda x: sum(x)/n_voters)
   predictions = pd.concat([predictions_ne, predictions_e])
   test_y = predictions['label'].values
   preds = predictions['prediction'].values
   probs = predictions['probabilities'].values
   cm = confusion_matrix(test_y, preds)
   scores = pd.DataFrame([[roc_auc_score(test_y, 1-probs), accuracy_score(test_y, preds),
                           balanced_accuracy_score(test_y, preds),
                           cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                           cm[1, 1] / (cm[1, 0] + cm[1, 1]),
                           matthews_corrcoef(test_y, preds),cm]], 
                           columns=["ROC-AUC", "Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM'], index=[seed])
   return scores, predictions

def classify(n_voters, repeat, n_splits, jobs, verbose):
  if jobs == 1:
    print(f'Running seq on 1 cpu...')
    result = [predict_cv_sv(df_X, df_y, n_voters=args.voters,  n_splits=n_splits, balanced=args.balanced, seed=seed, verbose=verbose) for seed in range(repeat)]
  else:
    if jobs == -1:
        print(f'Running par on {os.cpu_count()} cpus...')
    else:
        print(f'Running par on {jobs} cpus...')
    result = Parallel(n_jobs=jobs, prefer='threads')(delayed(predict_cv_sv)(df_X, df_y, n_voters=args.voters, n_splits=n_splits, balanced=args.balanced, seed=seed, verbose=verbose) for seed in range(repeat))
  return result

columns_names = ["ROC-AUC", "Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame()
preds = pd.DataFrame()
out = classify(args.voters, args.repeat, args.folds, args.jobs, verbose)
print(out)
for iter,res in enumerate(out):
   scores = pd.concat([scores,res[0]])
   preds = pd.concat([preds,res[1]])
if args.scorefile is not None:
   scores.to_csv(args.scorefile, index=False)
else:
   print(scores)
if args.predfile is not None:
   preds.to_csv(args.predfile, index=False)
else:
   print(preds)

df_scores = pd.DataFrame([f'{val:.4f}±{err:.4f}' for val, err in zip(scores.loc[:, scores.columns != "CM"].mean(axis=0).values,
                          scores.loc[:, scores.columns != "CM"].std(axis=0))] + [(scores[['CM']].sum()/args.repeat).values[0].tolist()],
                          columns=['measure'], index=scores.columns)
import sys
distrib = np.unique(df_y[label_name].values, return_counts=True)
ofile = sys.stdout if args.outfile is None else open(args.outfile, "a")
ofile.write(f'METHOD: LGBM\tVOTERS: {args.voters}\tBALANCE: {"yes" if args.balanced else "no"}\n')
ofile.write(f'PROBL: {" vs ".join(list(np.unique(df_y.values)))}\n')
ofile.write(f'INPUT: {" ".join(str(os.path.basename(x)) for x in args.inputfile)}\n')
ofile.write(f'LABEL: {os.path.basename(args.labelfile)} DISTRIB: {distrib[0][0]} : {distrib[1][0]}, {distrib[0][1]}: {distrib[1][1]}\n')
ofile.write(tabulate(df_scores, headers='keys', tablefmt='psql') + '\n')
ofile.close()