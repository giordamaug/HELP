import pandas as pd
import numpy as np
import random
from HELPpy.preprocess.loaders import feature_assemble_df
import os,sys
import argparse
from joblib import Parallel, delayed
from HELPpy.models.prediction import k_fold_cv, VotingSplitClassifier
import tabulate
from ast import literal_eval
from tabulate import tabulate
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

if args.balanced:
   clf = VotingSplitClassifier(n_voters=args.voters, n_jobs=args.jobs, random_state=-1, class_weight='balanced')
else:
   clf = VotingSplitClassifier(n_voters=args.voters, n_jobs=args.jobs, random_state=-1)

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
df_X, df_y = feature_assemble_df(df_lab, features=features, seed=1, saveflag=False, verbose=verbose)

def classify(nfolds, repeat, jobs, verbose):
  if jobs == 1:
    print(f'Running seq on 1 cpu...')
    result = [k_fold_cv(df_X, df_y, clf, n_splits=nfolds, seed=rseed, verbose=verbose) for rseed in range(repeat)]
  else:
    if jobs == -1:
        print(f'Running par on {os.cpu_count()} cpus...')
    else:
        print(f'Running par on {jobs} cpus...')
    result = Parallel(n_jobs=jobs, prefer='threads')(delayed(k_fold_cv)(df_X, df_y, clf, n_splits=nfolds, seed=rseed, verbose=verbose) for rseed in range(repeat))
  return result

columns_names = ["ROC-AUC", "Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame()
preds = pd.DataFrame()
out = classify(args.folds, args.repeat, args.jobs, verbose)
for iter,res in enumerate(out):
   scores = pd.concat([scores,res[1]])
   preds = pd.concat([preds,res[2]])
if args.scorefile is not None:
   scores.to_csv(args.scorefile, index=False)
else:
   print(scores)

if args.predfile is not None:
   preds.to_csv(args.predfile, index=False)
else:
   print(preds)
df_scores = pd.DataFrame([f'{val:.4f}±{err:.4f}' for val, err in zip(scores.loc[:, scores.columns != "CM"].mean(axis=0).values,
                          scores.loc[:, scores.columns != "CM"].std(axis=0))] + [(scores[['CM']].sum()).values[0].tolist()],
                          columns=['measure'], index=scores.columns)
import sys
distrib = np.unique(df_y[label_name].values, return_counts=True)
ofile = sys.stdout if args.outfile is None else open(args.outfile, "a")
ofile.write(f'METHOD: LGBM\tMODE: {"prob" if args.proba else "pred"}\tBALANCE: {"yes" if args.balanced else "no"}\n')
ofile.write(f'PROBL: {" vs ".join(list(np.unique(df_y.values)))}\n')
ofile.write(f'INPUT: {" ".join(str(os.path.basename(x)) for x in args.inputfile)}\n')
ofile.write(f'LABEL: {os.path.basename(args.labelfile)} DISTRIB: {distrib[0][0]} : {distrib[1][0]}, {distrib[0][1]}: {distrib[1][1]}\n')
ofile.write(f'SUBSAMPLE: 1:{args.subfolds}\n' if args.subfolds>0 else 'SUBSAMPLE: NONE\n')
ofile.write(tabulate(df_scores, headers='keys', tablefmt='psql') + '\n')
ofile.close()