import pandas as pd
from help.preprocess.loaders import feature_assemble
import os,sys
import argparse
from joblib import Parallel, delayed
from help.models.prediction import predict_cv

parser = argparse.ArgumentParser(description='BIOMAT 2022 Workbench')
parser.add_argument('-i', "--inputfile", dest='inputfile', metavar='<inputfile>', nargs="+", type=str, help='input attribute filename list', required=True)
parser.add_argument('-X', "--excludelabels", dest='excludelabels', metavar='<excludelabels>', nargs="+", default=[], help='labels to exclude (default NaN, values any list)', required=False)
parser.add_argument('-L', "--labelname", dest='labelname', metavar='<labelname>',  type=str, help='label name (default label)', default='label', required=False)
parser.add_argument('-l', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename', required=True)
parser.add_argument('-A', "--aliases", dest='aliases', default="{}", metavar='<aliases>', required=False)
parser.add_argument('-Z', "--normalize", dest='normalize', metavar='<normalize>', type=str, help='normalize mode (default: None, choice: None|zscore|minmax)', choices=[None, 'zscore', 'minmax'], default=None, required=False)
parser.add_argument('-I', "--imputation", dest='imputation', metavar='<imputation>', type=str, help='imputation mode (default: None, choice: None|mean|zero)', choices=[None, 'mean', 'zero'], default=None, required=False)
parser.add_argument('-b', "--seed", dest='seed', metavar='<seed>', type=int, help='seed (default: 1)' , default='1', required=False)
parser.add_argument('-r', "--repeat", dest='repeat', metavar='<repeat>', type=int, help='n. of iteration (default: 10)' , default=10, required=False)
parser.add_argument('-f', "--folds", dest='folds', metavar='<folds>', type=int, help='n. of cv folds (default: 5)' , default=5, required=False)
parser.add_argument('-j', "--jobs", dest='jobs', metavar='<jobs>', type=int, help='n. of parallel jobs (default: -1)' , default=-1, required=False)
parser.add_argument('-B', "--batch", action='store_true', required=False)
parser.add_argument('-P', "--proba", action='store_true', required=False)
parser.add_argument('-o', "--outfile", dest='outfile', metavar='<outfile>', type=str, help='output file', required=False)
parser.add_argument('-s', "--scorefile", dest='scorefile', metavar='<scorefile>', type=str, help='score file', required=False)
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
print = vprint   

label_file = args.labelfile
features = []
for attrfile in args.inputfile:
    if 'Emb' in os.path.basename(attrfile):
        features += [{'fname': attrfile, 'fixna' : False, 'normalize': None}]
    else:
        features += [{'fname': attrfile, 'fixna' : True, 'normalize': 'std'}]

print(features)
df_X, df_y = feature_assemble(label_file = label_file, features=features, subsample=False, seed=1, saveflag=False, verbose=False)

#scores, measures = predict_cv(df_X, df_y, n_splits=5, balanced=True, display=True, outfile=args.outfile)
#scores.to_csv(args.scorefile)
#print(measures)

#sys.exit(0)

def classify(nfolds, repeat, jobs, verbose):
  if jobs == 1:
    print(f'Running seq on 1 cpu...')
    out = [predict_cv(df_X, df_y, n_splits=nfolds, balanced=True, seed=seed, verbose=verbose) for seed in range(repeat)]
  else:
    if jobs == -1:
        print(f'Running par on {os.cpu_count()} cpus...')
    else:
        print(f'Running par on {jobs} cpus...')
    out = Parallel(n_jobs=jobs, prefer='threads')(delayed(predict_cv)(df_X, df_y, n_splits=nfolds, balanced=True, seed=seed, verbose=verbose) for seed in range(repeat))
  return out

columns_names = ["ROC-AUC", "Accuracy","BA", "Sensitivity", "Specificity","MCC", 'CM']
scores = pd.DataFrame(columns=columns_names)
out = classify(args.folds, args.repeat, args.jobs, False)
for iter,res in enumerate(out):
  scores = pd.concat([scores,res[1]])
if args.scorefile is not None:
  scores.to_csv(args.scorefile, index=False)
else:
   print(scores)