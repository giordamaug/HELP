#!/usr/bin/env python
# coding: utf-8

# # let's install pycaret !
# 

# In[2]:


# Install PyCaret
import numpy as np
import pandas as pd
import os, sys
from IPython.display import display

from pycaret.utils import version

# PyCaret version
print(version())


# In[3]:


from pycaret.classification import *


# In[86]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List
def load_features(filenames: List[str] = [], fixnans= [], normalizes=[], colname: str="label", 
                  verbose: bool = False, show_progress: bool = False) -> pd.DataFrame:
    """
    Load and assemble features and labels for machine learning tasks.

    :param List[str] features: List of feature filepaths
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param int seed: Random seed for reproducibility. Default is 1.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.
    :param bool show_progress: Whether to print progress bar while loading file. Default is False.

    :returns: Tuple containing the assembled features (X) and labels (Y) DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        
    :example:

    .. code-block:: python

        colname = "target_column"
        seed = 1
        verbose = False

        df_label = pd.read_csv("label_file.csv2, index_col=0)
        X, Y = load_features(['path/to/feature_file1.csv', 'path/to/feature_file2.csv'], fix_na=True, colname, seed, verbose)
    """

    # Common indices among labels and features
    x = pd.DataFrame()

    # Process each feature file
    for f,fixna,norm in zip(filenames, fixnans, normalizes):
        feat_df = pd.read_csv(f, index_col=0)
        feat_df.index = feat_df.index.map(str)
        fname = os.path.basename(f).rsplit('.', 1)[0]

        # Handle missing values if required
        if verbose:
            cntnan = feat_df.isna().sum().sum()
            print(f"[{fname}] found {cntnan} Nan...")
        if fixna:
            if verbose:
                print(f"[{fname}] Fixing NaNs with mean ...")
            feat_df = feat_df.fillna(feat_df.mean())

        # Normalize features
        if norm == 'std':
            scaler = MinMaxScaler()
            if verbose:
                print(f"[{fname}] Normalization with {norm} ...")
            feat_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index, columns=feat_df.columns)
        elif norm == 'max':
            scaler = StandardScaler()
            if verbose:
                print(f"[{fname}] Normalization with {norm}...")
            feat_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index, columns=feat_df.columns)
        else:
            if verbose:
                print(f"[{fname}] No normalization...")

        # merge features features
        x = pd.merge(x, feat_df, left_index=True, right_index=True, how='outer')

    # Return the assembled features (X) and labels (Y)
    return x


# # Load the dataset and split

# In[90]:


from sklearn.model_selection import train_test_split
path = '/home/maurizio/PLOS_CompBiology/HELP/data/'
attributes = load_features([os.path.join(path, 'Kidney_BIO.csv'), 
                            #os.path.join(path, 'Kidney_CCcfs.csv'),
                            os.path.join(path, 'Kidney_EmbN2V_128.csv')], 
                            fixnans=[True, True, False], normalizes=['std', 'std', None], verbose=True)
label = pd.read_csv(os.path.join(path, 'Kidney_HELP.csv'), index_col=0).replace({'aE':'NE', 'sNE': 'NE'})
idx_common = np.intersect1d(attributes.index.values, label.index.values)
attributes = attributes.loc[idx_common]
label = label.loc[idx_common]
X_train, X_test, y_train, y_test = train_test_split(attributes, label, shuffle=False)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


# # Start tuning session

# In[93]:


clf1 = setup(data = train, 
             target = 'label',
             numeric_imputation = 'mean',
             categorical_features = [], session_id = 444,
             fold_strategy = "stratifiedkfold", fold=5,
             #ignore_features = ['Name','Ticket','Cabin'],
             verbose = True)


# # Define our model

# In[94]:


from sklearn.base import clone, BaseEstimator
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier 
from sklearn.preprocessing import LabelEncoder
class veLGBM(BaseEstimator):

    def __init__(self, n_voters=10, voting='soft', n_jobs=-1, verbose=False, random_state=42, **kwargs):
        self.kwargs = kwargs
        # intialize ensemble ov voters
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_voters = n_voters
        self.estimators_ = [LGBMClassifier(**kwargs, verbose=-1, random_state=random_state) for i in range(n_voters)]
    
    def __sklearn_clone__(self):
        return self

    def _fit_single_estimator(self, i, X, y, index_ne, index_e):
        """Private function used to fit an estimator within a job."""
        df_X = np.append(X[index_ne], X[index_e], axis=0)
        df_y = np.append(y[index_ne], y[index_e], axis=0)
        clf = clone(self.estimators_[i])
        clf.fit(df_X, df_y)
        return clf
    
    def fit(self, X, y):
        # Find the majority and minority class
        #assert (isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame)) and (isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame)), "Only array or pandas dataframe input!"
        X = X.values
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.values.ravel())

        unique, counts = np.unique(y, return_counts=True)
        minlab = unique[np.argmin(counts)]
        maxlab = unique[np.argmax(counts)]

        if self.verbose:
            print(f"Majority {maxlab} {max(counts)}, minority {minlab} {min(counts)}")

        # Separate majority and minority class
        all_index_ne = np.where(y == maxlab)[0]
        index_e = np.where(y == minlab)[0]

        # Split majority class among voters
        if self.random_state >= 0:
            np.random.seed(self.random_state)
            np.random.shuffle(all_index_ne)
            np.random.shuffle(index_e)
        splits = np.array_split(all_index_ne, self.n_voters)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_estimator)(i,X, y, index_ne, index_e) 
                                                        for i,index_ne in enumerate(splits))
        return self
    
    def predict_proba(self, X, y=None):
        # Find the majority and minority class
        #assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame), "Only array or pandas dataframe input!"
        X = X.values
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.sum(probabilities, axis=0)/self.n_voters
    
    def predict(self, X, y=None):
        #assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame), "Only array or pandas dataframe input!"
        X = X.values
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.argmax(np.sum(probabilities, axis=0)/self.n_voters, axis=1)
    
#velgbm = veLGBM()
# train using create_model
#velgbm_trained = create_model(velgbm)


# In[98]:


from sklearn.metrics import *
from imblearn.metrics import specificity_score
[remove_metric(m) for m in ['Precision', 'F1', 'Kappa']]  # remove unused metrics
add_metric('Sensitivity', 'Sensitivity', specificity_score, greater_is_better = True)
add_metric('Specificity', 'Specificity', recall_score, greater_is_better = True)
add_metric('Balanced Accuracy', 'BA', balanced_accuracy_score, greater_is_better = True)
#add_metric('ROC-AUC', 'ROC-AUC', roc_auc_score, greater_is_better = True, multiclass=False)


get_metrics()


# In[100]:


from sklearn.metrics import balanced_accuracy_score
#classifiers = [veLGBM(), 'lightgbm', 'xgboost', 'ada', 'rf', 'dt', 'gbc', 'lda', 'lr', 'et', 'svm']
#add_metric('Balanced Accuracy', 'BA', balanced_accuracy_score, greater_is_better = True) 
#results = compare_models(include=classifiers, sort='BA')


# In[105]:


#df = pull()


# In[114]:


#results.get_params()


# In[115]:


#df.to_csv("pycaret_best_classifier_metrics.csv", index=True)


# In[109]:


#print(df.to_latex())


# In[ ]:


velgbm = veLGBM()
# train using create_model
velgbm_trained = create_model(velgbm)

# tune model
tuned_dt = tune_model(velgbm_trained, 
                      optimize = 'BA',
                      return_train_score=True, 
                      custom_grid={'n_voters':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]})


# In[120]:


df2 = pull()


# In[122]:


df2.to_csv("tuned_veLGB.csv")


# In[ ]:


velgbm = veLGBM()
velgbm_trained  = create_model(velgbm) 
hparams = {"n_voters" :[2, 4, 6, 8, 10, 12, 14, 16]}
tuned_rf, tuner = tune_model(velgbm_trained, optimize = 'BA', search_algorithm='grid', custom_grid=hparams, return_tuner=True)
print(tuner)
pd.DataFrame(tuner.cv_results_).to_csv("results.csv", inde=True)
print(tuned_rf)



