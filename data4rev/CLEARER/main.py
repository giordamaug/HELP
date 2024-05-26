import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from tqdm import tqdm

# Set working directory scassa
#os.chdir("/to/CLEARER_directory/")
EsInfo = pd.read_csv("Class_labels/Sc.csv", sep=",", header=0)
print(EsInfo.head())
print(EsInfo['Essential CEG'].value_counts())

# Generate class labels suitable for Python
EsInfo['Essential CEG'] = EsInfo['Essential CEG'].astype('category').cat.codes
print(EsInfo['Essential CEG'].value_counts())

# Load combined features
Data = pd.read_csv("Features/Sc_features.csv.gz", sep=",", header=0, compression='gzip')
Data.set_index('genes', inplace=True)

# Assign class labels
Data['label'] = EsInfo.set_index('Gene').loc[Data.index, 'Essential CEG']

# Randomize Data
np.random.seed(69)
Data = Data.sample(frac=1).reset_index()

# Split Data
seq = np.round(np.linspace(0, len(Data), 6)).astype(int)
val_sets = [Data.iloc[seq[i]:seq[i+1]] for i in range(5)]
train_sets = [Data.drop(val.index) for val in val_sets]

# Feature selection
N = 5
for i in tqdm(range(N), descr="Feature selection step", total=N):
    train_set = train_sets[i]
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]

    # Lasso feature selection using LogisticRegressionCV
    clf = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', scoring='roc_auc', max_iter=1000).fit(X_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    X_train_selected = model.transform(X_train)

    # Remove highly correlated features
    corr_matrix = pd.DataFrame(X_train_selected).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    X_train_selected = pd.DataFrame(X_train_selected).drop(to_drop, axis=1)

    train_sets[i] = pd.concat([X_train_selected, y_train.reset_index(drop=True)], axis=1)
    val_sets[i] = val_sets[i][train_sets[i].columns]

# Machine learning
def train_rf(train_data):
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scores = cross_val_score(rf, X_resampled, y_resampled, cv=kf, scoring='roc_auc')
    rf.fit(X_resampled, y_resampled)
    return rf, scores.mean()

results = Parallel(n_jobs=5)(delayed(train_rf)(train_sets[i]) for i in range(5))
rf_list, auc_scores = zip(*results)

# Performance evaluation on test set
def evaluate_model(rf, val_data):
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]
    
    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)[:, 1]
    
    cm = confusion_matrix(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    
    return cm, roc_auc, pr_auc

eval_results = [evaluate_model(rf_list[i], val_sets[i]) for i in range(5)]
cm_list, roc_auc_list, pr_auc_list = zip(*eval_results)

metrics = pd.DataFrame({
    'roc_auc': roc_auc_list,
    'pr_auc': pr_auc_list
})

metrics.loc['mean'] = metrics.mean()
metrics.loc['std'] = metrics.std()

metrics.to_csv("test_rf.csv", index=True)

# Performance evaluation on training set
def get_best_kappa(rf):
    results = pd.DataFrame(rf.cv_results_)
    return results.loc[results['mean_test_score'].idxmax()]

train_metrics = pd.concat([get_best_kappa(rf_list[i]) for i in range(5)])
train_metrics.to_csv("train_rf.csv", index=False)
