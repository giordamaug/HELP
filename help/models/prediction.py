import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from tabulate import tabulate
from typing import List,Dict,Union,Tuple


def set_seed(seed=1):
    """
    Set random and numpy random seed for reproducibility

    :param int seed: inistalization seed

    :returns None.
    """
    random.seed(seed)
    np.random.seed(seed)

def predict_cv(X, Y, n_splits=10, method='LGBM', balanced=False, saveflag: bool = False, outfile: str = 'predictions.csv', verbose: bool = False, display: bool = False,  seed: int = 42):
    """
    Perform cross-validated predictions using a LightGBM classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param str method: Classifier method (default LGBM)
    :param bool balanced: Whether to use class weights to balance the classes.
    :param bool saveflag: Whether to save the predictions to a CSV file.
    :param str or None outfile: File name for saving predictions.
    :param bool verbose: Whether to print verbose information.
    :param bool display: Whether to display a confusion matrix plot.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions, single measures and label predictions
    :rtype: Tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame)

    :example
 
    .. code-block:: python

        # Example usage
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        result, _, _ = predict_cv(X_data, Y_data, n_splits=5, balanced=True, saveflag=False, outfile=None, verbose=True, display=True, seed=42)
    """
    methods = {'RF': RandomForestClassifier, 'LGBM': LGBMClassifier}

    # silent twdm if no verbosity
    #if not verbose: 
    #    def notqdm(iterable, *args, **kwargs): return iterable
    #    tqdm = notqdm
    # get the list of genes
    genes = Y.index

    # Encode target variable labels
    encoder = LabelEncoder()
    X = X.values
    y = encoder.fit_transform(Y.values.ravel())

    # Display class information
    distrib = Counter(y)
    classes_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    if verbose: print(f'{classes_mapping}\n{Y.value_counts()}')

    # Set random seed
    set_seed(seed)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize classifier
    #clf = LGBMClassifier(class_weight='balanced', verbose=-1) if balanced else LGBMClassifier(verbose=-1)
    clf = methods[method](class_weight='balanced') if balanced else methods[method]()

    nclasses = len(np.unique(y))
    mm = np.array([], dtype=np.int64)
    gg = np.array([])
    yy = np.array([], dtype=np.int64)
    predictions = np.array([], dtype=np.int64)
    probabilities = np.array([])

    # Columns for result summary
    columns_names = ["ROC-AUC", "Accuracy", "BA", "Sensitivity", "Specificity", "MCC", 'CM']
    scores = pd.DataFrame()

    if verbose:
        print(f'Classification with {method}...')

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=f"{n_splits}-fold", disable=not verbose)):
        train_x, train_y, test_x, test_y = X[train_idx], y[train_idx], X[test_idx], y[test_idx],
        mm = np.concatenate((mm, test_idx))
        probs = clf.fit(train_x, train_y).predict_proba(test_x)
        preds = np.argmax(probs, axis=1)
        gg = np.concatenate((gg, genes[test_idx]))
        yy = np.concatenate((yy, test_y))
        cm = confusion_matrix(test_y, preds)
        predictions = np.concatenate((predictions, preds))
        probabilities = np.concatenate((probabilities, probs[:, 0]))

        # Calculate and store evaluation metrics for each fold
        roc_auc = roc_auc_score(test_y, probs[:, 1]) if nclasses == 2 else roc_auc_score(test_y, probs, multi_class="ovr", average="macro")
        scores = pd.concat([scores, pd.DataFrame([[roc_auc,
                                                    accuracy_score(test_y, preds),
                                                    balanced_accuracy_score(test_y, preds),
                                                    cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                                                    cm[1, 1] / (cm[1, 0] + cm[1, 1]),
                                                    matthews_corrcoef(test_y, preds),
                                                    cm]],
                                                  columns=columns_names, index=[fold])],
                           axis=0)

    # Calculate mean and standard deviation of evaluation metrics
    df_scores = pd.DataFrame([f'{val:.4f}±{err:.4f}' for val, err in zip(scores.loc[:, scores.columns != "CM"].mean(axis=0).values,
                                                                     scores.loc[:, scores.columns != "CM"].std(axis=0))] +
                             [(scores[['CM']].sum()).values[0].tolist()],
                             columns=['measure'], index=scores.columns)

    # Display confusion matrix if requested
    if display:
        ConfusionMatrixDisplay(confusion_matrix=np.array(df_scores.loc['CM']['measure']), display_labels=encoder.inverse_transform(clf.classes_)).plot()

    # Create DataFrame for storing detailed predictions
    df_results = pd.DataFrame({'gene': gg, 'label': yy, 'prediction': predictions, 'probabilities': probabilities})
    df_results = df_results.set_index(['gene'])

    # Save detailed predictions to a CSV file if requested
    if saveflag:
        df_results.to_csv(outfile)

    # Return the summary statistics of cross-validated predictions, the single measures and the prediction results
    return df_scores, scores, df_results

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
