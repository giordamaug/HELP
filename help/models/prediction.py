import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from lightgbm import LGBMClassifier
from tqdm import tqdm
from tabulate import tabulate
from typing import List,Dict,Union,Tuple


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

def predict_cv(X, Y, n_splits=10, balanced=False, saveflag: bool = False, outfile: str = 'predictions.csv', verbose: bool = False, display: bool = False,  seed: int = 42):
    """
    Perform cross-validated predictions using a LightGBM classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param bool balanced: Whether to use class weights to balance the classes.
    :param bool saveflag: Whether to save the predictions to a CSV file.
    :param str or None outfile: File name for saving predictions.
    :param bool verbose: Whether to print verbose information.
    :param bool display: Whether to display a confusion matrix plot.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions.
    :rtype: pd.DataFrame

    :example
 
    .. code-block:: python

        # Example usage
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        result = predict_cv(X_data, Y_data, n_splits=5, balanced=True, saveflag=False, outfile=None, verbose=True, display=True, seed=42)
    """
    # get the list of genes
    genes = Y.index

    # Encode target variable labels
    encoder = LabelEncoder()
    X = X.values
    y = encoder.fit_transform(Y.values.ravel())

    # Display class information
    distrib = Counter(y)
    classes_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f'{classes_mapping}\n{Y.value_counts()}')

    # Set random seed
    set_seed(seed)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Initialize classifier
    clf = LGBMClassifier(class_weight='balanced', verbose=-1) if balanced else LGBMClassifier(verbose=-1)

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
        print(f'Classification with LightGBM...')

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=f"{n_splits}-fold")):
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
    df_results = pd.DataFrame({'gene': gg, 'label': yy, 'prediction': predictions})
    df_results = df_results.set_index(['gene'])

    # Save detailed predictions to a CSV file if requested
    if saveflag:
        df_results.to_csv(outfile)

    # Return the summary statistics of cross-validated predictions and the single measures
    return df_scores, scores
