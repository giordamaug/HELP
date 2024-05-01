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
from sklearn.base import is_classifier, clone
from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier 
import numpy as np
from tqdm import tqdm

class VotingSplitClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_voters=10, voting='soft', n_jobs=-1, verbose=False, random_state=42, **kwargs):
        self.kwargs = kwargs
        # intialize ensemble ov voters
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_voters = n_voters
        self.estimators_ = [LGBMClassifier(**kwargs, verbose=-1, random_state=random_state) for i in range(n_voters)]
        pass
    
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
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "Only array input!"
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

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_estimator)(i,X, y, index_ne, index_e) for i,index_ne in enumerate(tqdm(splits, desc=f"{self.n_voters}-voter", disable = not self.verbose)))
        return self
    
    def predict_proba(self, X, y=None):
        # Find the majority and minority class
        assert isinstance(X, np.ndarray), "Only array input!"
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.sum(probabilities, axis=0)/self.n_voters
    
    def predict(self, X, y=None):
        assert isinstance(X, np.ndarray), "Only array input!"
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.argmax(np.sum(probabilities, axis=0)/self.n_voters, axis=1)
    
def set_seed(seed=1):
    """
    Set random and numpy random seed for reproducibility

    :param int seed: inistalization seed

    :returns: None.
    """
    random.seed(seed)
    np.random.seed(seed)

def k_fold_cv(X, Y, estimator, n_splits=10, saveflag: bool = False, outfile: str = 'predictions.csv', verbose: bool = False, show_progress: bool = False, display: bool = False,  seed: int = 42):
    """
    Perform cross-validated predictions using a classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param estimator object: Classifier method (must have fit, predict, predict_proba methods)
    :param bool balanced: Whether to use class weights to balance the classes.
    :param bool saveflag: Whether to save the predictions to a CSV file.
    :param str or None outfile: File name for saving predictions.
    :param bool show_progress: Verbosity level for printing progress bar (default: False).
    :param bool verbose: Whether to print verbose information.
    :param bool display: Whether to display a confusion matrix plot.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions, single measures and label predictions
    :rtype: Tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame)

    :example:
 
    .. code-block:: python

        # Example usage
        from lightgbm import LGBMClassifier
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        clf = LGBMClassifier(random_state=0)
        df_scores, scores, predictions = k_fold_cv(df_X, df_y, clf, n_splits=5, verbose=True, display=True, seed=42)
    """
    assert is_classifier(estimator) and hasattr(estimator, 'fit') and callable(estimator.fit) and hasattr(estimator, 'predict_proba') and callable(estimator.predict_proba), "Bad estimator imput!"

    # get list of genes
    genes = Y.index

    # Encode target variable labels
    encoder = LabelEncoder()
    X = X.values
    y = encoder.fit_transform(Y.values.ravel())

    # Display class information
    classes_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    if verbose: print(f'{classes_mapping}\n{Y.value_counts()}')

    # Set random seed
    set_seed(seed)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

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
        print(f'Classification with {estimator.__class__.__name__}...')

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=f"{n_splits}-fold", disable=not show_progress)):
        train_x, train_y, test_x, test_y = X[train_idx], y[train_idx], X[test_idx], y[test_idx],
        mm = np.concatenate((mm, test_idx))
        # Initialize classifier
        clf = clone(estimator)
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

def predict_cv(X, Y, n_splits=10, method='LGBM', balanced=False, saveflag: bool = False, outfile: str = 'predictions.csv', verbose: bool = False, show_progress: bool = False, display: bool = False,  seed: int = 42):
    """
    Perform cross-validated predictions using a LightGBM classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param str method: Classifier method (default LGBM)
    :param bool balanced: Whether to use class weights to balance the classes.
    :param bool saveflag: Whether to save the predictions to a CSV file.
    :param str or None outfile: File name for saving predictions.
    :param bool show_progress: Verbosity level for printing progress bar (default: False).
    :param bool verbose: Whether to print verbose information.
    :param bool display: Whether to display a confusion matrix plot.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions, single measures and label predictions
    :rtype: Tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame)

    :example:
 
    .. code-block:: python

        # Example usage
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        result, _, _ = predict_cv(X_data, Y_data, n_splits=5, balanced=True, saveflag=False, outfile=None, verbose=True, display=True, seed=42)
    """
    methods = {'RF': RandomForestClassifier, 'LGBM': LGBMClassifier}

    # get list of genes
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
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=f"{n_splits}-fold", disable=not show_progress)):
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

def predict_cv_sv(X, Y, n_voters=1, n_splits=5, colname='label', balanced=False, seed=42, verbose=False, show_progress: bool = False):
    """
    Function to perform cross-validation with stratified sampling using LightGBM classifier 
    and a voting mechanism for binary classification. This function takes in features 
    (df_X) and labels (df_y) DataFrames for a classification problem, and performs cross-validation 
    with stratified sampling using LightGBM classifier. It then employs a voting mechanism to 
    handle imbalanced classes for binary classification tasks. Finally, it evaluates the predictions 
    and returns evaluation scores along with the predicted labels and probabilities.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_voters: Number of voters to split the majority class.
    :param int n_splits: Number of folds for cross-validation.
    :param str colname: Name of the column containing the labels.
    :param bool balanced: Whether to use class weights to balance the classes.
    :param int or None seed: Random seed for reproducibility.
    :param bool verbose: Whether to print verbose information.
    :param bool show_progress: Verbosity level for printing progress bar (default: False).

    :returns: 
        DataFrame scores: containing evaluation scores. 
        DataFrame predictions:  containing predicted labels and probabilities.
    :rtype: Tuple(pd.DataFrame,pd.DataFrame)
        
    :example:

    .. code-block:: python

        # Example usage
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        result, prediction = predict_cv_sv(X_data, Y_data, n_voters=10, n_splits=5, balanced=True, verbose=True, seed=42)
    """

    # Find the majority and minority class
    minlab = Y[colname].value_counts().nsmallest(1).index[0]
    maxlab = Y[colname].value_counts().nlargest(1).index[0]

    if verbose:
        print(f"Majority {maxlab} {Y[colname].value_counts()[maxlab]}, minority {minlab} {Y[colname].value_counts()[minlab]}")

    # Separate majority and minority class
    df_y_ne = Y[Y['label'] == maxlab]
    df_y_e = Y[Y['label'] != maxlab]

    # Split majority class among voters
    splits = np.array_split(df_y_ne, n_voters)

    # Initialize empty DataFrame for predictions
    predictions_ne = pd.DataFrame()
    predictions_e = pd.DataFrame(index=df_y_e.index)
    d = np.empty((len(df_y_e.index),), object)
    d[...] = [list() for _ in range(len(df_y_e.index))]
    predictions_e['probabilities'] = d
    predictions_e['label'] = np.array([0 for idx in df_y_e.index])
    predictions_e['prediction'] = np.array([np.nan for idx in df_y_e.index])

    # Perform cross-validation for each split
    for df_index_ne in splits:
        df_x = pd.concat([X.loc[df_index_ne.index], X.loc[df_y_e.index]])
        df_yy = pd.concat([Y.loc[df_index_ne.index], Y.loc[df_y_e.index]])
        
        _, _, preds = predict_cv(df_x, df_yy, n_splits=n_splits, method='LGBM', balanced=balanced, verbose=verbose, show_progress=show_progress, seed=seed)
        
        # Concatenate predictions for the minority class
        predictions_ne = pd.concat([predictions_ne, preds.loc[df_index_ne.index]])

        # Update probabilities for the minority class
        r = np.empty((len(df_y_e.index),), object)
        r[...] = [predictions_e.loc[idx]['probabilities'] + [preds.loc[idx]['probabilities']]  for idx in df_y_e.index]
        predictions_e['probabilities'] = r

    # Combine probabilities for the minority class
    predictions_e['prediction'] = predictions_e['probabilities'].map(lambda x: 0 if sum(x)/n_voters > 0.5 else 1)
    predictions_e['probabilities'] = predictions_e['probabilities'].map(lambda x: sum(x)/n_voters)

    # Combine predictions
    predictions = pd.concat([predictions_ne, predictions_e])

    # Evaluate predictions
    test_y = predictions['label'].values
    preds = predictions['prediction'].values
    probs = predictions['probabilities'].values
    cm = confusion_matrix(test_y, preds)
    
    # Calculate evaluation scores
    scores = pd.DataFrame([[roc_auc_score(test_y, 1-probs), accuracy_score(test_y, preds),
                            balanced_accuracy_score(test_y, preds),
                            cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                            cm[1, 1] / (cm[1, 0] + cm[1, 1]),
                            matthews_corrcoef(test_y, preds), cm]],
                          columns=["ROC-AUC", "Accuracy", "BA", "Sensitivity", "Specificity", "MCC", 'CM'], index=[seed])

    return scores, predictions
