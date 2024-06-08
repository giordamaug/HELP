import os
import pandas as pd
import numpy as np
import random
from sklearn.base import is_classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from collections import Counter
from lightgbm import LGBMClassifier
from tqdm import tqdm
from typing import List,Dict,Union,Tuple
from sklearn.base import is_classifier, clone
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier 
import numpy as np
from imblearn.over_sampling import SMOTE

class SplitVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Splitting And Voting Ensemble using a general Classifier set.

    :param n_voters: Number of voters in the ensemble.
    :type n_voters: int
    :param voting: Voting strategy ('soft' or 'hard').
    :type voting: str
    :param n_jobs: Number of jobs to run in parallel.
    :type n_jobs: int
    :param verbose: If True, prints progress messages.
    :type verbose: bool
    :param random_state: Seed for random number generator.
    :type random_state: int
    """
    def __init__(self, clf, n_voters=10, voting='soft', n_jobs=-1, verbose=False, random_state=42):
        # intialize ensemble ov voters
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_voters = n_voters
        assert is_classifier(clf), "clf must be a valid classifier object" 
        self.base_estimator = clf 
        self.estimators_ = []
    
    def __sklearn_clone__(self):
        """
        Clone the current estimator. This is a special method for scikit-learn compatibility.

        :return: A cloned instance of the current estimator.
        :rtype: SplitVotingEnsemble
        """
        return self

    def _fit_single_estimator(self, i, X, y, index_ne, index_e):
        """
        Private function used to fit a single estimator within a job.

        :param i: Index of the estimator.
        :type i: int
        :param X: Training data.
        :type X: np.ndarray
        :param y: Target values.
        :type y: np.ndarray
        :param index_ne: Indices of non-event class samples.
        :type index_ne: np.ndarray
        :param index_e: Indices of event class samples.
        :type index_e: np.ndarray
        :return: Fitted estimator.
        :rtype: LGBMClassifier
        """
        df_X = np.append(X[index_ne], X[index_e], axis=0)
        df_y = np.append(y[index_ne], y[index_e], axis=0)
        clf = clone(self.base_estimator)
        clf.fit(df_X, df_y)
        return clf
    
    def fit(self, X, y):
        """
        Fit the ensemble of LightGBM classifiers.

        :param X: Training data.
        :type X: pd.DataFrame or np.ndarray
        :param y: Target values.
        :type y: pd.Series or np.ndarray
        :return: Fitted instance of the class.
        :rtype: VotingEnsembleLGBM
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        encoder = LabelEncoder()
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        y = encoder.fit_transform(y)
        self.classes_ = np.unique(y)

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
        """
        Predict class probabilities for X.

        :param X: Input data.
        :type X: pd.DataFrame or np.ndarray
        :param y: Not used, present for API consistency by convention.
        :type y: None
        :return: Predicted class probabilities.
        :rtype: np.ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.sum(probabilities, axis=0)/self.n_voters
    
    def predict(self, X, y=None):
        """
        Predict class labels for X.

        :param X: Input data.
        :type X: pd.DataFrame or np.ndarray
        :param y: Not used, present for API consistency by convention.
        :type y: None
        :return: Predicted class labels.
        :rtype: np.ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        probabilities = np.array([self.estimators_[i].predict_proba(X) for i in range(self.n_voters)])
        return np.argmax(np.sum(probabilities, axis=0)/self.n_voters, axis=1)

    def score(self, X, y):
        """
        Return the balanced accuracy score on the given test data and labels.

        :param X: Test data.
        :type X: pd.DataFrame or np.ndarray
        :param y: True labels for X.
        :type y: pd.Series or np.ndarray
        :return: Balanced accuracy score.
        :rtype: float
        """
        return balanced_accuracy_score(y, (self.predict_proba(X) > 0.5).flatten())