import pandas as pd
from sklearn.base import is_classifier, clone, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMClassifier
from tqdm import tqdm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier 
import numpy as np

class SplitVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Splitting And Voting Ensemble using a general Classifier set.

    :param classifier: base classifier of the ensemble.
    :type classifier: int
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
    def __init__(self, clf, n_voters:int=-1, voting:str='soft', n_jobs:int=-1, verbose:bool=False, random_state:int=42):
        # intialize ensemble ov voters
        self.clf = clf
        assert is_classifier(clf), "clf must be a valid classifier object" 
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_voters = n_voters
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
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        if all([isinstance(i, (int, np.integer))  for i in y]):
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)
        else:
            self.label_encoder_ = None
        self.classes_ = np.unique(y)

        unique, counts = np.unique(y, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_labels = np.array([unique[i] for i in sorted_indices])
        secondmaxlab = sorted_labels[-1 : ][0]
        maxlabel = sorted_labels[-2 : ][0]
        maxcount = counts[sorted_indices[0]]
        secondmaxcount = counts[sorted_indices[1]]

        if self.verbose:
            print(f"Majority {maxlabel} {maxcount}, 2nd major {secondmaxlab} {secondmaxcount}")

        # Separate majority and minority class
        all_index_ne = np.where(y == maxlabel)[0]
        index_e = np.where(y != maxlabel)[0]

        # check if auto splitting
        if self.n_voters <= 0:
            n_members = round(maxcount / secondmaxcount)
            if n_members > 0:
                self.n_voters = n_members
            else: 
                self.n_voters = 1

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
        return self.classes_[np.argmax(np.sum(probabilities, axis=0)/self.n_voters, axis=1)]

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
        return balanced_accuracy_score(y, self.predict(X).flatten())
    
class SplitVotingEnsembleLGBM(BaseEstimator, ClassifierMixin):
    """
    Voting Ensemble using LightGBM Classifiers.

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
    :param boosting_type: Boosting type for LightGBM.
    :type boosting_type: str
    :param learning_rate: Learning rate for LightGBM.
    :type learning_rate: float
    """
    def __init__(self, n_voters:int=-1, voting:str='soft', n_jobs:int=-1, verbose:bool=False, random_state:int=42,
                 boosting_type:str='gbdt', learning_rate:float=0.1, n_estimators:int=100,
                 **kwargs):
        # intialize ensemble ov voters
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_voters = n_voters
        self.learning_rate = learning_rate
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.base_estimator = LGBMClassifier(verbose=-1, random_state=random_state, boosting_type=boosting_type, learning_rate=learning_rate, n_estimators=n_estimators, **kwargs)
        self.estimators_ = []
    
    def __sklearn_clone__(self):
        """
        Clone the current estimator. This is a special method for scikit-learn compatibility.

        :return: A cloned instance of the current estimator.
        :rtype: VotingEnsembleLGBM
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
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        if all([isinstance(i, (int, np.integer))  for i in y]):
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)
        else:
            self.label_encoder_ = None
        self.classes_ = np.unique(y)

        unique, counts = np.unique(y, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_labels = np.array([unique[i] for i in sorted_indices])
        secondmaxlab = sorted_labels[-2 : ][0]
        maxlabel = sorted_labels[-1 : ][0]
        maxcount = counts[sorted_indices[0]]
        secondmaxcount = counts[sorted_indices[1]]

        if self.verbose:
            print(f"Majority {maxlabel} {maxcount}, 2nd major {secondmaxlab} {secondmaxcount}")

        # Separate majority and minority class
        all_index_ne = np.where(y == maxlabel)[0]
        index_e = np.where(y != maxlabel)[0]

        # check if auto splitting
        if self.n_voters <= 0:
            n_members = round(maxcount / secondmaxcount)
            if n_members > 0:
                self.n_voters = n_members
            else: 
                self.n_voters = 1

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
        return self.classes_[np.argmax(np.sum(probabilities, axis=0)/self.n_voters, axis=1)]

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
        return balanced_accuracy_score(y, self.predict(X).flatten())

    @property
    def feature_importances_(self):
        """
        Computer feature importances by averaging on ensemble of voters.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            for each variable feature importances is the mean of each voter
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs)(delayed(getattr)(voter, "feature_importances_") for voter in self.estimators_)
        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)