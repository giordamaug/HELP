from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np
import pandas as pd

def imputer_knn(df: pd.DataFrame, n_neighbors: int=5, missing_values=np.nan, weights: str="uniform") -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    n_neighbors : int, optional, default=5
        Number of neighbors to consider for imputation using KNN.
    missing_values : any, optional, default=np.nan
        The placeholder for missing values in the input DataFrame.
    weights : str, optional, default="uniform"
        Weight function used in prediction during KNN imputation.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using KNN.

    Example
    -------
    >>> from sklearn.impute import KNNImputer
    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create a DataFrame with missing values
    >>> data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
    >>> df = pd.DataFrame(data)

    >>> # Impute missing values using KNN imputation
    >>> result = imputer_knn(df, n_neighbors=3, missing_values=np.nan, weights="distance")
    """

    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values, weights=weights)
    dfout = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return dfout


def imputer_mean(df: pd.DataFrame, missing_values=np.nan, strategy: str='mean') -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using mean imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    missing_values : any, optional, default=np.nan
        The placeholder for missing values in the input DataFrame.
    strategy : str, optional, default='mean'
        Imputation strategy, e.g., 'mean', 'median', 'most_frequent'.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed using mean imputation.

    Example
    -------
    >>> from sklearn.impute import SimpleImputer
    >>> import pandas as pd

    >>> # Create a DataFrame with missing values
    >>> data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
    >>> df = pd.DataFrame(data)

    >>> # Impute missing values using mean imputation
    >>> result = imputer_mean(df, missing_values=np.nan, strategy='mean')
    """

    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    dfout = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return dfout
