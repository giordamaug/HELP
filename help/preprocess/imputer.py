from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utility.selection import select_cell_lines

def imputer_knn_group(df: pd.DataFrame, df_map: pd.DataFrame, line_group: str='OncotreeLineage', line_col: str='ModelID',n_neighbors: int=5, missing_values=np.nan, weights: str="uniform", verbose: bool=False) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame grouped by specified lineages using K-Nearest Neighbors (KNN) imputation.

    :param pd.DataFrame df: Input DataFrame with missing values.
    :param pd.DataFrame df_map: DataFrame mapping cell lines to lineages.
    :param str line_group: Column specifying the grouping of cell lines (lineages). Default is 'OncotreeLineage'.
    :param str line_col: Column containing cell line identifiers. Default is 'ModelID'.
    :param int n_neighbors: Number of neighbors to consider for imputation using KNN. Default is 5.
    :param any missing_values: The placeholder for missing values in the input DataFrame. Default is np.nan.
    :param str weights: Weight function used in prediction during KNN imputation. Default is "uniform".
    :param bool verbose: If True, print progress information. Default is False.

    :return: DataFrame with missing values imputed using KNN grouped by lineages.
    
    :example:

    .. code-block:: python

        from sklearn.impute import KNNImputer
        import pandas as pd

        # Create a DataFrame with missing values
        data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
        df = pd.DataFrame(data)

        # Create a mapping DataFrame
        data_map = {'ModelID': ['M1', 'M2', 'M3', 'M4'], 'OncotreeLineage': ['L1', 'L2', 'L1', 'L2']}
        df_map = pd.DataFrame(data_map)

        # Impute missing values using KNN imputation grouped by lineages
        result = imputer_knn_group(df, df_map, line_group='OncotreeLineage', line_col='ModelID', n_neighbors=3, missing_values=np.nan, weights="distance", verbose=True)
    """
    tissuel = np.unique(df_map[df_map[line_col].isin(df.columns.values)][line_group].values)
    dfout = pd.DataFrame()
    if verbose:
        print(f"Imputation groups {tissuel}...")
        tissuel = tqdm(tissuel)
    for tissue in tissuel:
        cell_lines = select_cell_lines(df, df_map, [tissue], line_group=line_group, line_col=line_col, nested = False)
        imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values, weights=weights)
        df_tissue = df[cell_lines]
        df_tissue = pd.DataFrame(imputer.fit_transform(df_tissue), columns=df_tissue.columns, index=df_tissue.index)
        dfout = pd.concat([dfout, df_tissue], axis=1)
    return dfout


def imputer_knn(df: pd.DataFrame, n_neighbors: int=5, missing_values=np.nan, weights: str="uniform") -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation.

    :param pd.DataFrame df: Input DataFrame with missing values.
    :param int n_neighbors: Number of neighbors to consider for imputation using KNN.
        Optional, default is 5.
    :param any missing_values: The placeholder for missing values in the input DataFrame.
        Optional, default is np.nan.
    :param str weights: Weight function used in prediction during KNN imputation.
        Optional, default is "uniform".

    :returns pd.DataFrame: DataFrame with missing values imputed using KNN.

    :example:

    .. code-block:: python

    from sklearn.impute import KNNImputer
    import numpy as np
    import pandas as pd

    # Create a DataFrame with missing values
    data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
    df = pd.DataFrame(data)

    # Impute missing values using KNN imputation
    result = imputer_knn(df, n_neighbors=3, missing_values=np.nan, weights="distance")
    """

    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values, weights=weights)
    dfout = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return dfout


def imputer_mean(df: pd.DataFrame, missing_values=np.nan, strategy: str='mean') -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using mean imputation.

    :param pd.DataFrame df: Input DataFrame with missing values.
    :param any missing_values: The placeholder for missing values in the input DataFrame.
        Optional, default is np.nan.
    :param str strategy: Imputation strategy, e.g., 'mean', 'median', 'most_frequent'.
        Optional, default is "mean".

    :return pd.DataFrame: DataFrame with missing values imputed using mean imputation.

    :example

    .. code-block:: python

    from sklearn.impute import SimpleImputer
    import pandas as pd

    # Create a DataFrame with missing values
    data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
    df = pd.DataFrame(data)

    # Impute missing values using mean imputation
    result = imputer_mean(df, missing_values=np.nan, strategy='mean')
    """

    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    dfout = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return dfout
