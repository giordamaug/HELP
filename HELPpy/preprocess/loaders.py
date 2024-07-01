import pandas as pd
import os
from typing import List, Dict, Tuple, Union, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..utility.utils import pandas_readcsv
import pandas as pd
import numpy as np

def load_features(filenames: List[str] = [], fixnans: List[bool] = [], normalizes:List[str] = [], constrms:List[bool] = [],
                  verbose: bool = False, show_progress: bool = False) -> pd.DataFrame:
    """
    Load and assemble features for machine learning tasks.

    :param List[str] features: List of feature filepaths
    :param List[str] fixnans: List of booleans to enable fix of nan
    :param List[str] normalizes: List of str to set normalization (std|max|None)
    :param List[str] constrms: List of booleans to enable constant renoval
    :param int seed: Random seed for reproducibility. Default is 1.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.
    :param bool show_progress: Whether to print progress bar while loading file. Default is False.

    :returns: Tuple containing the assembled features (X) and labels (Y) DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        
    :example:

    .. code-block:: python

        seed = 1
        verbose = False

        X, Y = load_features(['path/to/feature_file1.csv', 'path/to/feature_file2.csv'], fixnans=[True, False], seed, verbose)
    """

    # Common indices among labels and features
    x = pd.DataFrame()

    # Process each feature file
    for f,fixna,norm,crm in zip(filenames, fixnans, normalizes, constrms):
        feat_df = pandas_readcsv(f, chunksize=1024, index_col=0, descr=f'{os.path.basename(f)}', disabled=not show_progress)
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
        else:
            if verbose:
                print(f"[{fname}] No Nan fixing...")

        # Remove contsnat features
        constfeatures = feat_df.columns[feat_df.nunique() <= 1].values
        if verbose:
            print(f"[{fname}] found {len(constfeatures)} Nan...")
        if crm:
            if verbose:
                print(f"[{fname}] Removing {len(constfeatures)} constant features ...")
            feat_df = feat_df.drop(constfeatures, axis=1)
        else:
            if verbose:
                print(f"[{fname}] No constant feature removal...")

        # Normalize features
        if norm == 'std':
            scaler = StandardScaler()
            if verbose:
                print(f"[{fname}] Normalization with {norm} ...")
            feat_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index, columns=feat_df.columns)
        elif norm == 'max':
            scaler = MinMaxScaler()
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