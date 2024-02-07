import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Callable

def feature_assemble(label_file: str, features: List[Dict[str, Union[str, bool]]] = [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std'}], 
                     colname: str="label", subsample: bool = False, seed: int = 1, fold: int = 4, saveflag: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    Parameters
    ----------
    label_file : str
        Path to the label file.
    features : List[Dict[str, Union[str, bool]]]
        List of dictionaries specifying feature files and their processing options.
    colname : str
        Name of the column in the label file to be used as the target variable.
    subsample : bool, optional
        Whether to subsample the data.
    seed : int or None, optional
        Random seed for reproducibility.
    fold : int or None, optional
        Number of folds for subsampling.
    saveflag : bool, optional
        Whether to save the assembled data to files.
    verbose : bool, optional
        Whether to print verbose messages during processing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the assembled features (X) and labels (Y) DataFrames.
    """

    # Load labels from the specified file
    if verbose:
        print(f"Loading {label_file}")
    
    lab_df = pd.read_csv(label_file, index_col=0)

    # Subsample the data if required
    if subsample:
        idxNE = lab_df[lab_df[colname] == 'NE'].index[np.random.choice(len(lab_df[lab_df[colname] == 'NE']), fold * len(lab_df[lab_df[colname] == 'E']), replace=False)]
        idxE = lab_df[lab_df[colname] == 'E'].index
        lab_df = pd.concat([lab_df.loc[idxNE], lab_df.loc[idxE]], axis=0).sample(frac=1)

    # Common indices among labels and features
    idx_common = lab_df.index.values
    x = pd.DataFrame(index=lab_df.index)

    # Process each feature
    for feat in features:
        feat_df = pd.read_csv(feat['fname'], index_col=0)
        feattype = os.path.split(feat['fname'])[1]
        feat_df.index = feat_df.index.map(str)

        # Handle missing values if required
        if feat['fixna']:
            cntnan = feat_df.isna().sum().sum()
            if verbose:
                print(f"[{feattype}] found {cntnan} Nan...")
            feat_df = feat_df.fillna(x.mean())

        # Normalize features
        if feat['normalize'] == 'std':
            scaler = MinMaxScaler()
            if verbose:
                print(f"[{feattype}] Normalization with {feat['normalize']} ...")
            feat_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index, columns=feat_df.columns)
        elif feat['normalize'] == 'max':
            scaler = StandardScaler()
            if verbose:
                print(f"[{feattype}] Normalization with {feat['normalize']}...")
            feat_df = pd.DataFrame(scaler.fit_transform(feat_df), index=feat_df.index, columns=feat_df.columns)
        else:
            if verbose:
                print(f"[{feattype}] No normalization...")

        # Update common indices and concatenate features
        idx_common = np.intersect1d(idx_common, feat_df.index.values)
        x = pd.concat([x.loc[idx_common], feat_df.loc[idx_common]], axis=1)

    if verbose:
        print(f'{len(idx_common)} labeled genes over a total of {len(lab_df)}')
        print(f'{x.shape} data input')

    # Save the assembled data if required
    if saveflag:
        x.to_csv(os.path.join(work_dir, "X.csv"))
        lab_df[[colname]].loc[idx_common].to_csv(os.path.join(work_dir, "Y.csv"))

    # Return the assembled features (X) and labels (Y)
    return x, lab_df[[colname]].loc[idx_common]