import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Union, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import pandas as pd
import numpy as np

def scale_to_essentials(ge_fit, ess_genes, noness_genes):
    """
    Scales gene expression data to essential and non-essential genes.
  
    :param pd.DataFrame ge_fit: DataFrame containing the gene expression data.
    :param list ess_genes: List of essential genes for scaling.
    :param list noness_genes: List of non-essential genes for scaling.
  
    :return: Scaled gene expression data.
    :rtype: pd.DataFrame
  
    :example:

    .. code-block:: python
      
        scaled_data = scale_to_essentials(gene_expression_data, ess_genes_list, noness_genes_list)
    """
    essential_indices = ge_fit.index.isin(ess_genes)
    nonessential_indices = ge_fit.index.isin(noness_genes)

    scaled_ge_fit = ge_fit.apply(lambda x: (x - np.nanmedian(x[nonessential_indices]))
                                          / (np.nanmedian(x[nonessential_indices]) - np.nanmedian(x[essential_indices])) if np.nanmedian(x[essential_indices]) != 0 else 0, axis=0)

    return scaled_ge_fit


def feature_assemble(label_file: str, features: List[Dict[str, Union[str, bool]]] = [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std'}], 
                     colname: str="label", subsample: bool = False, seed: int = 1, fold: int = 4, saveflag: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    :param str label_file: Path to the label file.
    :param List[Dict[str, Union[str, bool]]] features: List of dictionaries specifying feature files and their processing options.
        Default is [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std'}].
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param bool subsample: Whether to subsample the data. Default is False.
    :param int seed: Random seed for reproducibility. Default is 1.
    :param int fold: Number of folds for subsampling. Default is 4.
    :param bool saveflag: Whether to save the assembled data to files. Default is False.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.

    :returns: Tuple containing the assembled features (X) and labels (Y) DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        
    :example:

    .. code-block:: python

        label_file = "path/to/label_file.csv"
        features = [{'fname': 'path/to/feature_file.csv', 'fixna': True, 'normalize': 'std'}]
        colname = "target_column"
        subsample = False
        seed = 1
        fold = 4
        saveflag = False
        verbose = False

        X, Y = feature_assemble(label_file, features, colname, subsample, seed, fold, saveflag, verbose)
    """

    # Load labels from the specified file
    if verbose:
        print(f"Loading {label_file}")
    
    lab_df = pd.read_csv(label_file, index_col=0)

    # Subsample the data if required (subsample majority class fild-times rispect the minority class)
    minlab = lab_df[colname].value_counts().nsmallest(1).index[0]
    maxlab = lab_df[colname].value_counts().nlargest(1).index[0]
    if verbose: print("Majority" , maxlab, lab_df[colname].value_counts()[maxlab], "minoriy", minlab, lab_df[colname].value_counts()[minlab])
    if subsample:
        #if lab_df[colname].value_counts()[maxlab] >= 4*lab_df[colname].value_counts()[minlab]:
            idxNE = lab_df[lab_df[colname] == maxlab].index[np.random.choice(len(lab_df[lab_df[colname] == maxlab]), fold * len(lab_df[lab_df[colname] == minlab]), replace=False)]
            idxRest = lab_df[(lab_df[colname] != maxlab) & ((lab_df[colname] != minlab))].index
            idxE = lab_df[lab_df[colname] == minlab].index
            if idxRest.size > 0:
                lab_df = pd.concat([lab_df.loc[idxNE], lab_df.loc[idxRest], lab_df.loc[idxE]], axis=0).sample(frac=1)
            else:
                lab_df = pd.concat([lab_df.loc[idxNE], lab_df.loc[idxE]], axis=0).sample(frac=1)
        #else:
        #    warnings.warn("Subsampling cannot be applied: majority class is less then 4 time the minority one.")

    # Common indices among labels and features
    idx_common = lab_df.index.values
    x = pd.DataFrame(index=lab_df.index)

    # Process each feature
    for feat in features:
        feat_df = pd.read_csv(feat['fname'], index_col=0)
        feattype = os.path.split(feat['fname'])[1]
        feat_df.index = feat_df.index.map(str)

        # Handle missing values if required
        if verbose:
            cntnan = feat_df.isna().sum().sum()
            print(f"[{feattype}] found {cntnan} Nan...")
        if feat['fixna']:
            if verbose:
                print(f"[{feattype}] Fixing NaNs with mean ...")
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

def feature_assemble_df(lab_df: pd.DataFrame, features: List[Dict[str, Union[str, bool]]] = [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std'}], 
                     colname: str="label", subsample: bool = False, seed: int = 1, fold: int = 4, saveflag: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    :param str label_file: Path to the label file.
    :param List[Dict[str, Union[str, bool]]] features: List of dictionaries specifying feature files and their processing options.
        Default is [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std'}].
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param bool subsample: Whether to subsample the data. Default is False.
    :param int seed: Random seed for reproducibility. Default is 1.
    :param int fold: Number of folds for subsampling. Default is 4.
    :param bool saveflag: Whether to save the assembled data to files. Default is False.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.

    :returns: Tuple containing the assembled features (X) and labels (Y) DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        
    :example:

    .. code-block:: python

        label_file = "path/to/label_file.csv"
        features = [{'fname': 'path/to/feature_file.csv', 'fixna': True, 'normalize': 'std'}]
        colname = "target_column"
        subsample = False
        seed = 1
        fold = 4
        saveflag = False
        verbose = False

        X, Y = feature_assemble(label_file, features, colname, subsample, seed, fold, saveflag, verbose)
    """

    # Subsample the data if required (subsample majority class fild-times rispect the minority class)
    minlab = lab_df[colname].value_counts().nsmallest(1).index[0]
    maxlab = lab_df[colname].value_counts().nlargest(1).index[0]
    if verbose: print("Majority" , maxlab, lab_df[colname].value_counts()[maxlab], "minoriy", minlab, lab_df[colname].value_counts()[minlab])
    if subsample:
        #if lab_df[colname].value_counts()[maxlab] >= 4*lab_df[colname].value_counts()[minlab]:
            idxNE = lab_df[lab_df[colname] == maxlab].index[np.random.choice(len(lab_df[lab_df[colname] == maxlab]), fold * len(lab_df[lab_df[colname] == minlab]), replace=False)]
            idxRest = lab_df[(lab_df[colname] != maxlab) & ((lab_df[colname] != minlab))].index
            idxE = lab_df[lab_df[colname] == minlab].index
            if idxRest.size > 0:
                lab_df = pd.concat([lab_df.loc[idxNE], lab_df.loc[idxRest], lab_df.loc[idxE]], axis=0).sample(frac=1)
            else:
                lab_df = pd.concat([lab_df.loc[idxNE], lab_df.loc[idxE]], axis=0).sample(frac=1)
        #else:
        #    warnings.warn("Subsampling cannot be applied: majority class is less then 4 times the minority one.")
            if verbose:
                print(f'Subsampling with factor 1:{fold}')


    # Common indices among labels and features
    idx_common = lab_df.index.values
    x = pd.DataFrame(index=lab_df.index)

    # Process each feature
    for feat in features:
        feat_df = pd.read_csv(feat['fname'], index_col=0)
        feattype = os.path.split(feat['fname'])[1]
        feat_df.index = feat_df.index.map(str)

        # Handle missing values if required
        if verbose:
            cntnan = feat_df.isna().sum().sum()
            print(f"[{feattype}] found {cntnan} Nan...")
        if feat['fixna']:
            if verbose:
                print(f"[{feattype}] Fixing NaNs with mean ...")
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