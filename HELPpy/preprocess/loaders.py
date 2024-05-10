import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Union, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

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


def feature_assemble(label_file: str, features: List[Dict[str, Union[str, bool]]] = [{'fname': 'BIO.csv', 'fixna' : False, 'normalize': 'std', 'nchunks': 1}], 
                     colname: str="label", subsample: bool = False, seed: int = 1, fold: int = 4, verbose: bool = False, show_progress: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    :param str label_file: Path to the label file.
    :param List[Dict[str, Union[str, bool]]] features: List of dictionaries specifying feature files and their processing options.
        Default is [{'fname': 'BIO.csv', 'fixna' : False, 'normalize': 'std', 'nchunks': 1}].
        'fname' : str, filename of attributes (in CSV format)
        'fixna' : bool, flag to enable fixing missing values with mean in column
        'normalize': std|max|None, normalization option (z-score, minmax, or no normalization)
        'nchunks': int, number of chunck the attribute file is split   
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param bool subsample: Whether to subsample the data. Default is False.
    :param int seed: Random seed for reproducibility. Default is 1.
    :param int fold: Number of folds for subsampling. Default is 4.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.
    :param bool show_progress: Whether to print progress bar while loading file. Default is False.

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
        verbose = False

        X, Y = feature_assemble(label_file, features, colname, subsample, seed, fold, verbose)
    """

    # Load labels from the specified file
    if verbose:
        print(f"Loading label file {label_file}")
    
    lab_df = pd.read_csv(label_file, index_col=0)

    # Subsample the data if required (subsample majority class fild-times rispect the minority class)
    minlab = lab_df[colname].value_counts().nsmallest(1).index[0]
    maxlab = lab_df[colname].value_counts().nlargest(1).index[0]
    if verbose: print("Majority" , maxlab, lab_df[colname].value_counts()[maxlab], "minority", minlab, lab_df[colname].value_counts()[minlab])
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
        # check for chunk splits
        if 'nchunks' in feat and type(feat['nchunks']) == int and feat['nchunks'] > 1:
            dfl = []
            for id in tqdm(range(feat['nchunks']), desc="Loading file in chunks", disable=not show_progress):
               filename, file_ext = os.path.splitext(feat['fname'])
               dfl += [pd.read_csv(f"{filename}_{id}{file_ext}", index_col=0)]
            feat_df = pd.concat(dfl)
        else:
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
            feat_df = feat_df.fillna(feat_df.mean())

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

    # Return the assembled features (X) and labels (Y)
    return x, lab_df[[colname]].loc[idx_common]

def feature_assemble_df(lab_df: pd.DataFrame, features: List[Dict[str, Union[str, bool]]] = [{'fname': 'bio+gtex.csv', 'fixna' : True, 'normalize': 'std', 'nchunks': 1}], 
                     colname: str="label", subsample: bool = False, seed: int = 1, fold: int = 4, verbose: bool = False, show_progress: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    :param pd.DataFrame lab_df: DataFrame of labels (in column named colnname).
    :param List[Dict[str, Union[str, bool]]] features: List of dictionaries specifying feature files and their processing options.
        Default is [{'fname': 'BIO.csv', 'fixna' : False, 'normalize': 'std', 'nchunks': 1}].
        'fname' : str, filename of attributes (in CSV format)
        'fixna' : bool, flag to enable fixing missing values with mean in column
        'normalize': std|max|None, normalization option (z-score, minmax, or no normalization)
        'nchunks': int, number of chunck the attribute file is split   
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param bool subsample: Whether to subsample the data. Default is False.
    :param int seed: Random seed for reproducibility. Default is 1.
    :param int fold: Number of folds for subsampling. Default is 4.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.
    :param bool show_progress: Whether to print progress bar while loading file. Default is False.

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
        verbose = False

        df_label = pd.read_csv("label_file.csv2, index_col=0)
        X, Y = feature_assemble_df(df_label, colname='label', features, colname, subsample, seed, fold, verbose)
    """

    # Subsample the data if required (subsample majority class fild-times rispect the minority class)
    minlab = lab_df[colname].value_counts().nsmallest(1).index[0]
    maxlab = lab_df[colname].value_counts().nlargest(1).index[0]
    if verbose: print("Majority" , maxlab, lab_df[colname].value_counts()[maxlab], "minority", minlab, lab_df[colname].value_counts()[minlab])
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

    # Process each feature file
    for feat in features:
        # check for chunk splits
        if 'nchunks' in feat and type(feat['nchunks']) == int and feat['nchunks'] > 1:
            dfl = []
            for id in tqdm(range(feat['nchunks']), desc="Loading file in chunks", disable=not show_progress):
               filename, file_ext = os.path.splitext(feat['fname'])
               dfl += [pd.read_csv(f"{filename}_{id}{file_ext}", index_col=0)]
            feat_df = pd.concat(dfl)
        else:
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
            feat_df = feat_df.fillna(feat_df.mean())

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

    # Return the assembled features (X) and labels (Y)
    return x, lab_df[[colname]].loc[idx_common]

def ipy_feature_assemble_df(lab_df: pd.DataFrame, features: List[Dict[str, Union[str, bool]]] = [], 
                     colname: str="label", verbose: bool = False, show_progress: bool = False, progressbar=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble features and labels for machine learning tasks.

    :param pd.DataFrame lab_df: DataFrame of labels (in column named colnname).
    :param List[Dict[str, Union[str, bool]]] features: List of dictionaries specifying feature files and their processing options.
        Default is [{'fname': 'BIO.csv', 'fixna' : False, 'normalize': 'std', 'nchunks': 1}].
        'fname' : str, filename of attributes (in CSV format)
        'fixna' : bool, flag to enable fixing missing values with mean in column
        'normalize': std|max|None, normalization option (z-score, minmax, or no normalization)
        'nchunks': int, number of chunck the attribute file is split   
    :param str colname: Name of the column in the label file to be used as the target variable. Default is "label".
    :param int seed: Random seed for reproducibility. Default is 1.
    :param bool verbose: Whether to print verbose messages during processing. Default is False.
    :param bool show_progress: Whether to print progress bar while loading file. Default is False.

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
        verbose = False

        df_label = pd.read_csv("label_file.csv2, index_col=0)
        X, Y = ipy_feature_assemble_df(df_label, colname='label', features, colname, subsample, seed, fold, verbose)
    """

    # Common indices among labels and features
    idx_common = lab_df.index.values
    x = pd.DataFrame(index=lab_df.index)

    # Process each feature file
    for feat in features:
        # check for chunk splits
        if 'nchunks' in feat and type(feat['nchunks']) == int and feat['nchunks'] > 1:
            dfl = []
            for id in tqdm(range(feat['nchunks']), desc="Loading file in chunks", disable=not show_progress):
               filename, file_ext = os.path.splitext(feat['fname'])
               dfl += [ipy_readcsv(f"{filename}_{id}{file_ext}", index_col=0, progressbar=progressbar, descr=feat['fname'])]
               #dfl += [pd.read_csv(f"{filename}_{id}{file_ext}", index_col=0)]
            feat_df = pd.concat(dfl)
        else:
            feat_df = ipy_readcsv(feat['fname'], index_col=0, progressbar=progressbar, descr=os.path.basename(feat['fname']))
        progressbar.layout.display = None
        feattype = os.path.split(feat['fname'])[1]
        feat_df.index = feat_df.index.map(str)

        # Handle missing values if required
        if verbose:
            cntnan = feat_df.isna().sum().sum()
            print(f"[{feattype}] found {cntnan} Nan...")
        if feat['fixna']:
            if verbose:
                print(f"[{feattype}] Fixing NaNs with mean ...")
            feat_df = feat_df.fillna(feat_df.mean())

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

    # Return the assembled features (X) and labels (Y)
    return x, lab_df[[colname]].loc[idx_common]

def ipy_readcsv(filename, chunksize=50, sep=',', index_col=None, comment='#', progressbar=None, descr:str=None):
    # Get number of lines in file.
    with open(filename, 'r') as fp:
        try:
            has_headings = csv.Sniffer().has_header(fp.read(1024))
            lines = len(fp.readlines())-1
        except csv.Error:
            # The file seems to be empty
            lines = len(fp.readlines())
    progressbar.layout.display = None
    if descr is not None:
        progressbar.description = descr
    progressbar.value=0
    progressbar.min=0
    progressbar.max=lines
    # Read file in chunks, updating progress bar after each chunk.
    listdf = []
    for i,chunk in enumerate(pd.read_csv(filename,chunksize=chunksize, index_col=index_col, comment=comment, sep=sep)):
        listdf.append(chunk)
        progressbar.value += chunk.shape[0]
    df = pd.concat(listdf,ignore_index=False)
    progressbar.layout.display = 'none'
    return df

def ipy_writecsv(filename, df: pd.DataFrame, chunksize=10, index=False, sep=',', progressbar=None):
    # Get number of lines in file.
    progressbar.layout.display = None
    progressbar.value=0
    progressbar.min=0
    progressbar.max=len(df)
    # Write file in chunks, updating progress bar after each chunk.
    num_chunks = len(df) // chunksize + 1
    for i, chunk in enumerate(np.array_split(df, num_chunks)):
        mode = 'w' if i == 0 else 'a'
        chunk.to_csv(filename, index=index, mode=mode, sep=sep)
        progressbar.value += chunk.shape[0]
    progressbar.layout.display = 'none'
    return df