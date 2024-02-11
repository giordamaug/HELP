import pandas as pd
from typing import Callable, List, Dict, Tuple
import statistics
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_multiotsu
import warnings
from tqdm import tqdm

def multi_threshold_with_nan_by_column(matrix, num_thresholds, algorithm='otsu'):
    # Apply thresholds and segmentation column-wise
    """
    The matrix quantization algorithm.

    :param np.ndarray matrix: Input matrix.
    :param int num_thresholds: number of quantized levels.
    :param str algorithm: quantization algorithm type 'otsu|'linspace' (default: 'otsu').

    :returns: The quantized array and the thresholds used.
    :rtype: Tuple[np.ndarray, c]

    :example:

    .. code-block:: python

        # Example usage
        data_matrix = np.array([[1.2, 2.1, 3.6, np.nan],
                                [4.5, 5.7, np.nan, 6.3],
                                [4.1, 8.4, 3.0, 10.2],
                                [7.7, np.nan, 9, 10],
                                [2.9, 12.5, 8.2, 1.0],
                                [1.1, 2.2, 9.0, np.nan]])

        num_thresholds = 3  # Adjust the number of thresholds as needed
        segmented_matrix, thres = multi_threshold_with_nan_by_column(data_matrix, num_thresholds, mode='otsu')
    """
    segmented_matrix = np.empty_like(matrix, dtype=float)
    segmented_matrix.fill(0)
    thresholds = []
    for col_idx in tqdm(range(matrix.shape[1])):
        col_digitize = np.empty_like(matrix[:,col_idx], dtype=int)
        col_digitize.fill(0)
        col = matrix[:, col_idx]
        valid_values = col[~np.isnan(col)]
        # if all NaN in columns, left it as it is
        if len(valid_values) == 0:
            segmented_matrix[:,col_idx] = col
            continue  # Skip if all values in the column are NaN
        if algorithm == 'otsu':
            cnt = len(set(valid_values))
            # if there are less distint valid_values than bins ... otsu raise an error.
            if cnt < num_thresholds:
                warnings.warn('Too fews distint values in column which is less then binarization values... using linspace mode!')
                thresh = np.linspace(np.nanmin(valid_values), np.nanmax(valid_values), num_thresholds + 1)[1:-1]
            else:
                thresh = threshold_multiotsu(valid_values, num_thresholds)
        elif algorithm == 'linspace':
            thresh = np.linspace(np.nanmin(valid_values), np.nanmax(valid_values), num_thresholds + 1)[1:-1]
        else:
            raise Exception("Thresholding method not supported")
        thresholds += [thresh]
        # Define a placeholder value (can be any value not present in the array)
        placeholder_value = -1
        # Replace NaN values with the placeholder
        col_no_nan = np.where(np.isnan(col), placeholder_value, col)
        # Perform digitization on the modified array
        for i, threshold in enumerate(thresh):
            col_digitize[(col_no_nan[:] > threshold)] = i + 1
        # Revert the placeholder values back to NaN
        col_digitize_with_nan = np.where(col_no_nan == placeholder_value, np.nan, col_digitize)
        segmented_matrix[:,col_idx] = col_digitize_with_nan
    return segmented_matrix, thresholds

def modemax(a: np.ndarray, reducefoo: Callable[[List[int]], int] = max) -> np.ndarray:
    """
    Computes the mode of an array along each row. In case of ex-aequo modes, return the value computed by reducefoo (default: max).

    :param np.ndarray a: Input 2D array.
    :param Callable[[List[int]], int] reducefoo: Reduction function (default: max).

    :returns: Mode values.
    :rtype: np.ndarray

    :example:

    .. code-block:: python

        # Example usage
        from help.models.labelling import modemax
        input_array = np.array([[np.nan, 2, 3, 3],
                                [2, np.nan, 4, 4],
                                [4, 5, 6, 6]])

        mode_values = modemax(input_array)
    """
    res = []
    m = a.max()
    for x in range(a.shape[0]):
        modes = statistics.multimode([x for x in a[x,:] if ~np.isnan(x)])
        if modes == []: 
            print(a[x,:])
            res += [m]
        else:
            res += [max(modes)] # if max(modes) is not np.nan else sorted(set(modes))[-2]]
    return np.array(res)

def labelling_core(df: pd.DataFrame, columns: List[str] = [], n_classes: int=2,
                  verbose: bool = False, labelnames: Dict[int, str] = {0: 'E', 1: 'NE'},
                  mode='flat-multi', algorithm='otsu', rowname: str = 'gene', colname: str = 'label') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Core function for HELP labelling algorithm.

    :param pd.DataFrame df: Input DataFrame.
    :param List[str] columns: List of column names.
    :param int n_classes: Number of classes for 'flat-multi' labelling mode, 
                          In 'two-by-two' mode the param is ignored: two binary labelling are computed
                          (default: 2).
    :param bool verbose: Verbosity level for printing information (default: False).
    :param Dict[int, str] labelnames: Dictionary mapping class labels to names (default: {0: 'E', 1: 'NE'}).
    :param str mode: quantization modes: 'flat-multi' - the quantization applies to all matrix (the mode on all rows give the labels).
                                         'two-by-two' - a binary quantization is done on all matrix (the mode on all rows give the binary labels), 
                                         then a binary quantization is applied only on 1-label rows (the mode on these rows give the binary labels 1 and 2) 
                                         (default: 'flat-multi').
    :param str algorithm: quantization algorithm type otsu|linspace (default: 'otsu'). 
    :param str rowname: Name of the DataFrame index (default: 'gene').
    :param str colname: Name of the label column (default: 'label').

    :returns: Output labelled DataFrame and quantized array.
    :rtype: Tuple[pd.DataFrame, np.ndarray]

    :example

    .. code-block:: python

        # Example usage
        from help.models.labelling import labelling_core
        input_df = pd.DataFrame(...)
        columns_list = ['feature1', 'feature2', 'feature3']
        output_df, quantized_array = labelling_core(input_df, columns=[], labelnames={0: 'E', 1: 'NE'}, n_classes, algorithm='otsu', mode='flat-multi')
    """
    # Extract data from the dataframe
    if columns == []:
        T = df.to_numpy()
    else:
        T = df[columns].to_numpy()

    if mode == 'two-by-two':
        if verbose: print("[two-by-two]: 1. Two-class labelling:") 
        # Perform quantization
        Q, Thr = multi_threshold_with_nan_by_column(T, 2, algorithm=algorithm)
        Labels2 = modemax(Q)
        if verbose: print(Labels2.shape)
        dfout =  pd.DataFrame(index=df.index)
        dfout.index.name = rowname
        dfout[colname] = Labels2
        if verbose: print("[two-by-two]: 2. Two-class labelling on 1-label rows:")
        NE_genes = dfout[dfout[colname]==1].index
        if columns == []:
            TNE = df.loc[NE_genes].to_numpy()
        else:
            TNE = df[columns].loc[NE_genes].to_numpy()
        NumberOfClasses = 2
        QNE, ThrNE = multi_threshold_with_nan_by_column(TNE, 2, algorithm=algorithm)
        Labels = modemax(QNE)
        if verbose: print(Labels.shape)
        dfout2 =  pd.DataFrame(index=NE_genes)
        dfout2.index.name = rowname
        dfout2[colname] = Labels
        dfout2 = dfout2.replace({0: 1, 1: 2})
        dfout.loc[dfout.index.isin(dfout2.index), [colname]] = dfout2[[colname]]
        dfout = dfout.replace(labelnames)
    elif mode == 'flat-multi':
        if verbose: print("[flat-multi]: 1. multi-class labelling:") 
        # Perform quantization
        Q, Thr = multi_threshold_with_nan_by_column(T, n_classes, algorithm=algorithm)
        Labels = modemax(Q)
        dfout =  pd.DataFrame(index=df.index)
        dfout.index.name = rowname
        dfout[colname] = Labels
    else:
        raise Exception("Labelling mode not supported!")
    return dfout
    
def labelling(df: pd.DataFrame, columns: List[List[str]] = [], n_classes: int=2, 
             verbose: bool = False, labelnames: Dict[int, str] = {1 : 'NE', 0: 'E'},
             mode='flat-multi', rowname: str = 'gene', colname: str = 'label', algorithm='otsu') -> pd.DataFrame:
    """
    Main function for HELP labelling algorithm.

    :param pd.DataFrame df: Input DataFrame.
    :param List[List[str]] columns: List of column names for partitioning (default: []).
    :param bool three_class: Flag for three-class labeling (default: False).
    :param bool verbose: Verbosity level for printing information (default: False).
    :param Dict[int, str] labelnames: Dictionary mapping class labels to names (default: {}).
    :param str mode: quantization modes: 'flat-multi' - the quantization applies to all matrix (the mode on all rows give the labels).
                                         'two-by-two' - a binary quantization is done on all matrix (the mode on all rows give the binary labels), 
                                         then a binary quantization is applied only on 1-label rows (the mode on these rows give the binary labels 1 and 2) 
                                         (default: 'flat-multi').
    :param str rowname: Name of the DataFrame index (default: 'gene').
    :param str colname: Name of the label column (default: 'label').

    :returns: Output DataFrame with labels.
    :rtype: pd.DataFrame

    :example

    .. code-block:: python

        # Example usage
        from help.models.labelling import Help
        input_df = pd.DataFrame(...)
        output_df = Help.labelling(input_df, columns=[], n_classes=2, labelnames={0: 'E', 1: 'NE'}, algorithm='otsu', mode='flat-multi')
    """
    if mode=='two-by-two': n_classes = 3
    assert len(labelnames) == n_classes, "Label dictionary not same size of no. of classes!"
    if all(isinstance(sub, list) for sub in columns) and len(columns) > 0:      # use mode of mode across tissues
        if verbose: print(f'performing mode of mode on {n_classes}-class labelling ({mode}).')
        L_tot = np.empty(shape=(len(df), 0))
        for lines in columns:
            labels = labelling_core(df, columns=lines, verbose=verbose, mode=mode,  
                                  labelnames=labelnames, rowname=rowname, colname=colname, 
                                  n_classes=n_classes, algorithm=algorithm)
            labels = labels.replace(dict(map(reversed, labelnames.items()))).infer_objects(copy=False)
            L_tot = np.hstack((L_tot, labels.values))
        # Execute mode on each tissue and sort'em
        modeOfmode = modemax(L_tot)
        dfout =  pd.DataFrame(index=df[sum(columns, [])].index)
        dfout.index.name = rowname
        dfout[colname] = modeOfmode
        dfout = dfout.replace(labelnames)
    elif any(isinstance(sub, list) for sub in columns) and len(columns) > 0:
        raise Exception("Wrong columns partition format.")
    else:
        if verbose: print(f'performing flat mode on {n_classes}-class labelling ({mode}).')
        dfout = labelling_core(df, columns=columns, verbose=verbose, mode=mode,  
                                  labelnames=labelnames, rowname=rowname, colname=colname, 
                                  n_classes=n_classes,algorithm=algorithm)
    dfout.index.name = rowname
    dfout = dfout.replace(labelnames)
    if verbose: print(dfout.value_counts())
    return dfout
