import pandas as pd
from typing import Callable, List, Dict, Tuple
import statistics
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_multiotsu, threshold_yen
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
        elif algorithm == 'yen':
            if num_thresholds > 2:
                warnings.warn('Yen thresholding only upport one threshow (binary separation)... using linspace mode!')
                thresh = np.linspace(np.nanmin(valid_values), np.nanmax(valid_values), num_thresholds + 1)[1:-1]
            else:
                thresh = np.array([threshold_yen(valid_values)])
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

def rows_with_all_nan(df):    
    idx = df.index[df.isnull().all(1)]
    nans = df.loc[idx]
    return idx.values
    
def modemax_nan(a: np.ndarray, reducefoo: Callable[[List[int]], int] = max) -> np.ndarray:
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
            res += [np.nan]
        else:
            res += [reducefoo(modes)] # if max(modes) is not np.nan else sorted(set(modes))[-2]]
    return np.array(res)

def labelling_core(df: pd.DataFrame, columns: List[str] = [], n_classes: int=2,
                  verbose: bool = False, labelnames: Dict[int, str] = {0: 'E', 1: 'NE'},
                  mode='flat-multi', algorithm='otsu', rowname: str = 'gene', colname: str = 'label', reducefoo=reducefoo) -> Tuple[pd.DataFrame, np.ndarray]:
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
    :param Callable[[List[int]], int] reducefoo: function used for solving ex-aequo in mode.

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
        Labels2 = modemax_nan(Q, reducefoo=reducefoo)
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
        Labels = modemax_nan(QNE, reducefoo=reducefoo)
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
        Labels = modemax_nan(Q, reducefoo=reducefoo)
        dfout =  pd.DataFrame(index=df.index)
        dfout.index.name = rowname
        dfout[colname] = Labels
    else:
        raise Exception("Labelling mode not supported!")
    return dfout
    
def labelling(df: pd.DataFrame, columns: List[List[str]] = [], n_classes: int=2, 
             verbose: bool = False, labelnames: Dict[int, str] = {1 : 'NE', 0: 'E'},
             mode='flat-multi', rowname: str = 'gene', colname: str = 'label', algorithm='otsu', reducefoo=max) -> pd.DataFrame:
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
    :param str algorithm: quantization algorithm type otsu|linspace (default: 'otsu'). 
    :param Callable[[List[int]], int] reducefoo: function used for solving ex-aequo in mode.

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
        # Mode per groups
        if verbose: print(f'performing mode of mode on {n_classes}-class labelling ({mode}).')
        L_tot = np.empty(shape=(len(df), 0))
        for lines in columns:
            # check if there are rows with all Nans
            nanrows = rows_with_all_nan(df[lines])
            if len(nanrows) > 0:
                warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
            labels = labelling_core(df, columns=lines, verbose=verbose, mode=mode,  
                                  labelnames=labelnames, rowname=rowname, colname=colname, 
                                  n_classes=n_classes, algorithm=algorithm, reducefoo=reducefoo)
            labels = labels.replace(dict(map(reversed, labelnames.items()))).infer_objects()
            L_tot = np.hstack((L_tot, labels.values))
        # Execute mode on each tissue and sort'em
        modeOfmode = modemax_nan(L_tot, reducefoo=reducefoo)
        dfout =  pd.DataFrame(index=df[sum(columns, [])].index)
        dfout.index.name = rowname
        dfout[colname] = modeOfmode
        dfout = dfout.replace(labelnames)
    elif any(isinstance(sub, list) for sub in columns) and len(columns) > 0:
        raise Exception("Wrong columns partition format.")
    else:
        if verbose: print(f'performing flat mode on {n_classes}-class labelling ({mode}).')
        nanrows = rows_with_all_nan(df[columns])
        if len(nanrows) > 0:
            warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will ha NaN labels in your output.")
        dfout = labelling_core(df, columns=columns, verbose=verbose, mode=mode,  
                                  labelnames=labelnames, rowname=rowname, colname=colname, 
                                  n_classes=n_classes,algorithm=algorithm, reducefoo=reducefoo)
    dfout.index.name = rowname
    dfout = dfout.replace(labelnames)
    if verbose: print(dfout.value_counts())
    return dfout


#
# The old Otsu labelling version:
# from Otsu_module.py relase, translated into python from Lucia's matlab code.
# it produces 
#
class Help:
    def __init__(self, verbose: bool=False):
        self.verbose = verbose

    @staticmethod
    def quantize_2d_array(T: np.ndarray, n: int, verbose: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes a 2D NumPy array column-wise.

        :param np.ndarray T: The input 2D array.
        :param int n: Number of quantization levels.
        :param int verbose: Verbosity level for printing information (default: 0).

        :returns: Tuple containing the quantized array and threshold vector.
        :rtype: Tuple[np.ndarray, np.ndarray]

        :example: 

        .. code-block:: python

            import numpy as np
            from help.models.labelling import Help

            # Example usage
            input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            quantization_levels = 3
            verbosity = 1

            quantized_array, threshold_vector = quantize_2d_array(input_array, quantization_levels, verbosity)
        """
        Q = np.zeros((T.shape[0], T.shape[1]), dtype=np.uint8)
        ValueForNaNs = np.nanmax(T) + 1
        threshVec = np.zeros(T.shape[1])
        
        for c in tqdm(range(0, T.shape[1])):
            ColValues = T[:, c]
            ColTh, _ = Help.otsu_thresholding(ColValues, n - 1)
            If1 = ColValues.copy()
            ismiss = np.isnan(ColValues)
            
            if np.any(ismiss):
                If1[ismiss] = ValueForNaNs
                if verbose: print(f'Cell line {c} has {np.sum(ismiss)} NaNs')
            
            threshVec[c] = ColTh
            ConNans = np.digitize(If1, [np.nanmin(T), ColTh, np.nanmax(T)])
            ConNans[ismiss] = n + 1
            Q[:, c-1] = ConNans
            
        return Q, threshVec
    
    @staticmethod
    def otsu_thresholding(A: np.ndarray, N: int = 1) -> Tuple[float, float]:
        """
        Computes Otsu's threshold for image segmentation.

        :param np.ndarray A: Input array.
        :param int N: Number of classes (1 or 2).

        :returns: Tuple containing the threshold value and related metric.
        :rtype: Tuple[float, float]

        :example
        
        .. code-block:: python

            import numpy as np
            from help.models.labelling import Help

            # Example usage
            input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            num_classes = 2

            threshold_value, metric = Help.otsu_thresholding(input_array, num_classes)
        """
        if N < 1 or N > 2:
            raise ValueError("N must be either 1 or 2.")
        
        num_bins = 256
        p, minA, maxA = Help.compute_pdf(A, num_bins)
        assert len(p) > 0, "Cannot compute PDF."
        
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(num_bins))
        mu_t = mu[-1]

        sigma_b_squared = Help.compute_sigma_b_squared(N, num_bins, omega, mu, mu_t)
        
        maxval = np.nanmax(sigma_b_squared)
        assert np.isfinite(maxval), "Cannot find a finite maximum for sigma_b_squared."
        
        if N == 1:
            idx = np.nanargmax(sigma_b_squared)
            #idx = np.where(sigma_b_squared == maxval)[0]
            thresh = np.mean(idx) - 1
        else:
            maxR, maxC = np.unravel_index(np.argmax(sigma_b_squared), sigma_b_squared.shape)
            thresh = np.mean([maxR, maxC]) - 1
        
        thresh = minA + thresh / 255 * (maxA - minA)
        metric = maxval / np.sum(p * ((np.arange(num_bins) - mu_t) ** 2))
        return thresh, metric
    
    @staticmethod
    def compute_pdf(A: np.ndarray, num_bins: int) -> Tuple[np.ndarray, float, float]:
        """
        Computes the probability density function (PDF) of an array.

        :param np.ndarray A: Input array.
        :param int num_bins: Number of bins for histogram.

        :returns: Tuple containing the PDF, minimum value, and maximum value.
        :rtype: Tuple[np.ndarray, float, float]

        :example

        .. code-block:: python

            from help.models.labelling import Help
            import numpy as np

            # Example usage
            input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            num_bins = 5

            pdf, min_val, max_val = Help.compute_pdf(input_array, num_bins)
        """
        A = A.ravel()  # Vectorize A for faster histogram computation

        # replace NaNs with empty values (both in float and integer matrix)
        A = A[np.isfinite(A)]
        # if A was full of only NaNs return a null distribution
        if A.size == 0:
            return np.array([]), np.nan, np.nan

        if np.issubdtype(A.dtype, np.floating):   # if a float A the scale to the range [0 1]
            # apply sclaing only to finite elements
            idx_isfinite = np.isfinite(A)
            if np.any(idx_isfinite):
                minA = np.min(A[idx_isfinite])
                maxA = np.max(A[idx_isfinite])
            else:  # ... A has only Infs and NaNs, return a null distribution
                minA = np.min(A)
                maxA = np.max(A)
            A = (A - minA) / (maxA - minA)
        else:  # if an integer A no need to scale
            minA = np.min(A)
            maxA = np.max(A)
        if minA == maxA:   # if therei no range, retunr null distribution
            return np.array([]), minA, maxA
        else:
            counts, _ = np.histogram(A, bins=num_bins, range=(0, 1))
            distrib = counts / np.sum(counts)
            return distrib, minA, maxA
        
    @staticmethod
    def compute_sigma_b_squared(N: int, num_bins: int, omega: np.ndarray, mu: np.ndarray, mu_t: float) -> np.ndarray:
        """
        Computes sigma_b_squared for Otsu's thresholding.

        :param int N: Number of classes (1 or 2).
        :param int num_bins: Number of bins for histogram.
        :param np.ndarray omega: Cumulative distribution function.
        :param np.ndarray mu: Cumulative mean values.
        :param float mu_t: Total mean.

        :returns: Sigma_b_squared values.
        :rtype: np.ndarray

        :example

        .. code-block:: python

            # Example usage
            from help.models.labelling import Help
            N_value = 2
            num_bins = 256
            omega_array = np.array([0.2, 0.8])
            mu_array = np.array([100.0, 150.0])
            total_mean = 125.0

            sigma_b_squared_values = Help.compute_sigma_b_squared(N_value, num_bins, omega_array, mu_array, total_mean)
        """
        if N == 1:
            sigma_b_squared = np.ones( (len(omega)) ) * np.nan
            np.divide((mu_t * omega - mu)**2,(omega * (1 - omega)), out=sigma_b_squared, where=(omega * (1 - omega))!=0)
            #sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega))
        elif N == 2:
            # Rows represent thresh(1) (lower threshold) and columns represent
            # thresh(2) (higher threshold).
            omega0 = np.tile(omega, (num_bins, 1))
            mu_0_t = np.tile((mu_t - mu / omega), (num_bins, 1))
            omega1 = omega.reshape(num_bins, 1) - omega
            mu_1_t = mu_t - (mu - mu.T) / omega1
            
            # Set entries corresponding to non-viable solutions to NaN
            allPixR, allPixC = np.meshgrid(np.arange(num_bins), np.arange(num_bins))
            pixNaN = allPixR >= allPixC  # Enforce thresh(1) < thresh(2)
            omega0[pixNaN] = np.nan
            omega1[pixNaN] = np.nan
            
            term1 = omega0 * (mu_0_t ** 2)
            term2 = omega1 * (mu_1_t ** 2)
            omega2 = 1 - (omega0 + omega1)
            omega2[omega2 <= 0] = np.nan  # Avoid divide-by-zero Infs in term3
            term3 = ((omega0 * mu_0_t + omega1 * mu_1_t) ** 2) / omega2
            sigma_b_squared = term1 + term2 + term3
        
        return sigma_b_squared

    @staticmethod
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
            from help.models.labelling import Help
            input_array = np.array([[1, 2, 3, 3],
                                    [2, 3, 4, 4],
                                    [4, 5, 6, 6]])

            mode_values = Help.modemax(input_array)
        """
        return np.array([reducefoo(statistics.multimode(a[x,:])) for x in range(a.shape[0])])

    @staticmethod
    def help_core(df: pd.DataFrame, columns: List[str], three_class: bool = False,
                  verbose: bool = False, labelnames: Dict[int, str] = {0: 'E', 1: 'NE'},
                  rowname: str = 'gene', colname: str = 'label') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Core function for HELP algorithm.

        :param pd.DataFrame df: Input DataFrame.
        :param List[str] columns: List of column names.
        :param bool three_class: Flag for three-class labeling (default: False).
        :param bool verbose: Verbosity level for printing information (default: False).
        :param Dict[int, str] labelnames: Dictionary mapping class labels to names (default: {0: 'E', 1: 'NE'}).
        :param str rowname: Name of the DataFrame index (default: 'gene').
        :param str colname: Name of the label column (default: 'label').

        :returns: Output DataFrame and quantized array.
        :rtype: Tuple[pd.DataFrame, np.ndarray]

        :example

        .. code-block:: python

            # Example usage
            from help.models.labelling import Help
            input_df = pd.DataFrame(...)
            columns_list = ['feature1', 'feature2', 'feature3']
            output_df, quantized_array = Help.help_core(input_df, columns_list, three_class=True, verbose=True, labelnames={0: 'E', 1: 'NE'}, rowname='gene', colname='label')
        """
        # check labelnames with two or three class modes
        if three_class:
            assert list(labelnames.keys()) == [0,1,2], f"Label names {labelnames} incompatible with three class mode."
        else:
            assert list(labelnames.keys()) == [0,1], f"Label names {labelnames} incompatible with two class mode"
        # Extract data from the dataframe
        if columns == []:
            T = df.to_numpy()
        else:
            T = df[columns].to_numpy()
        NumberOfClasses = 2
        
        if verbose: print("Two-class labelling:") 
        # Perform quantization
        Q2, Thr = Help.quantize_2d_array(T, NumberOfClasses, verbose = verbose)
        Q2 = Q2 - 1
        #modeQ2 = stats.mode(Q2, axis=1, keepdims=False).mode
        modeQ2 = Help.modemax(Q2)
        if verbose: print(modeQ2.shape)
        
        dfout =  pd.DataFrame(index=df.index)
        dfout.index.name = rowname
        dfout[colname] = modeQ2   # .ravel()
        if verbose: print(dfout.value_counts())
        
        if three_class: 
            if verbose: print("Three-class labelling:")
            NE_genes = dfout[dfout[colname]==1].index
            if columns == []:
                TNE = df.loc[NE_genes].to_numpy()
            else:
                TNE = df[columns].loc[NE_genes].to_numpy()
            NumberOfClasses = 2
            QNE2, ThrNE = Help.quantize_2d_array(TNE, NumberOfClasses, verbose = verbose)
            QNE2 = QNE2 - 1
            #modeQ2NE = stats.mode(QNE2, axis=1, keepdims=False).mode
            modeQ2NE = Help.modemax(QNE2)
            if verbose: print(modeQ2NE.shape)
            dfout2 =  pd.DataFrame(index=NE_genes)
            dfout2.index.name = rowname
            dfout2[colname] = modeQ2NE #.ravel()
            dfout2 = dfout2.replace({0: 1, 1: 2})
            dfout.loc[dfout.index.isin(dfout2.index), [colname]] = dfout2[[colname]]
            if verbose: print(dfout.value_counts())
        dfout = dfout.replace(labelnames)
        return dfout, Q2

    @staticmethod
    def labelling(df: pd.DataFrame, columns: List[List[str]] = [], three_class: bool = False,
             verbose: bool = False, labelnames: Dict[int, str] = {},
             rowname: str = 'gene', colname: str = 'label') -> pd.DataFrame:
        """
        Main function for HELP algorithm.

        :param pd.DataFrame df: Input DataFrame.
        :param List[List[str]] columns: List of column names for partitioning (default: []).
        :param bool three_class: Flag for three-class labeling (default: False).
        :param bool verbose: Verbosity level for printing information (default: False).
        :param Dict[int, str] labelnames: Dictionary mapping class labels to names (default: {}).
        :param str rowname: Name of the DataFrame index (default: 'gene').
        :param str colname: Name of the label column (default: 'label').

        :returns: Output DataFrame with labels.
        :rtype: pd.DataFrame

        :example
       
        .. code-block:: python

            # Example usage
            from help.models.labelling import Help
            input_df = pd.DataFrame(...)
            output_df = Help.labelling(input_df, columns=[], three_class=True, verbose=True, labelnames={0: 'E', 1: 'NE'}, rowname='gene', colname='label')
        """
        if labelnames == {}:
            labelnames = {0: 'E', 1:'aE', 2:'sNE'} if three_class else {0: 'E', 1:'NE'}
        if all(isinstance(sub, list) for sub in columns) and len(columns) > 0:              # use mode of mode across tissues
            if verbose: print('performing mode of mode.')
            Q2_tot = []
            for lines in columns:
                _, Q2 = Help.help_core(df, lines, three_class=three_class, verbose=verbose, labelnames=labelnames, rowname=rowname, colname=colname)
                Q2_tot += [Q2]
            # Execute mode on each tissue and sort'em
            if verbose: print(Q2_tot)
            Q2Mode_tot = []
            for k in Q2_tot:
                #Q2Mode_tot.append(stats.mode(k, axis=1, keepdims=False).mode)
                Q2Mode_tot.append(Help.modemax(k))

            Q2Mode_tot = np.vstack(Q2Mode_tot).T
        
            # Get mode of mode among tissues
            #modeOfmode = stats.mode(Q2Mode_tot, axis=1, keepdims=False).mode
            modeOfmode = Help.modemax(Q2Mode_tot)
            dfout =  pd.DataFrame(index=df[sum(columns, [])].index)
            dfout.index.name = rowname
            dfout[colname] = modeOfmode #.ravel()
            dfout = dfout.replace(labelnames)
            if verbose: print(dfout.value_counts())
            return dfout
        elif any(isinstance(sub, list) for sub in columns) and len(columns) > 0:
            raise Exception("Wrong columns partition format.")
        else:
            dfout, Q2 = Help.help_core(df, columns, three_class=three_class, verbose=verbose,  labelnames=labelnames, rowname=rowname, colname=colname)
            return dfout
