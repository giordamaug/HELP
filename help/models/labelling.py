import numpy as np
import numpy.typing as npt
import pandas as pd
import sys
from scipy import stats
import warnings
from tqdm import tqdm
from typing import List, Tuple, Callable, Dict
import statistics

class Help:
    def __init__(self, verbose: bool=False):
        self.verbose = verbose

    @staticmethod
    def QuantizeByColumns(T: np.ndarray, n: int, verbose: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes a 2D NumPy array column-wise.
    
        Parameters:
        -----------
        T : np.ndarray
            The input 2D array.
        n : int
            Number of quantization levels.
        verbose : int, optional
            Verbosity level for printing information (default: 0).
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Quantized array and threshold vector.
        """
        Q = np.zeros((T.shape[0], T.shape[1]), dtype=np.uint8)
        ValueForNaNs = np.nanmax(T) + 1
        threshVec = np.zeros(T.shape[1])
        
        for c in tqdm(range(0, T.shape[1])):
            ColValues = T[:, c]
            ColTh, _ = otsuthresholding(ColValues, n - 1)
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
    def otsuthresholding(A: np.ndarray, N: int = 1) -> Tuple[float, float]:
        """
        Computes Otsu's threshold for image segmentation.
    
        Parameters:
        -----------
        A : np.ndarray
            Input array.
        N : int
            Number of classes (1 or 2).
    
        Returns
        -------
        Tuple[float, float]
            Threshold value and related metric.
        """
        if N < 1 or N > 2:
            raise ValueError("N must be either 1 or 2.")
        
        num_bins = 256
        p, minA, maxA = getpdf(A, num_bins)
        assert len(p) > 0, "Cannot compute PDF."
        
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(num_bins))
        mu_t = mu[-1]

        sigma_b_squared = compute_sigma_b_squared(N, num_bins, omega, mu, mu_t)
        
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
    def getpdf(A: np.ndarray, num_bins: int) -> Tuple[np.ndarray, float, float]:
        """
        Computes the probability density function (PDF) of an array.
    
        Parameters:
        -----------
        A : np.ndarray
            Input array.
        num_bins : int
            Number of bins for histogram.
    
        Returns
        -------
        Tuple[np.ndarray, float, float]
            PDF, minimum value, and maximum value.
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
    
        Parameters:
        -----------
        N : int
            Number of classes (1 or 2).
        num_bins : int
            Number of bins for histogram.
        omega : np.ndarray
            Cumulative distribution function.
        mu : np.ndarray
            Cumulative mean values.
        mu_t : float
            Total mean.
    
        Returns
        -------
        np.ndarray
        Sigma_b_squared values.
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
    
        Parameters:
        -----------
        a : np.ndarray
            Input 2D array.
        reducefoo : Callable[[List[int]], int], optional
            Reduction function (default: max).
    
        Returns
        -------
        np.ndarray
            Mode values.
        """
        return np.array([reducefoo(statistics.multimode(a[x,:])) for x in range(a.shape[0])])

    @staticmethod
    def help_core(df: pd.DataFrame, columns: List[str], three_class: bool = False,
                  verbose: bool = False, labelnames: Dict[int, str] = {0: 'E', 1: 'NE'},
                  rowname: str = 'gene', colname: str = 'label') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Core function for HELP algorithm.
    
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[str]
            List of column names.
        three_class : bool, optional
            Flag for three-class labeling (default: False).
        verbose : bool, optional
            Verbosity level for printing information (default: False).
        labelnames : Dict[int, str], optional
            Dictionary mapping class labels to names (default: {0: 'E', 1: 'NE'}).
        rowname : str, optional
            Name of the DataFrame index (default: 'gene').
        colname : str, optional
            Name of the label column (default: 'label').
    
        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            Output DataFrame and quantized array.
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
        Q2, Thr = QuantizeByColumns(T, NumberOfClasses, verbose = verbose)
        Q2 = Q2 - 1
        #modeQ2 = stats.mode(Q2, axis=1, keepdims=False).mode
        modeQ2 = modemax(Q2)
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
            QNE2, ThrNE = QuantizeByColumns(TNE, NumberOfClasses, verbose = verbose)
            QNE2 = QNE2 - 1
            #modeQ2NE = stats.mode(QNE2, axis=1, keepdims=False).mode
            modeQ2NE = modemax(QNE2)
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
    
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[List[str]], optional
            List of column names for partitioning (default: []).
        three_class : bool, optional
            Flag for three-class labeling (default: False).
        verbose : bool, optional
            Verbosity level for printing information (default: False).
        labelnames : Dict[int, str], optional
            Dictionary mapping class labels to names (default: {}).
        rowname : str, optional
            Name of the DataFrame index (default: 'gene').
        colname : str, optional
            Name of the label column (default: 'label').
    
        Returns
        -------
        pd.DataFrame
            Output DataFrame with labels.
        """
        if labelnames == {}:
            labelnames = {0: 'E', 1:'aE', 2:'sNE'} if three_class else {0: 'E', 1:'NE'}
        if all(isinstance(sub, list) for sub in columns) and len(columns) > 0:              # use mode of mode across tissues
            if verbose: print('performing mode of mode.')
            Q2_tot = []
            for lines in columns:
                _, Q2 = help_core(df, lines, three_class=three_class, verbose=verbose, labelnames=labelnames, rowname=rowname, colname=colname)
                Q2_tot += [Q2]
            # Execute mode on each tissue and sort'em
            if verbose: print(Q2_tot)
            Q2Mode_tot = []
            for k in Q2_tot:
                #Q2Mode_tot.append(stats.mode(k, axis=1, keepdims=False).mode)
                Q2Mode_tot.append(modemax(k))

            Q2Mode_tot = np.vstack(Q2Mode_tot).T
        
            # Get mode of mode among tissues
            #modeOfmode = stats.mode(Q2Mode_tot, axis=1, keepdims=False).mode
            modeOfmode = modemax(Q2Mode_tot)
            dfout =  pd.DataFrame(index=df[sum(columns, [])].index)
            dfout.index.name = rowname
            dfout[colname] = modeOfmode #.ravel()
            dfout = dfout.replace(labelnames)
            if verbose: print(dfout.value_counts())
            return dfout
        elif any(isinstance(sub, list) for sub in columns) and len(columns) > 0:
            raise Exception("Wrong columns partition format.")
        else:
            dfout, Q2 = help_core(df, columns, three_class=three_class, verbose=verbose,  labelnames=labelnames, rowname=rowname, colname=colname)
            return dfout
        