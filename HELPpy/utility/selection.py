import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Union, Callable
import matplotlib.pyplot as plt
from ..models.labelling import labelling
from ..visualization.plot import svenn_intesect
import random
from ..models.labelling import labelling
from ..visualization.plot import svenn_intesect
import random

def delrows_with_nan_percentage(df: pd.DataFrame, perc: float=100., verbose=False):    
    """
    Filter rows in a DataFrame based on the percentage of NaN values.

    Parameters:
    :param: pd.DataFrame df: The input DataFrame.
    :param: float perc: The percentage of NaN values allowed in each row. Default is 0.0.

    :return: A new DataFrame with rows filtered based on the specified percentage of NaN values.
    :rtype: pd.DataFrame
    """
    min_count =  int(((100-perc)/100)*df.shape[1] + 1)  
    df_filtered = df.dropna(axis=0, thresh=min_count)
    if verbose:
        print(f"Removed {len(df)-len(df_filtered)} rows from {len(df)} with at least {perc}% NaN")
    return df_filtered

def filter_crispr_by_model(df: pd.DataFrame, df_map: pd.DataFrame, minlines: int=1, 
                           line_colname: str='ModelID', line_group: str='OncotreeLineage'):
    """
    Filter a CRISPR DataFrame based on a mapping DataFrame and specified conditions.

    :param: pd.DataFrame df: The CRISPR DataFrame to be filtered.
    :param: pd.DataFrame df_map: The mapping DataFrame containing information about cell lines and models.
    :param: int minlines int: The minimum number of lines required for a tissue in the model. Default is 1.
    :param: str line_colname: The column name in both DataFrames representing the cell line ID. Default is 'ModelID'.
    :param: str line_group: The column name in the mapping DataFrame representing the tissue/lineage group. Default is 'OncotreeLineage'.

    :return: A new DataFrame with CRISPR data filtered based on the selected cell lines and conditions.
    :rtype: pd.DataFrame
    """
    # Get cell lines from the mapping DataFrame
    map_cell_lines = df_map[~df_map[line_group].isna()][line_colname].values

    # Intersect cell lines in the CRISPR DataFrame with cell lines in the mapping DataFrame
    dep_cell_lines = np.intersect1d(df.columns, map_cell_lines)

    # Filter mapping DataFrame based on common cell lines
    df_map_filtered = df_map[df_map[line_colname].isin(dep_cell_lines)]

    # Select tissue models with lines greater than or equal to minlines
    sel_cell_lines = filter_cellmap(df_map_filtered, minlines, line_group=line_group)

    # Return filtered CRISPR DataFrame based on selected cell lines
    return df[np.intersect1d(df.columns, sel_cell_lines[line_colname].values)]


def filter_cellmap(df_map: pd.DataFrame, minlines: int=1, line_group: str='OncotreeLineage'):

    """
    Filters a cell map DataFrame based on the minimum number of lines per group.
    
    Parameters:
    :param pd.DataFrame df_map: The input DataFrame containing cell map information.
    :param int minlines: The minimum number of lines required to retain a group.
    :param str line_group: Column name for the grouping information in the cell map DataFrame. Default: 'OncotreeLineage'.
    
    :return: Filtered DataFrame containing only the groups that meet the minimum lines criteria.
    :rtype: pd.DataFrame
    
    :example:
    
    .. code-block:: python

        filtered_df = filter_cellmap(cell_map_data, minlines=10, line_group='OncotreeLineage')
    """
    tl = df_map[line_group].dropna().value_counts()
    tissue_list = [x[0] for x in list(filter(lambda x: x[1] >= minlines, zip(tl.index.values.astype(str) , tl.values)))]
    return df_map[df_map[line_group].isin(tissue_list)]
    
# select cell lines from depmap CRISPR file
def select_cell_lines(df: pd.DataFrame, df_map: pd.DataFrame, tissue_list: Union[str,List[str]], line_group='OncotreeLineage', line_col='ModelID', nested=False, verbose=0):
    """
    Select cell lines based on tissue and mapping information.

    :param pd.DataFrame df: DataFrame containing cell line information.
    :param pd.DataFrame df_map: DataFrame containing mapping information.
    :param List[str] tissue_list: List of tissues for which cell lines need to be selected.
    :param str line_group: The column in 'df_map' to use for line selection (default is 'ModelID').
    :param str line_col: The column in 'df_map' to use for tissue selection (default is 'OncotreeLineage').
    :param bool nested: Whether to return cell lines as nested lists (lists for each tissue to enable mode of mode in labelling).
    :param int verbose: Verbosity level for printing information.

    :return: List of selected cell lines, either flattened or nested based on the 'nested' parameter.
    :rtype: List

    :example:

    .. code-block:: python

        df = pd.DataFrame(...)
        df_map = pd.DataFrame(...)
        tissue_list = ['Tissue1', 'Tissue2']
        selected_lines = select_cell_lines(df, df_map, tissue_list, line_group='OncotreeLineage', line_col='ModelID', nested=False, verbose=1)
    """
    lines = []

    # Threat the case that all tissues are selected for mode on tissue-specific labels
    if tissue_list=='all':
        # Get all cell lines from the mapping DataFrame 
        map_cell_lines = df_map[line_col].values
        # Intersect with cell lines in the main DataFrame
        dep_cell_lines = np.intersect1d(df.columns, map_cell_lines)
        # get tissue list cell lines belong to from mapping
        tissue_list=np.unique(df_map[df_map[line_col].isin(dep_cell_lines)][line_group].values)

    # Iterate over each tissue in the provided list
    for tissue in tissue_list:
        # Get the cell lines from the mapping DataFrame for the given tissue
        map_cell_lines = df_map[df_map[line_group] == tissue][line_col].values

        # Intersect with cell lines in the main DataFrame
        dep_cell_lines = np.intersect1d(df.columns, map_cell_lines)

        # check list of lines is not empyt (the tissue may not be in the Model)
        if dep_cell_lines == []:
            raise Exception(f"empty lits of line ... the tissue {tissue} may not be in the model.")
        
        # Append cell lines to the result list (either nested or flattened)
        if nested:
            lines += [list(dep_cell_lines)]
        else:
            lines += list(dep_cell_lines)

        # Print verbose information if requested
        if verbose:
            print(f'There are {len(dep_cell_lines)} "{tissue}" cell-lines in the CRISPR file '
                f'in common with the {len(map_cell_lines)} cell-lines in DepMap')

    # Print total selected cell lines if verbose
    if verbose:
        print(f'A total of {len(lines)} have been selected for {tissue_list}.')

    # Return the list of selected cell lines
    return lines

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

def EG_tissues_intersect(tissues: Dict[str, pd.DataFrame], common_df: None or pd.DataFrame() = None,
                         labelname: str='label', labelval: str='E', display: bool = False, verbose: bool = False, 
                         barheight: int = 2, barwidth: int = 10, fontsize: int = 17) -> Tuple[set,set,set]:
    """
    Calculate the intersection and differences of gene sets across multiple tissues.

    :param Dict[str, pd.DataFrame] tissues: Dictionary of tissue names and associated dataframes.
    :param Union[None, pd.DataFrame] common_df: DataFrame containing common data.
    :param str labelname: Name of the label column in the dataframes.
    :param str labelval: Value to consider as the target label.
    :param bool display: Whether to display a Venn diagram.
    :param bool verbose: Whether to print verbose information.
    :param int barheight: Height of bars in the Venn diagram.
    :param int barwidth: Width of the Venn diagram.
    :param int fontsize: Font size of the Venn diagram labels.

    :return: A tuple containing sets of genes for each tissue,
             the intersection of genes, and differences in genes.
    :rtype: Tuple[Dict[str, set], set, Dict[str, set]] 

    :example:

    .. code-block:: python

        tissues = {'Tissue1': pd.DataFrame(...), 'Tissue2': pd.DataFrame(...), ...}
        common_df = pd.DataFrame(...)  # Optional
        sets, inset, diffs = EG_tissues_intersect(tissues, common_df, labelname='label', labelval='E', display=True)
    """
    sets = {}

    # If subtract_common is True, calculate the set of pan-tissue labels
    if common_df is None:
        common_set = set()
    else:
        common_set = set(common_df[common_df[labelname] == labelval].index.values)
        if verbose:
            print(f"Subtracting {len(common_set)} common EGs...")

    # Iterate over each tissue in the provided list
    for tissue, df in tissues.items():
        newset = set(df[df[labelname] == labelval].index.values)

        # subtract common eg labels
        newset = newset - common_set

        # Add the set of EGs for the tissue to the list
        sets[tissue] = newset

    # If display is True, display a Venn diagram
    if display:
        svenn_intesect(list(sets.values()), list(sets.keys()), figsize=(barwidth, barheight * len(tissues)), fontsize=fontsize)

    # Calculate the intersection of sets
    inset = set.intersection(*list(sets.values()))

    # Print verbose information about the overlapping genes
    if verbose:
        print(f'Overlapping of {len(inset)} genes between {list(sets.keys())}')

    # Calculate differences in EGs for each tissue
    setsl = list(sets.values())
    tl = list(sets.keys())
    diffs = {}
    for i, tl in enumerate(tl):
        setrest = setsl[:i] + setsl[i + 1:]
        if len(setrest) > 0:
            diffs[tl] = setsl[i] - set.union(*setrest)
        else:
            diffs[tl] = setsl[i]
        if verbose:
            print(f'{len(diffs[tl])} genes only in {tl}')

    # Return the sets of EGs, intersection, and differences
    return sets, inset, diffs
                             
# Compute intersection of essential genes by tissues
def EG_tissues_intersect_dolabelling(df: pd.DataFrame, df_map: pd.DataFrame, tissues: List[str] = [], subtract_common: bool = False, three_class: bool = False,
                              display: bool = False, verbose: bool = False, barheight: int = 2, barwidth: int = 10, fontsize: int = 17) -> pd.DataFrame:
    """
    Identify overlapping and unique Essential Genes (EGs) by tissues.

    :param pd.DataFrame df: DataFrame containing cell line information.
    :param pd.DataFrame df_map: DataFrame containing mapping information.
    :param List[str] tissues: List of tissues for which EGs need to be identified.
    :param bool subtract_common: Whether to subtract common EGs from pantissue labeling.
    :param bool three_class: Whether to use a three-class labeling (E, NE, NC).
    :param bool display: Whether to display a Venn diagram.
    :param bool verbose: Verbosity level for printing information.
    :param int barheight: Height of the Venn diagram.
    :param int barwidth: Width of the Venn diagram.
    :param int fontsize: Font size for the Venn diagram.

    :return: Tuple containing sets of EGs, intersection of EGs, and differences in EGs.
    :rtype: Tuple[List[set], set, Dict[str, set]] 
    :example:

    .. code-block:: python

        df = pd.DataFrame(...)
        df_map = pd.DataFrame(...)
        tissues = ['Tissue1', 'Tissue2']
        sets, inset, diffs = EG_tissues_intersect_dolabelling(df, df_map, tissues, subtract_common=True, three_class=False, display=True, verbose=True)
    """
 
    sets = []

    # If subtract_common is True, calculate the set of pan-tissue labels
    if subtract_common:
        if verbose:
            print("Subtracting common EG of pan-tissue labeling")
        pan_labels_df = labelling(df, verbose=verbose)
        panset = set(pan_labels_df[pan_labels_df['label'] == 'E'].index.values)

    # Iterate over each tissue in the provided list
    for tissue in tissues:
        # Select cell lines for the given tissue
        cell_lines = select_cell_lines(df, df_map, [tissue])

        # If there are cell lines, calculate the set of EGs for the tissue
        if len(cell_lines) > 0:
            labels_df = labelling(df, cell_lines, verbose=verbose)
            newset = set(labels_df[labels_df['label'] == 'E'].index.values)

            # If subtract_common is True, subtract the pan-tissue labels
            if subtract_common:
                newset = newset - panset

            # Add the set of EGs for the tissue to the list
            sets += [newset]

    # If display is True, display a Venn diagram
    if display:
        svenn_intesect(sets, tissues, figsize=(barwidth, barheight * len(tissues)), fontsize=fontsize)

    # Calculate the intersection of sets
    inset = set.intersection(*sets)

    # Print verbose information about the overlapping genes
    if verbose:
        print(f'Overlapping of {len(inset)} genes')

    # Calculate differences in EGs for each tissue
    diffs = {}
    for i, tissue in enumerate(tissues):
        diffs[tissue] = sets[i] - set.union(*(sets[:i] + sets[i + 1:]))
        if verbose:
            print(f'{len(diffs[tissue])} genes only in {tissue}')

    # Return the sets of EGs, intersection, and differences
    return sets, inset, diffs
