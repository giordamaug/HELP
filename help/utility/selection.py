import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Callable
import matplotlib.pyplot as plt
from ..models.labelling import Help
from ..visualization.plot import svenn_intesect

# select cell lines from depmap CRISPR file
def select_cell_lines(df: pd.DataFrame, df_map: pd.DataFrame, tissue_list: List[str], line_group: str='lineage1', line_col: str='ModelID', nested=False, verbose=0):
    """
    Select cell lines based on tissue and mapping information.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cell line information.
    df_map : pd.DataFrame
        DataFrame containing mapping information.
    tissue_list : List[str]
        List of tissues for which cell lines need to be selected.
    nested : bool, optional
        Whether to return cell lines as nested lists (lists for each tissue).
    verbose : int, optional
        Verbosity level for printing information.

    Returns
    -------
    List
        List of selected cell lines, either flattened or nested based on the 'nested' parameter.
    """

    lines = []

    # Iterate over each tissue in the provided list
    for tissue in tissue_list:
        # Get the cell lines from the mapping DataFrame for the given tissue
        map_cell_lines = df_map[df_map[line_group] == tissue][line_col].values

        # Intersect with cell lines in the main DataFrame
        dep_cell_lines = np.intersect1d(df.columns, map_cell_lines)

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

# Compute intersection of essential genes by tissues
def EG_by_tissues_intersect(df: pd.DataFrame, df_map: pd.DataFrame, tissues: List[str] = [], subtract_common: bool = False, three_class: bool = False,
                              display: bool = False, verbose: bool = False, barheight: int = 2, barwidth: int = 10, fontsize: int = 17) -> pd.DataFrame:
    """
    Identify overlapping and unique Essential Genes (EGs) by tissues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cell line information.
    df_map : pd.DataFrame
        DataFrame containing mapping information.
    tissues : List[str]
        List of tissues for which EGs need to be identified.
    subtract_common : bool, optional
        Whether to subtract common EGs from pantissue labeling.
    three_class : bool, optional
        Whether to use a three-class labeling (E, NE, NC).
    display : bool, optional
        Whether to display a Venn diagram.
    verbose : bool, optional
        Verbosity level for printing information.
    barheight : int, optional
        Height of the Venn diagram.
    barwidth : int, optional
        Width of the Venn diagram.
    fontsize : int, optional
        Font size for the Venn diagram.

    Returns
    -------
    Tuple[pd.DataFrame]
        Tuple containing sets of EGs, intersection of EGs, and differences in EGs.
    """
 
    sets = []

    # If subtract_common is True, calculate the set of pan-tissue labels
    if subtract_common:
        if verbose:
            print("Subtracting common EG of pan-tissue labeling")
        pan_labels_df = Help(verbose=verbose).labelling(df)
        panset = set(pan_labels_df[pan_labels_df['label'] == 'E'].index.values)

    # Iterate over each tissue in the provided list
    for tissue in tissues:
        # Select cell lines for the given tissue
        cell_lines = select_cell_lines(df, df_map, [tissue])

        # If there are cell lines, calculate the set of EGs for the tissue
        if len(cell_lines) > 0:
            labels_df = Help(verbose=verbose).labelling(df, cell_lines)
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
