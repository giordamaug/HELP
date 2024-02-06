from supervenn import supervenn
from typing import List
import matplotlib.pyplot as plt

def svenn_intesect(sets: List[set], labels: List[str], figsize=(10,20), fontsize=10, saveflag: bool=False) ->  None:
    """
    Generate a Supervenn diagram to visualize the intersection of multiple sets.

    Parameters
    ----------
    sets : List[set]
        List of sets to be visualized.
    labels : List[str]
        List of labels corresponding to each set.
    figsize : tuple, optional
        Figure size in inches, as a tuple (width, height) (default is (10, 20)).
    fontsize : int, optional
        Font size for labels (default is 10).
    saveflag : bool, optional
        Whether to save the generated diagram as an image (default is False).

    Returns
    -------
    None

    The function generates a Supervenn diagram using the 'supervenn' library and Matplotlib.
    The diagram visualizes the intersection of multiple sets with labeled areas.

    Example
    -------
    ```
    # Usage example:
    set1 = {1, 2, 3, 4, 5}
    set2 = {3, 4, 5, 6, 7}
    set3 = {5, 6, 7, 8, 9}
    svenn_intersect([set1, set2, set3], ["Set A", "Set B", "Set C"], saveflag=True)
    ```
    """
    plt.figure(figsize=figsize)
    supervenn(sets, labels, widths_minmax_ratio=0.05, side_plots='right')
    plt.xlabel('no. Genes', fontsize = fontsize)
    plt.ylabel('CFG label', fontsize = fontsize)
    if saveflag: plt.savefig(f"{'_'.join(labels)}_svenn.jpg", dpi=600)

