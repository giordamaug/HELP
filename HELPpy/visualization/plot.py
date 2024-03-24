from supervenn import supervenn
from typing import List
import matplotlib.pyplot as plt

def svenn_intesect(sets: List[set], labels: List[str], figsize=(10,20), fontsize=10, ylabel='EG', xlabel ='no. Genes', saveflag: bool=False) ->  None:
    """
    Generate a Supervenn diagram to visualize the intersection of multiple sets.

    :param List[set] sets: List of sets to be visualized.
    :param List[str] labels: List of labels corresponding to each set.
    :param tuple figsize: Figure size in inches, as a tuple (width, height) (default is (10, 20)).
    :param int fontsize: Font size for labels (default is 10).
    :param str xlabel: label for x axis (default is 'no. Genes')
    :param str ylabel: label for y axis (default is 'EG')
    :param bool saveflag: Whether to save the generated diagram as an image (default is False).

    :return: None

    The function generates a Supervenn diagram using the 'supervenn' library and Matplotlib.
    The diagram visualizes the intersection of multiple sets with labeled areas.

    :example:

    .. code-block:: python

        # Usage example:
        set1 = {1, 2, 3, 4, 5}
        set2 = {3, 4, 5, 6, 7}
        set3 = {5, 6, 7, 8, 9}
        svenn_intesect([set1, set2, set3], ["Set A", "Set B", "Set C"], saveflag=True)
    """
    plt.figure(figsize=figsize)
    supervenn(sets, labels, widths_minmax_ratio=0.05, side_plots='right')
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    if saveflag: plt.savefig(f"{'_'.join(labels)}_svenn.jpg", dpi=600)

