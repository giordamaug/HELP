from supervenn import supervenn
from typing import List
import matplotlib.pyplot as plt

def svenn_intesect(sets: List[set], labels: List[str], figsize=(10,20), fontsize=10, saveflag: bool=False) ->  None:
    plt.figure(figsize=figsize)
    supervenn(sets, labels, widths_minmax_ratio=0.05, side_plots='right')
    plt.xlabel('no. Genes', fontsize = fontsize)
    plt.ylabel('CFG label', fontsize = fontsize)
    if saveflag: plt.savefig(f"{'_'.join(labels)}_svenn.jpg", dpi=600)

