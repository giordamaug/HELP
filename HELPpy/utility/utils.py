import pandas as pd
import numpy as np
import csv

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
def pandas_readcsv(filename, chunksize=50, sep=',', index_col=None, comment='#', descr:str=None, disabled=False):
    # Get number of lines in file.
    with open(filename, 'r') as fp:
        try:
            has_headings = csv.Sniffer().has_header(fp.read(1024))
            lines = len(fp.readlines())-1
        except csv.Error:
            # The file seems to be empty
            lines = len(fp.readlines())
    # Read file in chunks, updating progress bar after each chunk.
    listdf = []
    with tqdm(total=lines, desc=descr, disable=disabled) as bar:
        for i,chunk in enumerate(pd.read_csv(filename,chunksize=chunksize, index_col=index_col, comment=comment, sep=sep)):
            listdf.append(chunk)
            bar.update(chunksize)
    df = pd.concat(listdf,ignore_index=False)
    return df

def split_dataframe(df, chunk_size = 10): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def pandas_writecsv(filename, df: pd.DataFrame, chunksize=10, index=False, sep=',', descr:str=None, disabled=False):
    # Write file in chunks, updating progress bar after each chunk.
    if len(df) > chunksize + 1:
        num_chunks = len(df) // chunksize + 1
    else:
        num_chunks = 1
    with tqdm(total=num_chunks, desc=descr, disable=disabled) as bar:
        for i, chunk in enumerate(split_dataframe(df, chunksize)):
            mode = 'w' if i == 0 else 'a'
            chunk.to_csv(filename, index=index, mode=mode, sep=sep)
            bar.update()
    return df