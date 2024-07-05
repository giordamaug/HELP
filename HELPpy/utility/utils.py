"""
This module provides a set of functions for working with CSV files using pandas and tqdm for progress bar visualization.

Functions:
    - in_notebook: Checks if the code is being executed in a Jupyter notebook.
    - pdread_csv_fromurl: Reads a CSV file from a URL.
    - pandas_readcsv: Reads a CSV file in chunks and displays a progress bar.
    - split_dataframe: Splits a DataFrame into chunks.
    - pandas_writecsv: Writes a DataFrame to a CSV file in chunks and displays a progress bar.
"""

import pandas as pd
import numpy as np
import csv
import pycurl
from io import BytesIO

def in_notebook():
    """
    Determines if the code is being executed in a Jupyter notebook.

    Returns:
        bool: True if the code is being executed in a Jupyter notebook, otherwise False.
    """
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

def pdread_csv_fromurl(url, sep=',', index_col=None):
    """
    Reads a CSV file from a URL.

    Args:
        url (str): The URL of the CSV file.
        sep (str, optional): Field delimiter for the CSV file. Default is ','.
        index_col (int, optional): Column to use as the row labels of the DataFrame. Default is None.

    Returns:
        DataFrame: The DataFrame containing the data read from the CSV file.
    
    Example:
        df = pdread_csv_fromurl('http://example.com/data.csv')
    """
    # Create a Curl object
    c = pycurl.Curl()

    # Set the URL
    c.setopt(pycurl.URL, url)

    # Create a BytesIO object to store the downloaded data
    buffer = BytesIO()
    c.setopt(pycurl.WRITEDATA, buffer)
    c.setopt(pycurl.FOLLOWLOCATION, 1)

    # Perform the request
    c.perform()
    # Check if the request was successful (HTTP status code 200)
    http_code = c.getinfo(pycurl.HTTP_CODE)
    if http_code == 200:
        buffer.seek(0)
        # load the downloaded data to a pandas dataframe
        df = pd.read_table(buffer, sep=sep, index_col=index_col)
    else:
        raise Exception(f"Problem opening HTTP request: code {http_code}")
    # Close the Curl object
    c.close()
    return df

def pdread_csv_fromurl_old(url, sep=',', index_col=None):
    """
    Reads a CSV file from a URL.

    Args:
        url (str): The URL of the CSV file.
        sep (str, optional): Field delimiter for the CSV file. Default is ','.
        index_col (int, optional): Column to use as the row labels of the DataFrame. Default is None.

    Returns:
        DataFrame: The DataFrame containing the data read from the CSV file.
    
    Example:
        df = pdread_csv_fromurl('http://example.com/data.csv')
    """
    from io import BytesIO

    crl_obj = pycurl.Curl()
    b_obj = BytesIO()
    crl_obj.setopt(crl_obj.URL, url)
    crl_obj.setopt(crl_obj.WRITEDATA, b_obj)
    crl_obj.perform()
    crl_obj.close()
    b_obj.seek(0)
    return pd.read_table(b_obj, sep=sep, index_col=index_col, encoding='utf-8')

def pandas_readcsv(filename, descr:str=None, disabled=False, chunksize=100, **kargs):
    """
    Reads a CSV file in chunks and displays a progress bar.

    Args:
        filename (str): The name of the CSV file to read.
        descr (str, optional): Description to display on the progress bar. Default is None.
        disabled (bool, optional): Disable the progress bar if True. Default is False.
        chunksize (int, optional): Number of rows per chunk. Default is 100.
        **kargs: Additional arguments for `pd.read_csv`.

    Returns:
        DataFrame: The DataFrame containing the data read from the CSV file.

    Example:
        df = pandas_readcsv('data.csv', descr='Loading data', chunksize=500)
    """
    # Get number of lines in file.
    with open(filename, 'r') as fp:
        try:
            has_headings = csv.Sniffer().has_header(fp.read(1024))
            lines = len(fp.readlines()) - 1
        except csv.Error:
            # The file seems to be empty
            lines = len(fp.readlines())

    # Read file in chunks, updating progress bar after each chunk.
    listdf = []
    with tqdm(total=lines, desc=descr, disable=disabled) as bar:
        for i, chunk in enumerate(pd.read_csv(filename, chunksize=chunksize, **kargs)):
            listdf.append(chunk)
            bar.update(chunksize)
    df = pd.concat(listdf, ignore_index=False)
    return df

def split_dataframe(df, chunk_size=10):
    """
    Splits a DataFrame into chunks.

    Args:
        df (DataFrame): The DataFrame to split.
        chunk_size (int, optional): Number of rows per chunk. Default is 10.

    Returns:
        list: A list of DataFrames, each containing a chunk of the original data.

    Example:
        chunks = split_dataframe(df, chunk_size=100)
    """
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks

def pandas_writecsv(filename, df: pd.DataFrame, chunksize=10, index=False, sep=',', descr:str=None, disabled=False):
    """
    Writes a DataFrame to a CSV file in chunks and displays a progress bar.

    Args:
        filename (str): The name of the CSV file to write.
        df (DataFrame): The DataFrame to write to the CSV file.
        chunksize (int, optional): Number of rows per chunk. Default is 10.
        index (bool, optional): Write row names (index). Default is False.
        sep (str, optional): Field delimiter for the CSV file. Default is ','.
        descr (str, optional): Description to display on the progress bar. Default is None.
        disabled (bool, optional): Disable the progress bar if True. Default is False.

    Returns:
        DataFrame: The original DataFrame.

    Example:
        pandas_writecsv('output.csv', df, descr='Saving data', chunksize=500)
    """
    # Write file in chunks, updating progress bar after each chunk.
    if len(df) > chunksize + 1:
        num_chunks = len(df) // chunksize + 1
    else:
        num_chunks = 1
    with tqdm(total=num_chunks, desc=descr, disable=disabled) as bar:
        for i, chunk in enumerate(split_dataframe(df, chunksize)):
            if i == 0:
                chunk.to_csv(filename, index=index, mode='w', sep=sep)
            else:
                chunk.to_csv(filename, index=index, header=False, mode='a', sep=sep)
            bar.update()
    return df
