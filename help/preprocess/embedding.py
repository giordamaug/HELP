import pandas as pd
import numpy as np
import networkx as nx
from karateclub import DeepWalk, Node2Vec, AE
from typing import Dict
from tqdm import tqdm

def PPI_embed(df_net: pd.DataFrame, method: str='Node2Vec', dimensions: int=128, 
              walk_number: int=10, walk_length: int=80, workers: int=1, epochs: int=1, learning_rate: float=0.05, seed: int=42,
              params: Dict={"p": 1.0, "q": 1.0, "window_size": 5,  "min_count": 1}, 
              source: str = 'A', target: str='B', weight: str='combined_score', verbose: bool=False):
    """
    Embeds a protein-protein interaction (PPI) network using graph embedding techniques.

    :df_net pd.DataFrame: The input DataFrame containing the PPI network information.
    :method str: The graph embedding method. Options: 'DeepWalk', 'Node2Vec', 'AE'. Default: 'Node2Vec'.
    :dimensions int: The dimensionality of the embedding. Default: 128.
    :walk_number int: Number of walks per node. Default: 10.
    :walk_length int: Length of each walk. Default: 80.
    :workers int: Number of parallel workers. Default: 4.
    :epochs int: Number of training epochs. Default: 1.
    :learning_rate float: Learning rate for the embedding model. Default: 0.05.
    :seed int: Random seed for reproducibility. Default: 42.
    :params Dict: Additional parameters for the embedding method. Default: {"p": 1.0, "q": 1.0, "window_size": 5, "min_count": 1}.
    :source str: Column name for the source nodes in the PPI network DataFrame. Default: 'A'.
    :target str: Column name for the target nodes in the PPI network DataFrame. Default: 'B'.
    :weight str: Column name for the edge weights in the PPI network DataFrame. Default: 'combined_score'.
    :verbose bool: Whether to print progress information. Default: False.

    :return: DataFrame containing the node embeddings.
    :rtype: pd.DataFrame

    :example:
    .. code-block:: python
        df_embedding = PPI_embed(ppi_data, method='Node2Vec', dimensions=128, epochs=5, verbose=True)
    """
    assert "method" in ['DeepWalk', 'Node2Vec', 'AE'], "Embedding method not supported!"
    params['dimensions'] = dimensions
    params['walk_length'] = walk_length
    params['walk_number'] = walk_number
    params['workers'] = workers
    params['epochs'] = epochs
    params['learning_rate'] = learning_rate
    params['seed'] = seed
    genes = np.union1d(df_net[source].values, df_net[target].values)
    idx2gene_mapping = dict(zip(np.arange(len(genes)), genes)) 
    gene2idx_mapping = dict(zip(genes, np.arange(len(genes))))             # create mapping index by gene name
    edge_list = np.array([(gene2idx_mapping[v[0]], gene2idx_mapping[v[1]]) for v in list(df_net[[source,target]].values)])
    edge_attr = df_net[[weight]].values.T.ravel()
    # Create the movies undirected graph.
    ppiG = nx.Graph()
    for pair, w in tqdm(zip(edge_list, edge_attr), total=len(edge_list),  desc="Creating the PPI graph"):
        ppiG.add_edge(*pair, weight=w)
        
    if verbose: print("Total number of graph nodes:", ppiG.number_of_nodes())
    if verbose: print("Total number of graph edges:", ppiG.number_of_edges())
    degrees = []
    for node in ppiG.nodes:
        degrees.append(ppiG.degree[node])
    if verbose: print("Average node degree:", round(sum(degrees) / len(degrees), 2))
    if verbose: print(f"There are {len(list(nx.isolates(ppiG)))} isolated genes")

    embedder = globals()[method](**params)
    if verbose: print(f"Embedding with {method}")
    embedder.fit(ppiG)
    embedding = n2v.get_embedding()
    df_emb = pd.DataFrame(embedding, 
             columns = [f'n2v{i}' for i in range(embedding.shape[1])], 
             index = [idx2gene_mapping[i] for i in range(len(genes))]) 
    return df_emb
