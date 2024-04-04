1. Install HELP from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip this cell if you already have installed HELP.

.. code:: ipython3

    !pip install git+https://github.com/giordamaug/HELP.git

2. Download the input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a chosen tissue (here ``Kidney``), download from GitHub the label
file (here ``Kidney_HELP.csv``, computed as in Example 1) and the
attribute files (here BIO ``Kidney_BIO.csv``, CCcfs
``Kidney_CCcfs_1.csv``, …, ``Kidney_CCcfs_5.csv``, and N2V
``Kidney_EmbN2V_128.csv``).

Skip this step if you already have these input files locally.

.. code:: ipython3

    tissue='Kidney'
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_HELP.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_BIO.csv
    for i in range(5):
      !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_CCcfs_{i}.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_EmbN2V_128.csv

Observe that the CCcfs file has been subdivided into 5 separate files
for storage limitations on GitHub.

3. Load the input files and process the tissue attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  The label file (``Kidney_HELP.csv``) can be loaded via ``read_csv``;
   its three-class labels (``E``, ``aE``, ``sNE``) are converted to
   two-class labels (``E``, ``NE``);

-  The tissue gene attributes are loaded and assembled via
   ``feature_assemble_df`` using the downloaded datafiles BIO, CCcfs
   subdivided into 5 subfiles (``'nchunks': 5``) and embedding. We do
   not apply missing values fixing (``'fixna': False``), while we do
   apply data scaling (``'normalize': 'std'``) to the BIO and CCcfs
   attributes.

.. code:: ipython3

    tissue='Kidney'
    import pandas as pd
    from HELPpy.preprocess.loaders import feature_assemble_df
    df_y = pd.read_csv(f"{tissue}_HELP.csv", index_col=0)
    df_y = df_y.replace({'aE': 'NE', 'sNE': 'NE'})
    print(df_y.value_counts(normalize=False))
    features = [{'fname': f'{tissue}_BIO.csv', 'fixna' : False, 'normalize': 'std'},
                {'fname': f'{tissue}_CCcfs.csv', 'fixna' : False, 'normalize': 'std', 'nchunks' : 5},
                {'fname': f'{tissue}_EmbN2V_128.csv', 'fixna' : False, 'normalize': None}]
    df_X, df_y = feature_assemble_df(df_y, features=features, saveflag=False, verbose=True)


.. parsed-literal::

    label
    NE       16678
    E         1253
    Name: count, dtype: int64
    Majority NE 16678 minority E 1253
    [Kidney_BIO.csv] found 52532 Nan...
    [Kidney_BIO.csv] Normalization with std ...


.. parsed-literal::

    Loading file in chunks: 100%|██████████| 5/5 [00:02<00:00,  2.05it/s]


.. parsed-literal::

    [Kidney_CCcfs.csv] found 6676644 Nan...
    [Kidney_CCcfs.csv] Normalization with std ...
    [Kidney_EmbN2V_128.csv] found 0 Nan...
    [Kidney_EmbN2V_128.csv] No normalization...
    17236 labeled genes over a total of 17931
    (17236, 3456) data input


4. Estimate the performance of EGs prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instantiate the prediction model described in the HELP paper
(soft-voting ensemble ``VotingSplitClassifier`` of ``n_voters=10``
classifiers) and estimate its performance via 5-fold cross-validation
(``k_fold_cv`` with ``n_splits=5``). Then, print the obtained average
performances (``df_scores``)…

.. code:: ipython3

    from HELPpy.models.prediction import VotingSplitClassifier, k_fold_cv
    clf = VotingSplitClassifier(n_voters=10, n_jobs=-1, random_state=-1)
    df_scores, scores, predictions = k_fold_cv(df_X, df_y, clf, n_splits=5, seed=0, verbose=True)
    df_scores


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       16010
    E         1224
    Name: count, dtype: int64
    Classification with VotingSplitClassifier...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [01:15<00:00, 15.08s/it]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>measure</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ROC-AUC</th>
          <td>0.9584±0.0043</td>
        </tr>
        <tr>
          <th>Accuracy</th>
          <td>0.8848±0.0025</td>
        </tr>
        <tr>
          <th>BA</th>
          <td>0.8939±0.0070</td>
        </tr>
        <tr>
          <th>Sensitivity</th>
          <td>0.9044±0.0156</td>
        </tr>
        <tr>
          <th>Specificity</th>
          <td>0.8833±0.0031</td>
        </tr>
        <tr>
          <th>MCC</th>
          <td>0.5354±0.0079</td>
        </tr>
        <tr>
          <th>CM</th>
          <td>[[1107, 117], [1868, 14142]]</td>
        </tr>
      </tbody>
    </table>
    </div>



… and those in each fold (``scores``)

.. code:: ipython3

    scores




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ROC-AUC</th>
          <th>Accuracy</th>
          <th>BA</th>
          <th>Sensitivity</th>
          <th>Specificity</th>
          <th>MCC</th>
          <th>CM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.954258</td>
          <td>0.878735</td>
          <td>0.889496</td>
          <td>0.902041</td>
          <td>0.876952</td>
          <td>0.522809</td>
          <td>[[221, 24], [394, 2808]]</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.953289</td>
          <td>0.873223</td>
          <td>0.894068</td>
          <td>0.918367</td>
          <td>0.869769</td>
          <td>0.520189</td>
          <td>[[225, 20], [417, 2785]]</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.955901</td>
          <td>0.884827</td>
          <td>0.890891</td>
          <td>0.897959</td>
          <td>0.883823</td>
          <td>0.532617</td>
          <td>[[220, 25], [372, 2830]]</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.960578</td>
          <td>0.882507</td>
          <td>0.895296</td>
          <td>0.910204</td>
          <td>0.880387</td>
          <td>0.533671</td>
          <td>[[223, 22], [383, 2819]]</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.965238</td>
          <td>0.880731</td>
          <td>0.901747</td>
          <td>0.926230</td>
          <td>0.877264</td>
          <td>0.536883</td>
          <td>[[226, 18], [393, 2809]]</td>
        </tr>
      </tbody>
    </table>
    </div>



Show labels, predictions and their probabilities (``predictions``) and
save them in a csv file

.. code:: ipython3

    predictions




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>label</th>
          <th>prediction</th>
          <th>probabilities</th>
        </tr>
        <tr>
          <th>gene</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A2M</th>
          <td>1</td>
          <td>1</td>
          <td>0.016435</td>
        </tr>
        <tr>
          <th>A2ML1</th>
          <td>1</td>
          <td>1</td>
          <td>0.001649</td>
        </tr>
        <tr>
          <th>AAGAB</th>
          <td>1</td>
          <td>1</td>
          <td>0.230005</td>
        </tr>
        <tr>
          <th>AANAT</th>
          <td>1</td>
          <td>1</td>
          <td>0.002823</td>
        </tr>
        <tr>
          <th>AARS2</th>
          <td>1</td>
          <td>0</td>
          <td>0.529173</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>ZSCAN9</th>
          <td>1</td>
          <td>1</td>
          <td>0.004752</td>
        </tr>
        <tr>
          <th>ZSWIM6</th>
          <td>1</td>
          <td>1</td>
          <td>0.007049</td>
        </tr>
        <tr>
          <th>ZUP1</th>
          <td>1</td>
          <td>0</td>
          <td>0.532555</td>
        </tr>
        <tr>
          <th>ZYG11A</th>
          <td>1</td>
          <td>1</td>
          <td>0.005995</td>
        </tr>
        <tr>
          <th>ZZEF1</th>
          <td>1</td>
          <td>1</td>
          <td>0.075781</td>
        </tr>
      </tbody>
    </table>
    <p>17234 rows × 3 columns</p>
    </div>



.. code:: ipython3

    predictions.to_csv(f"csEGs_{tissue}_EvsNE.csv", index=True)

5. Compute TPR for ucsEGs and csEGs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the result files for ucsEGs (``ucsEG_Kidney.txt``) and csEGs
(``csEGs_Kidney_EvsNE.csv``) already computed for the tissue, compute
the TPRs (tpr) and show their bar plot.

.. code:: ipython3

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    labels = []
    data = []
    tpr = []
    genes = {}
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/ucsEG_{tissue}.txt
    ucsEGs = pd.read_csv(f"ucsEG_{tissue}.txt", index_col=0, header=None).index.values
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/csEGs_{tissue}_EvsNE.csv
    predictions = pd.read_csv(f"csEGs_{tissue}_EvsNE.csv", index_col=0)
    indices = np.intersect1d(ucsEGs, predictions.index.values)
    preds = predictions.loc[indices]
    num1 = len(preds[preds['label'] == preds['prediction']])
    den1 = len(preds[preds['label'] == 0])
    den2 = len(predictions[predictions['label'] == 0])
    num2 = len(predictions[(predictions['label'] == 0) & (predictions['label'] == predictions['prediction'])])
    labels += [f"ucsEGs\n{tissue}", f"csEGs\n{tissue}"]
    data += [float(f"{num1 /den1:.3f}"), float(f"{num2 /den2:.3f}")]
    tpr += [f"{num1}/{den1}", f"{num2}/{den2}"]
    genes[f'ucsEGs_{tissue}_y'] = preds[preds['label'] == preds['prediction']].index.values
    genes[f'ucsEGs_{tissue}_n'] = preds[preds['label'] != preds['prediction']].index.values
    genes[f'csEGs_{tissue}_y'] = predictions[(predictions['label'] == 0) & (predictions['label'] == predictions['prediction'])].index.values
    genes[f'csEGs_{tissue}_n'] = predictions[(predictions['label'] == 0) & (predictions['label'] != predictions['prediction'])].index.values
    print(f"ucsEG {tissue} TPR = {num1 /den1:.3f} ({num1}/{den1}) ucsEG {tissue} TPR =  {num2/den2:.3f} ({num2}/{den2})")
    
    f, ax = plt.subplots(figsize=(4, 4))
    palette = sns.color_palette("pastel", n_colors=2)
    sns.barplot(y = data, x = labels, ax=ax, hue= data, palette = palette, orient='v', legend=False)
    ax.set_ylabel('TPR')
    ax.set(yticklabels=[])
    for i,l,t in zip(range(4),labels,tpr):
        ax.text(-0.15 + (i * 1.03), 0.2, f"({t})", rotation='vertical')
    for i in ax.containers:
        ax.bar_label(i,)


.. parsed-literal::

    zsh:1: command not found: wget
    zsh:1: command not found: wget
    ucsEG Kidney TPR = 0.780 (46/59) ucsEG Kidney TPR =  0.897 (1114/1242)



.. image:: output_17_1.png


This code can be used to produce Fig 5(B) of the HELP paper by executing
an iteration cycle for both ``kidney`` and ``lung`` tissues.

At the end, we print the list of ucs_EGs for the tissue.

.. code:: ipython3

    genes[f'ucsEGs_{tissue}_y']




.. parsed-literal::

    array(['ACTG1', 'ACTR6', 'ARF4', 'ARPC4', 'CDK6', 'CHMP7', 'COPS3',
           'DCTN3', 'DDX11', 'DDX52', 'EMC3', 'EXOSC1', 'GEMIN7', 'GET3',
           'HGS', 'HTATSF1', 'KIF4A', 'MCM10', 'MDM2', 'METAP2', 'MLST8',
           'NCAPH2', 'NDOR1', 'OXA1L', 'PFN1', 'PIK3C3', 'PPIE', 'PPP1CA',
           'PPP4R2', 'RAB7A', 'RAD1', 'RBM42', 'RBMX2', 'RTEL1', 'SNRPB2',
           'SPTLC1', 'SRSF10', 'TAF1D', 'TMED10', 'TMED2', 'UBA5', 'UBC',
           'UBE2D3', 'USP10', 'VPS52', 'YWHAZ'], dtype=object)


