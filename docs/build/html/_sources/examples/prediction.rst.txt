1. Install HELP from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip this cell if you already have installed HELP.

.. code:: ipython3

    !pip install git+https://github.com/giordamaug/HELP.git

2. Download the input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this cell we download from GitHub repository the label file and the
attribute files. Skip this step if you already have these input files
locally.

.. code:: ipython3

    tissue='Kidney'
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_HELP.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_BIO.csv
    for i in range(5):
      !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_CCcfs_{i}.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_EmbN2V_128.csv

3. Process the tissue attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this code we load tissue gene attributes by several datafiles. We
apply missing values fixing and data scaling with
``sklearn.preprocessing.StandardScaler`` on the ``BIO`` and ``CCcfs``
attributes, while no normalization and fixing on embedding attributes
(``EmbN2V_128``). The attributes are all merged in one matrix by the
``feature_assemble`` function as input for the prediction model
building.

.. code:: ipython3

    tissue='Kidney'
    import pandas as pd
    from HELPpy.preprocess.loaders import feature_assemble_df
    import os
    df_y = pd.read_csv(f"{tissue}_HELP.csv", index_col=0)
    df_y = df_y.replace({'aE': 'NE', 'sNE': 'NE'})    # E vs NE problem
    #df_y = df_y[df_y['label'].isin(['aE']) == False]   # E vs sNE problem
    #df_y = df_y[df_y['label'].isin(['E']) == False]    # aE vs sNE problem
    #df_y = df_y[df_y['label'].isin(['sNE']) == False]  # E vs aE problem
    print(df_y.value_counts(normalize=False))
    features = [{'fname': f'{tissue}_BIO.csv', 'fixna' : False, 'normalize': 'std'},
                {'fname': f'{tissue}_CCcfs.csv', 'fixna' : False, 'normalize': 'std', 'nchunks' : 5},
                {'fname': f'{tissue}_EmbN2V_128.csv', 'fixna' : False, 'normalize': None}
                ]
    df_X, df_y = feature_assemble_df(df_y, features=features, verbose=True)
    print(df_y.value_counts(normalize=False))
    pd.merge(df_X, df_y, left_index=True, right_index=True, how='outer')


.. parsed-literal::

    label
    NE       16678
    E         1253
    dtype: int64
    Majority NE 16678 minority E 1253
    [Kidney_BIO.csv] found 52532 Nan...
    [Kidney_BIO.csv] Normalization with std ...


.. parsed-literal::

    Loading file in chunks: 100%|██████████| 5/5 [00:08<00:00,  1.77s/it]


.. parsed-literal::

    [Kidney_CCcfs.csv] found 6676644 Nan...
    [Kidney_CCcfs.csv] Normalization with std ...
    [Kidney_EmbN2V_128.csv] found 0 Nan...
    [Kidney_EmbN2V_128.csv] No normalization...
    17236 labeled genes over a total of 17931
    (17236, 3456) data input
    label
    NE       15994
    E         1242
    dtype: int64




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
          <th>Gene length</th>
          <th>Transcripts count</th>
          <th>GC content</th>
          <th>GTEX_kidney</th>
          <th>Gene-Disease association</th>
          <th>OncoDB_expression</th>
          <th>HPA_kidney</th>
          <th>GO-MF</th>
          <th>GO-BP</th>
          <th>GO-CC</th>
          <th>...</th>
          <th>Node2Vec_119</th>
          <th>Node2Vec_120</th>
          <th>Node2Vec_121</th>
          <th>Node2Vec_122</th>
          <th>Node2Vec_123</th>
          <th>Node2Vec_124</th>
          <th>Node2Vec_125</th>
          <th>Node2Vec_126</th>
          <th>Node2Vec_127</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A1BG</th>
          <td>0.003351</td>
          <td>0.020942</td>
          <td>0.501832</td>
          <td>2.044542e-05</td>
          <td>0.002950</td>
          <td>NaN</td>
          <td>0.000002</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.115385</td>
          <td>...</td>
          <td>0.120922</td>
          <td>-0.352630</td>
          <td>0.580697</td>
          <td>-0.659300</td>
          <td>-1.320486</td>
          <td>1.019308</td>
          <td>-0.469064</td>
          <td>0.123211</td>
          <td>0.557266</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>A1CF</th>
          <td>0.034865</td>
          <td>0.047120</td>
          <td>0.160530</td>
          <td>1.980884e-05</td>
          <td>NaN</td>
          <td>0.556939</td>
          <td>0.000232</td>
          <td>0.069767</td>
          <td>0.041026</td>
          <td>0.096154</td>
          <td>...</td>
          <td>-1.162494</td>
          <td>0.155702</td>
          <td>-1.162071</td>
          <td>0.534082</td>
          <td>0.798872</td>
          <td>0.149595</td>
          <td>-0.360515</td>
          <td>-1.060540</td>
          <td>-0.408493</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>A2M</th>
          <td>0.019624</td>
          <td>0.062827</td>
          <td>0.176932</td>
          <td>3.377232e-03</td>
          <td>0.073746</td>
          <td>0.584540</td>
          <td>0.005382</td>
          <td>0.302326</td>
          <td>0.056410</td>
          <td>0.076923</td>
          <td>...</td>
          <td>0.150766</td>
          <td>1.492019</td>
          <td>0.209449</td>
          <td>-1.034729</td>
          <td>-0.064318</td>
          <td>0.029690</td>
          <td>0.138344</td>
          <td>0.806095</td>
          <td>-0.496128</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>A2ML1</th>
          <td>0.026017</td>
          <td>0.041885</td>
          <td>0.299948</td>
          <td>5.123403e-07</td>
          <td>0.017699</td>
          <td>NaN</td>
          <td>0.000000</td>
          <td>0.069767</td>
          <td>0.005128</td>
          <td>0.038462</td>
          <td>...</td>
          <td>0.191344</td>
          <td>-0.542462</td>
          <td>0.746510</td>
          <td>0.082089</td>
          <td>-1.109212</td>
          <td>0.406936</td>
          <td>-1.332319</td>
          <td>-0.363864</td>
          <td>0.443284</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>A3GALT2</th>
          <td>0.005784</td>
          <td>0.000000</td>
          <td>0.473739</td>
          <td>1.421472e-06</td>
          <td>NaN</td>
          <td>0.663540</td>
          <td>0.000000</td>
          <td>0.069767</td>
          <td>0.015385</td>
          <td>0.057692</td>
          <td>...</td>
          <td>0.483003</td>
          <td>-0.197605</td>
          <td>0.164332</td>
          <td>0.040729</td>
          <td>-0.552362</td>
          <td>0.242761</td>
          <td>0.223486</td>
          <td>0.017539</td>
          <td>-0.526580</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>ZYG11A</th>
          <td>0.021209</td>
          <td>0.010471</td>
          <td>0.288257</td>
          <td>7.073108e-06</td>
          <td>NaN</td>
          <td>0.634761</td>
          <td>0.000055</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.000000</td>
          <td>...</td>
          <td>-0.717935</td>
          <td>-0.072597</td>
          <td>0.585837</td>
          <td>0.172081</td>
          <td>-0.278010</td>
          <td>0.170799</td>
          <td>0.267462</td>
          <td>-0.211294</td>
          <td>-0.940943</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>ZYG11B</th>
          <td>0.040775</td>
          <td>0.005236</td>
          <td>0.248648</td>
          <td>7.271294e-05</td>
          <td>NaN</td>
          <td>0.646090</td>
          <td>0.000238</td>
          <td>0.000000</td>
          <td>0.005128</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.372134</td>
          <td>0.007040</td>
          <td>-0.278071</td>
          <td>-1.309595</td>
          <td>-0.352476</td>
          <td>0.732887</td>
          <td>0.156505</td>
          <td>0.516706</td>
          <td>-0.412953</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>ZYX</th>
          <td>0.003958</td>
          <td>0.047120</td>
          <td>0.539522</td>
          <td>8.282866e-04</td>
          <td>NaN</td>
          <td>0.672638</td>
          <td>0.000177</td>
          <td>0.046512</td>
          <td>0.035897</td>
          <td>0.153846</td>
          <td>...</td>
          <td>-0.316321</td>
          <td>-0.382132</td>
          <td>0.400354</td>
          <td>0.322564</td>
          <td>0.400369</td>
          <td>0.188850</td>
          <td>0.593201</td>
          <td>-0.093008</td>
          <td>-0.508902</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>ZZEF1</th>
          <td>0.056017</td>
          <td>0.052356</td>
          <td>0.304484</td>
          <td>9.626291e-05</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.000121</td>
          <td>0.093023</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>-0.520060</td>
          <td>-0.000595</td>
          <td>-0.101278</td>
          <td>-0.468345</td>
          <td>0.240905</td>
          <td>-0.124018</td>
          <td>0.568793</td>
          <td>-0.422793</td>
          <td>-0.701705</td>
          <td>NE</td>
        </tr>
        <tr>
          <th>ZZZ3</th>
          <td>0.048909</td>
          <td>0.052356</td>
          <td>0.176758</td>
          <td>7.179946e-05</td>
          <td>0.000000</td>
          <td>NaN</td>
          <td>0.000267</td>
          <td>0.093023</td>
          <td>0.051282</td>
          <td>0.057692</td>
          <td>...</td>
          <td>-0.348640</td>
          <td>-0.423926</td>
          <td>-0.078769</td>
          <td>0.163239</td>
          <td>-0.302664</td>
          <td>0.505735</td>
          <td>0.001912</td>
          <td>0.406448</td>
          <td>-0.296505</td>
          <td>NE</td>
        </tr>
      </tbody>
    </table>
    <p>17236 rows × 3457 columns</p>
    </div>



4. Prediction
~~~~~~~~~~~~~

We process k-fold cross validation of a LightGBM classifier
(``n_splits=5``), and then we store predictions and print metrics.

.. code:: ipython3

    from HELPpy.models.prediction import predict_cv_sv
    df_scores, predictions = predict_cv_sv(df_X, df_y, n_voters=10, n_splits=5, balanced=True, verbose=True)
    df_scores


.. parsed-literal::

    Majority NE 15994, minority E 1242
    {'E': 0, 'NE': 1}
    label
    NE       1600
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:50<00:00, 10.05s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1600
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:51<00:00, 10.31s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1600
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:52<00:00, 10.60s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1600
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:51<00:00, 10.29s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:47<00:00,  9.44s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:49<00:00,  9.82s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:52<00:00, 10.57s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:49<00:00,  9.89s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:47<00:00,  9.55s/it]


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       1599
    E        1242
    dtype: int64
    Classification with LGBM...


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:41<00:00,  8.38s/it]




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
          <th>42</th>
          <td>0.963653</td>
          <td>0.906707</td>
          <td>0.905173</td>
          <td>0.903382</td>
          <td>0.906965</td>
          <td>0.584557</td>
          <td>[[1122, 120], [1488, 14506]]</td>
        </tr>
      </tbody>
    </table>
    </div>



5. True Positive rates of context-specific EGs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import numpy as np
    csEGs = pd.read_csv("csEG_Kidney.txt", index_col=0, header=None).index.values
    indices = np.intersect1d(csEGs, predictions.index.values)
    predictions = predictions.loc[indices]
    num = len(predictions[predictions['label'] == predictions['prediction']])
    den = len(predictions)
    print(f"csEG Kidney TPR = {num /den:.3f} ({num}/{den})")


.. parsed-literal::

    csEG Kidney TPR = 0.780 (46/59)

