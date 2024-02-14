.. code:: ipython3

    datapath = "<your-data-path>"

Process the tissue attributes
=============================

In this code we load tissue gene attributes by several datafiles. We
apply missing values fixing and data scaling with
``sklearn.preprocessing.StandardScaler`` on the ``BIO`` and ``CCcfs``
attributes, while no normalization and fixing on embedding attributes
(``EmbN2V_128``). The attributes are all merged in one matrix by the
``feature_assemble`` function as input for the prediction model
building.

.. code:: ipython3

    import pandas as pd
    from help.preprocess.loaders import feature_assemble
    import os
    label_file = os.path.join(datapath, 'label_Kidney.csv')
    features = [{'fname': os.path.join(datapath, 'Kidney_BIO.csv'), 'fixna' : True, 'normalize': 'std'},
                {'fname': os.path.join(datapath, 'Kidney_CCcfs.csv'), 'fixna' : True, 'normalize': 'std'},
                {'fname': os.path.join(datapath, 'Kidney_EmbN2V_128.csv'), 'fixna' : None, 'normalize': None}]
    df_X, df_y = feature_assemble(label_file = label_file, 
                                  features=features, subsample=False, seed=1, saveflag=False, verbose=True)
    pd.merge(df_X, df_y, left_index=True, right_index=True, how='outer')


.. parsed-literal::

    Loading ../data/label_Kidney.csv
    Majority NE 16678 minoriy E 1253
    [Kidney_BIO.csv] found 0 Nan...
    [Kidney_BIO.csv] Normalization with std ...
    [Kidney_CCcfs.csv] found 0 Nan...
    [Kidney_CCcfs.csv] Normalization with std ...
    [Kidney_EmbN2V_128.csv] No normalization...
    17236 labeled genes over a total of 17931
    (17236, 3459) data input




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
          <td>0.651992</td>
          <td>0.000002</td>
          <td>0.084365</td>
          <td>0.038663</td>
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
          <td>0.021528</td>
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
          <td>0.653681</td>
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
          <td>0.018692</td>
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
          <td>0.019355</td>
          <td>0.634761</td>
          <td>0.000055</td>
          <td>0.059024</td>
          <td>0.029878</td>
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
          <td>0.016435</td>
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
          <td>0.022104</td>
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
          <td>0.044248</td>
          <td>0.657491</td>
          <td>0.000121</td>
          <td>0.093023</td>
          <td>0.010256</td>
          <td>0.000000</td>
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
          <td>0.649737</td>
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
    <p>17236 rows × 3460 columns</p>
    </div>



Prediction
==========

We process k-fold cross validation of a LightGBM classifier
(``n_splits=5``), and then storing predictions and printing metrics.

.. code:: ipython3

    from help.models.prediction import predict_cv
    predict_cv(df_X, df_y, n_splits=5, balanced=True, display=True, outfile='pred_Kidney.csv') 


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       15994
    E         1242
    Name: count, dtype: int64


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:33<00:00,  6.77s/it]




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
          <td>0.0434±0.0055</td>
        </tr>
        <tr>
          <th>Accuracy</th>
          <td>0.9476±0.0026</td>
        </tr>
        <tr>
          <th>BA</th>
          <td>0.8344±0.0085</td>
        </tr>
        <tr>
          <th>Sensitivity</th>
          <td>0.7021±0.0173</td>
        </tr>
        <tr>
          <th>Specificity</th>
          <td>0.9667±0.0027</td>
        </tr>
        <tr>
          <th>MCC</th>
          <td>0.6321±0.0152</td>
        </tr>
        <tr>
          <th>CM</th>
          <td>[[872, 370], [533, 15461]]</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: output_4_3.png


Prediction with undersampling
=============================

Due to the strong unbalancing between the two classes, we can redo
prediction model building by undersampling the majority class: this is
done by re-applying the ``feature_assemble`` function with parameter
``subsample=True``: this flag set causes the majority class to be
downsampled to 4 times the dimension of the minority class. The we
re-apply the k-fold cross validation of the a LightGBM classifier.

.. code:: ipython3

    df_X, df_y = feature_assemble(label_file = label_file, 
                                  features=features, subsample=True, seed=1, verbose=True)
    predict_cv(df_X, df_y, n_splits=5, balanced=True, display=True, outfile='pred_Kidney.csv') 


.. parsed-literal::

    Loading ../data/label_Kidney.csv
    Majority NE 16678 minoriy E 1253
    [Kidney_BIO.csv] found 0 Nan...
    [Kidney_BIO.csv] Normalization with std ...
    [Kidney_CCcfs.csv] found 0 Nan...
    [Kidney_CCcfs.csv] Normalization with std ...
    [Kidney_EmbN2V_128.csv] No normalization...
    6043 labeled genes over a total of 6265
    (6043, 3459) data input
    {'E': 0, 'NE': 1}
    label
    NE       4801
    E        1242
    Name: count, dtype: int64


.. parsed-literal::

    5-fold: 100%|██████████| 5/5 [00:18<00:00,  3.79s/it]




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
          <td>0.0466±0.0032</td>
        </tr>
        <tr>
          <th>Accuracy</th>
          <td>0.9136±0.0097</td>
        </tr>
        <tr>
          <th>BA</th>
          <td>0.8689±0.0213</td>
        </tr>
        <tr>
          <th>Sensitivity</th>
          <td>0.7930±0.0425</td>
        </tr>
        <tr>
          <th>Specificity</th>
          <td>0.9448±0.0060</td>
        </tr>
        <tr>
          <th>MCC</th>
          <td>0.7361±0.0329</td>
        </tr>
        <tr>
          <th>CM</th>
          <td>[[985, 257], [265, 4536]]</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: output_6_3.png

