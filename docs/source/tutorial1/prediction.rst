Process the tissue attributes
=============================

In this code we load tissue gene attributes by several datafiles. We can
apply missin values fixing and data scaling.

.. code:: ipython3

    from help.preprocess.loaders import feature_assemble
    import os
    label_file = 'label_Kidney.csv'
    features = [{'fname': 'Kidney_BIO.csv', 'fixna' : True, 'normalize': 'std'},
                {'fname': 'Kidney_CCcfs.csv', 'fixna' : True, 'normalize': 'std'}
               ]
    df_X, df_y = feature_assemble(label_file = label_file, 
                               features=features, subsample=False, seed=1, saveflag=False, verbose=True)


.. parsed-literal::

    Loading label_Kidney.csv
    [Kidney_BIO.csv] found 0 Nan...
    [Kidney_BIO.csv] Normalization with std ...
    [Kidney_CCcfs.csv] found 0 Nan...
    [Kidney_CCcfs.csv] Normalization with std ...
    17577 labeled genes over a total of 18443
    (17577, 3331) data input


Prediction
==========

We process k-fold cross validation of a LightGBM classifier, and then
storing predictions andprinting metrics.

.. code:: ipython3

    from help.models.prediction import predict_cv
    predict_cv(df_X, df_y, n_splits=5, balanced=True, display=True, outfile='pred_Kidney.csv') 


.. parsed-literal::

    {'E': 0, 'NE': 1}
    label
    NE       16365
    E         1212
    Name: count, dtype: int64


.. parsed-literal::

    5-fold: 100%|██████████████████████████████████████████████████████████████████| 5/5 [00:25<00:00,  5.07s/it]




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
          <td>0.9485±0.0054</td>
        </tr>
        <tr>
          <th>Accuracy</th>
          <td>0.9427±0.0031</td>
        </tr>
        <tr>
          <th>BA</th>
          <td>0.8378±0.0148</td>
        </tr>
        <tr>
          <th>Sensitivity</th>
          <td>0.7161±0.0285</td>
        </tr>
        <tr>
          <th>Specificity</th>
          <td>0.9595±0.0015</td>
        </tr>
        <tr>
          <th>MCC</th>
          <td>0.6070±0.0241</td>
        </tr>
        <tr>
          <th>CM</th>
          <td>[[868, 344], [663, 15702]]</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: output_19_3.png

