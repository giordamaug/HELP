Load the PPI netowkr and apply embedding
========================================

.. code:: ipython3

    import pandas as pd
    from help.preprocess.embedding import PPI_embed
    df_net = pd.read_csv('../data/Kidney_PPI.csv')
    df_embed = PPI_embed(df_net, method="Node2Vec", verbose=True)


.. parsed-literal::

    Embedding with Node2Vec and params:
    walk_number: 10
    walk_length: 80
    p: 1.0
    q: 1.0
    dimensions: 128
    workers: 1
    window_size: 5
    epochs: 1
    learning_rate: 0.05
    min_count: 1
    seed: 42


.. parsed-literal::

    Creating the PPI graph: 100%|██████████| 1110251/1110251 [00:03<00:00, 315253.50it/s]


.. parsed-literal::

    Total number of graph nodes: 19314
    Total number of graph edges: 1107128
    Average node degree: 114.65
    There are 0 isolated genes


Save the embedding and display it
=================================

.. code:: ipython3

    df_embed.to_csv('../data/Kidney_EmbN2V_128.csv')
    df_embed




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
          <th>Node2Vec_0</th>
          <th>Node2Vec_1</th>
          <th>Node2Vec_2</th>
          <th>Node2Vec_3</th>
          <th>Node2Vec_4</th>
          <th>Node2Vec_5</th>
          <th>Node2Vec_6</th>
          <th>Node2Vec_7</th>
          <th>Node2Vec_8</th>
          <th>Node2Vec_9</th>
          <th>...</th>
          <th>Node2Vec_118</th>
          <th>Node2Vec_119</th>
          <th>Node2Vec_120</th>
          <th>Node2Vec_121</th>
          <th>Node2Vec_122</th>
          <th>Node2Vec_123</th>
          <th>Node2Vec_124</th>
          <th>Node2Vec_125</th>
          <th>Node2Vec_126</th>
          <th>Node2Vec_127</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>(clone tec14)</th>
          <td>0.023051</td>
          <td>0.014992</td>
          <td>-0.141039</td>
          <td>-0.158685</td>
          <td>0.157265</td>
          <td>0.154793</td>
          <td>-0.067947</td>
          <td>0.118270</td>
          <td>-0.177724</td>
          <td>0.292236</td>
          <td>...</td>
          <td>-0.211920</td>
          <td>-0.156176</td>
          <td>-0.124871</td>
          <td>-0.182223</td>
          <td>0.127756</td>
          <td>0.011527</td>
          <td>0.052182</td>
          <td>-0.061288</td>
          <td>-0.155626</td>
          <td>0.016534</td>
        </tr>
        <tr>
          <th>100 kDa coactivator</th>
          <td>0.261170</td>
          <td>-0.524495</td>
          <td>-0.051630</td>
          <td>0.285273</td>
          <td>-0.548887</td>
          <td>0.205792</td>
          <td>-0.112976</td>
          <td>-0.213943</td>
          <td>-0.377048</td>
          <td>0.065116</td>
          <td>...</td>
          <td>0.150574</td>
          <td>0.186826</td>
          <td>-0.177846</td>
          <td>-0.071855</td>
          <td>0.519840</td>
          <td>-0.550491</td>
          <td>0.041066</td>
          <td>-0.198094</td>
          <td>0.082833</td>
          <td>0.122187</td>
        </tr>
        <tr>
          <th>14-3-3 tau splice variant</th>
          <td>0.365579</td>
          <td>-0.080783</td>
          <td>0.243886</td>
          <td>0.021987</td>
          <td>-0.737077</td>
          <td>0.425155</td>
          <td>0.259391</td>
          <td>-0.252958</td>
          <td>0.281372</td>
          <td>0.527371</td>
          <td>...</td>
          <td>-0.141555</td>
          <td>0.131767</td>
          <td>-0.273269</td>
          <td>-0.154002</td>
          <td>0.216473</td>
          <td>-0.345973</td>
          <td>0.184679</td>
          <td>-0.043953</td>
          <td>0.128388</td>
          <td>-0.131168</td>
        </tr>
        <tr>
          <th>3'-phosphoadenosine-5'-phosphosulfate synthase</th>
          <td>0.178820</td>
          <td>-0.643468</td>
          <td>-0.027346</td>
          <td>-0.015240</td>
          <td>-0.280513</td>
          <td>-0.151708</td>
          <td>-0.250100</td>
          <td>-0.428457</td>
          <td>-0.844148</td>
          <td>0.185542</td>
          <td>...</td>
          <td>0.197366</td>
          <td>0.177569</td>
          <td>0.024677</td>
          <td>0.238864</td>
          <td>0.222474</td>
          <td>-0.353596</td>
          <td>0.237262</td>
          <td>-0.401935</td>
          <td>-0.266909</td>
          <td>0.137892</td>
        </tr>
        <tr>
          <th>3-beta-hydroxysteroid dehydrogenase</th>
          <td>0.210142</td>
          <td>0.390334</td>
          <td>-0.198415</td>
          <td>-0.352966</td>
          <td>-0.268311</td>
          <td>-0.065822</td>
          <td>0.061606</td>
          <td>-0.182339</td>
          <td>-0.294295</td>
          <td>-0.155456</td>
          <td>...</td>
          <td>-0.008944</td>
          <td>-0.148080</td>
          <td>0.073030</td>
          <td>-0.238127</td>
          <td>-0.019748</td>
          <td>0.046764</td>
          <td>0.049918</td>
          <td>-0.227539</td>
          <td>0.161030</td>
          <td>-0.244953</td>
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
          <th>pp10122</th>
          <td>-0.028681</td>
          <td>0.066035</td>
          <td>0.176171</td>
          <td>-0.432947</td>
          <td>0.222461</td>
          <td>0.190343</td>
          <td>-0.338676</td>
          <td>-0.341797</td>
          <td>0.197020</td>
          <td>0.137798</td>
          <td>...</td>
          <td>-0.202883</td>
          <td>0.005556</td>
          <td>-0.788547</td>
          <td>-0.340978</td>
          <td>0.130789</td>
          <td>0.051332</td>
          <td>-0.064851</td>
          <td>0.139500</td>
          <td>0.943976</td>
          <td>-0.461141</td>
        </tr>
        <tr>
          <th>tRNA-uridine aminocarboxypropyltransferase</th>
          <td>0.631674</td>
          <td>-0.199346</td>
          <td>-0.048313</td>
          <td>-0.281030</td>
          <td>-0.186646</td>
          <td>0.408410</td>
          <td>-0.364784</td>
          <td>0.122930</td>
          <td>-0.137979</td>
          <td>-0.195326</td>
          <td>...</td>
          <td>-0.105558</td>
          <td>-0.139158</td>
          <td>0.378296</td>
          <td>0.150751</td>
          <td>-0.083709</td>
          <td>0.009086</td>
          <td>0.042860</td>
          <td>0.004699</td>
          <td>0.214054</td>
          <td>-0.348393</td>
        </tr>
        <tr>
          <th>tmp_locus_54</th>
          <td>0.438675</td>
          <td>0.066111</td>
          <td>-0.017400</td>
          <td>-0.629313</td>
          <td>-0.198265</td>
          <td>0.144987</td>
          <td>-1.218005</td>
          <td>0.147556</td>
          <td>-0.902313</td>
          <td>0.258718</td>
          <td>...</td>
          <td>-0.397593</td>
          <td>0.626247</td>
          <td>-1.075213</td>
          <td>0.370802</td>
          <td>-1.487310</td>
          <td>0.186115</td>
          <td>0.881203</td>
          <td>-0.247443</td>
          <td>-0.532058</td>
          <td>-0.325482</td>
        </tr>
        <tr>
          <th>urf-ret</th>
          <td>0.176243</td>
          <td>-0.385673</td>
          <td>-0.027699</td>
          <td>-0.587800</td>
          <td>-0.083288</td>
          <td>-0.012951</td>
          <td>-0.432041</td>
          <td>-0.195146</td>
          <td>0.121513</td>
          <td>-0.024590</td>
          <td>...</td>
          <td>-0.052722</td>
          <td>0.172050</td>
          <td>-0.196149</td>
          <td>0.121417</td>
          <td>-0.023714</td>
          <td>-0.452542</td>
          <td>0.017421</td>
          <td>-0.064719</td>
          <td>0.212975</td>
          <td>-0.078501</td>
        </tr>
        <tr>
          <th>zf30</th>
          <td>0.019257</td>
          <td>0.029394</td>
          <td>-0.055157</td>
          <td>-0.044533</td>
          <td>0.028718</td>
          <td>-0.049564</td>
          <td>-0.077346</td>
          <td>0.005204</td>
          <td>-0.016760</td>
          <td>0.031810</td>
          <td>...</td>
          <td>-0.036262</td>
          <td>-0.073130</td>
          <td>0.011830</td>
          <td>0.014447</td>
          <td>0.057727</td>
          <td>-0.020929</td>
          <td>0.018321</td>
          <td>0.014865</td>
          <td>-0.048886</td>
          <td>-0.037714</td>
        </tr>
      </tbody>
    </table>
    <p>19314 rows × 128 columns</p>
    </div>
