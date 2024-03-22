1. Install HELP from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip this cell if you alread have installed HELP.

.. code:: ipython3

    !pip install git+https://github.com/giordamaug/HELP.git

2. Download the input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download from the DepMap portal the gene deletion expression scores
(``CRISPRGeneEffect.csv``) and the map between cell-lines and tissues
(``Model.csv``). Skip this step if you already have these input files
locally.

.. code:: ipython3

    !wget -c https://figshare.com/ndownloader/files/43346616 -O CRISPRGeneEffect.csv
    !wget -c https://figshare.com/ndownloader/files/43746708 -O Model.csv

3. Load the input file
~~~~~~~~~~~~~~~~~~~~~~

Load the CRISPR data and show the content.

.. code:: ipython3

    import pandas as pd
    import os
    df_orig = pd.read_csv("CRISPRGeneEffect.csv").rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
    print(f'{df_orig.isna().sum().sum()} NaNs over {len(df_orig)*len(df_orig.columns)} values')
    df_orig


.. parsed-literal::

    739493 NaNs over 20287300 values




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
          <th>gene</th>
          <th>ACH-000001</th>
          <th>ACH-000004</th>
          <th>ACH-000005</th>
          <th>ACH-000007</th>
          <th>ACH-000009</th>
          <th>ACH-000011</th>
          <th>ACH-000012</th>
          <th>ACH-000013</th>
          <th>ACH-000015</th>
          <th>ACH-000017</th>
          <th>...</th>
          <th>ACH-002693</th>
          <th>ACH-002710</th>
          <th>ACH-002785</th>
          <th>ACH-002799</th>
          <th>ACH-002800</th>
          <th>ACH-002834</th>
          <th>ACH-002847</th>
          <th>ACH-002922</th>
          <th>ACH-002925</th>
          <th>ACH-002926</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A1BG</th>
          <td>-0.122637</td>
          <td>0.019756</td>
          <td>-0.107208</td>
          <td>-0.031027</td>
          <td>0.008888</td>
          <td>0.022670</td>
          <td>-0.096631</td>
          <td>0.049811</td>
          <td>-0.099040</td>
          <td>-0.044896</td>
          <td>...</td>
          <td>-0.072582</td>
          <td>-0.033722</td>
          <td>-0.053881</td>
          <td>-0.060617</td>
          <td>0.025795</td>
          <td>-0.055721</td>
          <td>-0.009973</td>
          <td>-0.025991</td>
          <td>-0.127639</td>
          <td>-0.068666</td>
        </tr>
        <tr>
          <th>A1CF</th>
          <td>0.025881</td>
          <td>-0.083640</td>
          <td>-0.023211</td>
          <td>-0.137850</td>
          <td>-0.146566</td>
          <td>-0.057743</td>
          <td>-0.024440</td>
          <td>-0.158811</td>
          <td>-0.070409</td>
          <td>-0.115830</td>
          <td>...</td>
          <td>-0.237311</td>
          <td>-0.108704</td>
          <td>-0.114864</td>
          <td>-0.042591</td>
          <td>-0.132627</td>
          <td>-0.121228</td>
          <td>-0.119813</td>
          <td>-0.007706</td>
          <td>-0.040705</td>
          <td>-0.107530</td>
        </tr>
        <tr>
          <th>A2M</th>
          <td>0.034217</td>
          <td>-0.060118</td>
          <td>0.200204</td>
          <td>0.067704</td>
          <td>0.084471</td>
          <td>0.079679</td>
          <td>0.041922</td>
          <td>-0.003968</td>
          <td>-0.029389</td>
          <td>0.024537</td>
          <td>...</td>
          <td>-0.065940</td>
          <td>0.079277</td>
          <td>0.069333</td>
          <td>0.030989</td>
          <td>0.249826</td>
          <td>0.072790</td>
          <td>0.044097</td>
          <td>-0.038468</td>
          <td>0.134556</td>
          <td>0.067806</td>
        </tr>
        <tr>
          <th>A2ML1</th>
          <td>-0.128082</td>
          <td>-0.027417</td>
          <td>0.116039</td>
          <td>0.107988</td>
          <td>0.089419</td>
          <td>0.227512</td>
          <td>0.039121</td>
          <td>0.034778</td>
          <td>0.084594</td>
          <td>-0.003710</td>
          <td>...</td>
          <td>0.101541</td>
          <td>0.038977</td>
          <td>0.066599</td>
          <td>0.043809</td>
          <td>0.064657</td>
          <td>0.021916</td>
          <td>0.041358</td>
          <td>0.236576</td>
          <td>-0.047984</td>
          <td>0.112071</td>
        </tr>
        <tr>
          <th>A3GALT2</th>
          <td>-0.031285</td>
          <td>-0.036116</td>
          <td>-0.172227</td>
          <td>0.007992</td>
          <td>0.065109</td>
          <td>-0.130448</td>
          <td>0.028947</td>
          <td>-0.120875</td>
          <td>-0.052288</td>
          <td>-0.336776</td>
          <td>...</td>
          <td>0.005374</td>
          <td>-0.144070</td>
          <td>-0.256227</td>
          <td>-0.116473</td>
          <td>-0.294305</td>
          <td>-0.221940</td>
          <td>-0.146565</td>
          <td>-0.239690</td>
          <td>-0.116114</td>
          <td>-0.149897</td>
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
          <td>-0.289724</td>
          <td>0.032983</td>
          <td>-0.201273</td>
          <td>-0.100344</td>
          <td>-0.112703</td>
          <td>0.013401</td>
          <td>0.005124</td>
          <td>-0.089180</td>
          <td>-0.005409</td>
          <td>-0.070396</td>
          <td>...</td>
          <td>-0.296880</td>
          <td>-0.084936</td>
          <td>-0.128569</td>
          <td>-0.110504</td>
          <td>-0.087171</td>
          <td>0.024959</td>
          <td>-0.119911</td>
          <td>-0.079342</td>
          <td>-0.043555</td>
          <td>-0.045115</td>
        </tr>
        <tr>
          <th>ZYG11B</th>
          <td>-0.062972</td>
          <td>-0.410392</td>
          <td>-0.178877</td>
          <td>-0.462160</td>
          <td>-0.598698</td>
          <td>-0.296421</td>
          <td>-0.131949</td>
          <td>-0.145737</td>
          <td>-0.216393</td>
          <td>-0.257916</td>
          <td>...</td>
          <td>-0.332415</td>
          <td>-0.193408</td>
          <td>-0.327408</td>
          <td>-0.257879</td>
          <td>-0.349111</td>
          <td>0.015259</td>
          <td>-0.289412</td>
          <td>-0.347484</td>
          <td>-0.335270</td>
          <td>-0.307900</td>
        </tr>
        <tr>
          <th>ZYX</th>
          <td>0.074180</td>
          <td>0.113156</td>
          <td>-0.055349</td>
          <td>-0.001555</td>
          <td>0.095877</td>
          <td>0.067705</td>
          <td>-0.109147</td>
          <td>-0.034886</td>
          <td>-0.137350</td>
          <td>0.029457</td>
          <td>...</td>
          <td>-0.005090</td>
          <td>-0.218960</td>
          <td>-0.053033</td>
          <td>-0.041612</td>
          <td>-0.057478</td>
          <td>-0.306562</td>
          <td>-0.195097</td>
          <td>-0.085302</td>
          <td>-0.208063</td>
          <td>0.070671</td>
        </tr>
        <tr>
          <th>ZZEF1</th>
          <td>0.111244</td>
          <td>0.234388</td>
          <td>-0.002161</td>
          <td>-0.325964</td>
          <td>-0.026742</td>
          <td>-0.232453</td>
          <td>-0.164482</td>
          <td>-0.175850</td>
          <td>-0.168087</td>
          <td>-0.284838</td>
          <td>...</td>
          <td>-0.188751</td>
          <td>-0.120449</td>
          <td>-0.267081</td>
          <td>0.006148</td>
          <td>-0.189602</td>
          <td>-0.148368</td>
          <td>-0.206400</td>
          <td>-0.095965</td>
          <td>-0.094741</td>
          <td>-0.187813</td>
        </tr>
        <tr>
          <th>ZZZ3</th>
          <td>-0.467908</td>
          <td>-0.088306</td>
          <td>-0.186842</td>
          <td>-0.486660</td>
          <td>-0.320759</td>
          <td>-0.347234</td>
          <td>-0.277397</td>
          <td>-0.519586</td>
          <td>-0.282338</td>
          <td>-0.247634</td>
          <td>...</td>
          <td>-0.239991</td>
          <td>-0.311396</td>
          <td>-0.202158</td>
          <td>-0.195154</td>
          <td>-0.107107</td>
          <td>-0.579576</td>
          <td>-0.486525</td>
          <td>-0.346272</td>
          <td>-0.222404</td>
          <td>-0.452143</td>
        </tr>
      </tbody>
    </table>
    <p>18443 rows × 1100 columns</p>
    </div>



Then load the mapping information and show the content.

.. code:: ipython3

    df_map = pd.read_csv("Model.csv")
    df_map




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
          <th>ModelID</th>
          <th>PatientID</th>
          <th>CellLineName</th>
          <th>StrippedCellLineName</th>
          <th>DepmapModelType</th>
          <th>OncotreeLineage</th>
          <th>OncotreePrimaryDisease</th>
          <th>OncotreeSubtype</th>
          <th>OncotreeCode</th>
          <th>LegacyMolecularSubtype</th>
          <th>...</th>
          <th>TissueOrigin</th>
          <th>CCLEName</th>
          <th>CatalogNumber</th>
          <th>PlateCoating</th>
          <th>ModelDerivationMaterial</th>
          <th>PublicComments</th>
          <th>WTSIMasterCellID</th>
          <th>SangerModelID</th>
          <th>COSMICID</th>
          <th>LegacySubSubtype</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ACH-000001</td>
          <td>PT-gj46wT</td>
          <td>NIH:OVCAR-3</td>
          <td>NIHOVCAR3</td>
          <td>HGSOC</td>
          <td>Ovary/Fallopian Tube</td>
          <td>Ovarian Epithelial Tumor</td>
          <td>High-Grade Serous Ovarian Cancer</td>
          <td>HGSOC</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NIHOVCAR3_OVARY</td>
          <td>HTB-71</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2201.0</td>
          <td>SIDM00105</td>
          <td>905933.0</td>
          <td>high_grade_serous</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ACH-000002</td>
          <td>PT-5qa3uk</td>
          <td>HL-60</td>
          <td>HL60</td>
          <td>AML</td>
          <td>Myeloid</td>
          <td>Acute Myeloid Leukemia</td>
          <td>Acute Myeloid Leukemia</td>
          <td>AML</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>HL60_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
          <td>CCL-240</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>55.0</td>
          <td>SIDM00829</td>
          <td>905938.0</td>
          <td>M3</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ACH-000003</td>
          <td>PT-puKIyc</td>
          <td>CACO2</td>
          <td>CACO2</td>
          <td>COAD</td>
          <td>Bowel</td>
          <td>Colorectal Adenocarcinoma</td>
          <td>Colon Adenocarcinoma</td>
          <td>COAD</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>CACO2_LARGE_INTESTINE</td>
          <td>HTB-37</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SIDM00891</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ACH-000004</td>
          <td>PT-q4K2cp</td>
          <td>HEL</td>
          <td>HEL</td>
          <td>AML</td>
          <td>Myeloid</td>
          <td>Acute Myeloid Leukemia</td>
          <td>Acute Myeloid Leukemia</td>
          <td>AML</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>HEL_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
          <td>ACC 11</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>783.0</td>
          <td>SIDM00594</td>
          <td>907053.0</td>
          <td>M6</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ACH-000005</td>
          <td>PT-q4K2cp</td>
          <td>HEL 92.1.7</td>
          <td>HEL9217</td>
          <td>AML</td>
          <td>Myeloid</td>
          <td>Acute Myeloid Leukemia</td>
          <td>Acute Myeloid Leukemia</td>
          <td>AML</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>HEL9217_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
          <td>HEL9217</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SIDM00593</td>
          <td>NaN</td>
          <td>M6</td>
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
          <th>1916</th>
          <td>ACH-003157</td>
          <td>PT-QDEP9D</td>
          <td>ABM-T0822</td>
          <td>ABMT0822</td>
          <td>ZIMMMPLC</td>
          <td>Lung</td>
          <td>Non-Cancerous</td>
          <td>Immortalized MPLC Cells</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1917</th>
          <td>ACH-003158</td>
          <td>PT-nszsxG</td>
          <td>ABM-T9220</td>
          <td>ABMT9220</td>
          <td>ZIMMSMCI</td>
          <td>Muscle</td>
          <td>Non-Cancerous</td>
          <td>Immortalized Smooth Muscle Cells, Intestinal</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1918</th>
          <td>ACH-003159</td>
          <td>PT-AUxVvV</td>
          <td>ABM-T9233</td>
          <td>ABMT9233</td>
          <td>ZIMMRSCH</td>
          <td>Hair</td>
          <td>Non-Cancerous</td>
          <td>Immortalized Hair Follicle Inner Root Sheath C...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1919</th>
          <td>ACH-003160</td>
          <td>PT-AUxVvV</td>
          <td>ABM-T9249</td>
          <td>ABMT9249</td>
          <td>ZIMMGMCH</td>
          <td>Hair</td>
          <td>Non-Cancerous</td>
          <td>Immortalized Hair Germinal Matrix Cells</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1920</th>
          <td>ACH-003161</td>
          <td>PT-or1hkT</td>
          <td>ABM-T9430</td>
          <td>ABMT9430</td>
          <td>ZIMMPSC</td>
          <td>Pancreas</td>
          <td>Non-Cancerous</td>
          <td>Immortalized Pancreatic Stromal Cells</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>1921 rows × 36 columns</p>
    </div>



4. Filter the information to be exploited
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter the genes mapped to tissues (``OncotreeLineage`` column in the
mapping file) having less than ``minlines`` cell-lines:

.. code:: ipython3

    from help.utility.selection import filter_crispr_by_model
    df = filter_crispr_by_model(df_orig, df_map, minlines=10, line_group='OncotreeLineage')
    df




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
          <th>gene</th>
          <th>ACH-000001</th>
          <th>ACH-000004</th>
          <th>ACH-000005</th>
          <th>ACH-000007</th>
          <th>ACH-000009</th>
          <th>ACH-000011</th>
          <th>ACH-000012</th>
          <th>ACH-000013</th>
          <th>ACH-000015</th>
          <th>ACH-000017</th>
          <th>...</th>
          <th>ACH-002693</th>
          <th>ACH-002710</th>
          <th>ACH-002785</th>
          <th>ACH-002799</th>
          <th>ACH-002800</th>
          <th>ACH-002834</th>
          <th>ACH-002847</th>
          <th>ACH-002922</th>
          <th>ACH-002925</th>
          <th>ACH-002926</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>A1BG</th>
          <td>-0.122637</td>
          <td>0.019756</td>
          <td>-0.107208</td>
          <td>-0.031027</td>
          <td>0.008888</td>
          <td>0.022670</td>
          <td>-0.096631</td>
          <td>0.049811</td>
          <td>-0.099040</td>
          <td>-0.044896</td>
          <td>...</td>
          <td>-0.072582</td>
          <td>-0.033722</td>
          <td>-0.053881</td>
          <td>-0.060617</td>
          <td>0.025795</td>
          <td>-0.055721</td>
          <td>-0.009973</td>
          <td>-0.025991</td>
          <td>-0.127639</td>
          <td>-0.068666</td>
        </tr>
        <tr>
          <th>A1CF</th>
          <td>0.025881</td>
          <td>-0.083640</td>
          <td>-0.023211</td>
          <td>-0.137850</td>
          <td>-0.146566</td>
          <td>-0.057743</td>
          <td>-0.024440</td>
          <td>-0.158811</td>
          <td>-0.070409</td>
          <td>-0.115830</td>
          <td>...</td>
          <td>-0.237311</td>
          <td>-0.108704</td>
          <td>-0.114864</td>
          <td>-0.042591</td>
          <td>-0.132627</td>
          <td>-0.121228</td>
          <td>-0.119813</td>
          <td>-0.007706</td>
          <td>-0.040705</td>
          <td>-0.107530</td>
        </tr>
        <tr>
          <th>A2M</th>
          <td>0.034217</td>
          <td>-0.060118</td>
          <td>0.200204</td>
          <td>0.067704</td>
          <td>0.084471</td>
          <td>0.079679</td>
          <td>0.041922</td>
          <td>-0.003968</td>
          <td>-0.029389</td>
          <td>0.024537</td>
          <td>...</td>
          <td>-0.065940</td>
          <td>0.079277</td>
          <td>0.069333</td>
          <td>0.030989</td>
          <td>0.249826</td>
          <td>0.072790</td>
          <td>0.044097</td>
          <td>-0.038468</td>
          <td>0.134556</td>
          <td>0.067806</td>
        </tr>
        <tr>
          <th>A2ML1</th>
          <td>-0.128082</td>
          <td>-0.027417</td>
          <td>0.116039</td>
          <td>0.107988</td>
          <td>0.089419</td>
          <td>0.227512</td>
          <td>0.039121</td>
          <td>0.034778</td>
          <td>0.084594</td>
          <td>-0.003710</td>
          <td>...</td>
          <td>0.101541</td>
          <td>0.038977</td>
          <td>0.066599</td>
          <td>0.043809</td>
          <td>0.064657</td>
          <td>0.021916</td>
          <td>0.041358</td>
          <td>0.236576</td>
          <td>-0.047984</td>
          <td>0.112071</td>
        </tr>
        <tr>
          <th>A3GALT2</th>
          <td>-0.031285</td>
          <td>-0.036116</td>
          <td>-0.172227</td>
          <td>0.007992</td>
          <td>0.065109</td>
          <td>-0.130448</td>
          <td>0.028947</td>
          <td>-0.120875</td>
          <td>-0.052288</td>
          <td>-0.336776</td>
          <td>...</td>
          <td>0.005374</td>
          <td>-0.144070</td>
          <td>-0.256227</td>
          <td>-0.116473</td>
          <td>-0.294305</td>
          <td>-0.221940</td>
          <td>-0.146565</td>
          <td>-0.239690</td>
          <td>-0.116114</td>
          <td>-0.149897</td>
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
          <td>-0.289724</td>
          <td>0.032983</td>
          <td>-0.201273</td>
          <td>-0.100344</td>
          <td>-0.112703</td>
          <td>0.013401</td>
          <td>0.005124</td>
          <td>-0.089180</td>
          <td>-0.005409</td>
          <td>-0.070396</td>
          <td>...</td>
          <td>-0.296880</td>
          <td>-0.084936</td>
          <td>-0.128569</td>
          <td>-0.110504</td>
          <td>-0.087171</td>
          <td>0.024959</td>
          <td>-0.119911</td>
          <td>-0.079342</td>
          <td>-0.043555</td>
          <td>-0.045115</td>
        </tr>
        <tr>
          <th>ZYG11B</th>
          <td>-0.062972</td>
          <td>-0.410392</td>
          <td>-0.178877</td>
          <td>-0.462160</td>
          <td>-0.598698</td>
          <td>-0.296421</td>
          <td>-0.131949</td>
          <td>-0.145737</td>
          <td>-0.216393</td>
          <td>-0.257916</td>
          <td>...</td>
          <td>-0.332415</td>
          <td>-0.193408</td>
          <td>-0.327408</td>
          <td>-0.257879</td>
          <td>-0.349111</td>
          <td>0.015259</td>
          <td>-0.289412</td>
          <td>-0.347484</td>
          <td>-0.335270</td>
          <td>-0.307900</td>
        </tr>
        <tr>
          <th>ZYX</th>
          <td>0.074180</td>
          <td>0.113156</td>
          <td>-0.055349</td>
          <td>-0.001555</td>
          <td>0.095877</td>
          <td>0.067705</td>
          <td>-0.109147</td>
          <td>-0.034886</td>
          <td>-0.137350</td>
          <td>0.029457</td>
          <td>...</td>
          <td>-0.005090</td>
          <td>-0.218960</td>
          <td>-0.053033</td>
          <td>-0.041612</td>
          <td>-0.057478</td>
          <td>-0.306562</td>
          <td>-0.195097</td>
          <td>-0.085302</td>
          <td>-0.208063</td>
          <td>0.070671</td>
        </tr>
        <tr>
          <th>ZZEF1</th>
          <td>0.111244</td>
          <td>0.234388</td>
          <td>-0.002161</td>
          <td>-0.325964</td>
          <td>-0.026742</td>
          <td>-0.232453</td>
          <td>-0.164482</td>
          <td>-0.175850</td>
          <td>-0.168087</td>
          <td>-0.284838</td>
          <td>...</td>
          <td>-0.188751</td>
          <td>-0.120449</td>
          <td>-0.267081</td>
          <td>0.006148</td>
          <td>-0.189602</td>
          <td>-0.148368</td>
          <td>-0.206400</td>
          <td>-0.095965</td>
          <td>-0.094741</td>
          <td>-0.187813</td>
        </tr>
        <tr>
          <th>ZZZ3</th>
          <td>-0.467908</td>
          <td>-0.088306</td>
          <td>-0.186842</td>
          <td>-0.486660</td>
          <td>-0.320759</td>
          <td>-0.347234</td>
          <td>-0.277397</td>
          <td>-0.519586</td>
          <td>-0.282338</td>
          <td>-0.247634</td>
          <td>...</td>
          <td>-0.239991</td>
          <td>-0.311396</td>
          <td>-0.202158</td>
          <td>-0.195154</td>
          <td>-0.107107</td>
          <td>-0.579576</td>
          <td>-0.486525</td>
          <td>-0.346272</td>
          <td>-0.222404</td>
          <td>-0.452143</td>
        </tr>
      </tbody>
    </table>
    <p>18443 rows × 1091 columns</p>
    </div>



and remove also those having more than a certain percentage of NaN
values (here 80%):

.. code:: ipython3

    from help.utility.selection import delrows_with_nan_percentage
    # remove rows with more than perc NaNs
    df_nonan = delrows_with_nan_percentage(df, perc=80)


.. parsed-literal::

    Removed 512 rows from 18443 with at least 80% NaN


5. Compute EGs common to all tissues (pan-tissue labeling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, pan-tissue EGs are obtained by 1. identifying EGs in all
tissue-specific cell-lines and 2. computing the label of each gene as
the mode of the obtained labels.

In order to do that, we need to select from the mapping file all
cell-lines (``tissue_list='all'``) as a nested list of cell-lines (lists
of lists for each tissue, obtained with ``'nested=True'``):

labelling EGs across tissues
''''''''''''''''''''''''''''

In this example we compute common EGs by applying the labelling
algorithm within each tissue-specufic cell lines. Then the common
essentiality label is computed by making the mode of previously-computed
labels across tissue. In order to do that, we with need to select
cell-lines form the ``Model.csv`` as a nested list of lists of
cell-lines. THis is obtained by properly calling the
``select-cell-Lines`` function.

.. code:: ipython3

    from help.utility.selection import select_cell_lines
    cell_lines = select_cell_lines(df_nonan, df_map, tissue_list='all', nested=True)
    print(f"Selecting {len(cell_lines)} tissues for a total of {sum([len(x) for x in cell_lines])} cell-lines")


.. parsed-literal::

    Selecting 24 tissues for a total of 1091 cell-lines


Then, we compute the two-class labeling (``mode='flat-multi'``) using
the Otsu algorithm (``algorithm='otsu'``), returning the mode of the
labels (due to the input nested list of cell-lines), save the results in
a csv file (``'PanTissue_group_HELP.csv'``) and print their summary:

.. code:: ipython3

    from help.models.labelling import labelling
    # remove rows with all nans
    df_common = labelling(df_nonan, columns=cell_lines, n_classes=2, labelnames={0:'E', 1: 'NE'}, mode='flat-multi', algorithm='otsu')
    df_common.to_csv("PanTissue_group_HELP.csv")
    df_common.value_counts()


.. parsed-literal::

      0%|          | 0/34 [00:00<?, ?it/s]

.. parsed-literal::

    100%|██████████| 34/34 [00:00<00:00, 609.85it/s]
    100%|██████████| 32/32 [00:00<00:00, 694.73it/s]
    100%|██████████| 37/37 [00:00<00:00, 737.01it/s]
    100%|██████████| 59/59 [00:00<00:00, 739.17it/s]
    100%|██████████| 48/48 [00:00<00:00, 708.80it/s]
    100%|██████████| 86/86 [00:00<00:00, 721.52it/s]
    100%|██████████| 18/18 [00:00<00:00, 707.24it/s]
    100%|██████████| 65/65 [00:00<00:00, 719.63it/s]
    100%|██████████| 15/15 [00:00<00:00, 726.04it/s]
    100%|██████████| 72/72 [00:00<00:00, 748.21it/s]
    100%|██████████| 37/37 [00:00<00:00, 685.16it/s]
    100%|██████████| 24/24 [00:00<00:00, 722.50it/s]
    100%|██████████| 119/119 [00:00<00:00, 742.11it/s]
    100%|██████████| 81/81 [00:00<00:00, 721.42it/s]
    100%|██████████| 37/37 [00:00<00:00, 717.46it/s]
    100%|██████████| 59/59 [00:00<00:00, 748.28it/s]
    100%|██████████| 47/47 [00:00<00:00, 721.70it/s]
    100%|██████████| 41/41 [00:00<00:00, 718.57it/s]
    100%|██████████| 19/19 [00:00<00:00, 710.78it/s]
    100%|██████████| 10/10 [00:00<00:00, 702.00it/s]
    100%|██████████| 71/71 [00:00<00:00, 748.16it/s]
    100%|██████████| 36/36 [00:00<00:00, 721.75it/s]
    100%|██████████| 11/11 [00:00<00:00, 680.57it/s]
    100%|██████████| 33/33 [00:00<00:00, 702.29it/s]




.. parsed-literal::

    label
    NE       16681
    E         1250
    Name: count, dtype: int64



An alternative way for computing pan-tissue EGs could be to select all
cell-lines as a flat list of identifiers (``'nested=False'``), so
disregarding their mapping to tissues, and compute the EG labeling:

.. code:: ipython3

    from help.utility.selection import select_cell_lines
    cell_lines_un = select_cell_lines(df_nonan, df_map, tissue_list='all', nested=False)
    print(f"Selecting {len(cell_lines)} tissues for a total of {sum([len(x) for x in cell_lines_un])} cell-lines")
    df_common_flat = labelling(df_nonan, columns=cell_lines_un, n_classes=2, labelnames={0:'E', 1: 'NE'}, mode='flat-multi', algorithm='otsu')
    df_common_flat.to_csv("PanTissue.csv")
    df_common_flat.value_counts()


.. parsed-literal::

    Selecting 24 tissues for a total of 10910 cell-lines


.. parsed-literal::

    100%|██████████| 1091/1091 [00:01<00:00, 683.34it/s]




.. parsed-literal::

    label
    NE       16668
    E         1263
    Name: count, dtype: int64



In this case, the cell-lines contribute in the same way to the labelling
criterion regardless of the related tissue, thus providing a different,
less stringent labelling.

6. Subtract pan-tissue EGs from those of the chosen tissue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Context-specific EGs (csEGs) for a chosen tissue (here
``tissueK = 'Kidney'``) are obtained by subtracting the pan-tissue EGs
computed in the previous step (``df_common``) by the EGs identified for
the chosen tissue.

.. code:: ipython3

    import pandas as pd
    
    #Identification of EGs in Kidney tissue (as in Example 1)
    tissueK = 'Kidney'
    from help.utility.selection import select_cell_lines
    from help.models.labelling import labelling
    cell_linesK = select_cell_lines(df_nonan, df_map, [tissueK])
    print(f"Selecting {len(cell_linesK)} cell-lines")
    df_labelK = labelling(df_nonan, columns = cell_linesK, n_classes=2,
                          labelnames={0: 'E', 1: 'NE'},
                          mode='flat-multi', algorithm='otsu')
    df_labelK.to_csv(f"{tissueK}_HELP_twoClasses.csv")
    #Alternatively, you can download the Kidney labels already computed:
    #!wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/Kidney_HELP.csv
    
    #Identification of Kidney context-specific EGs
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    EG_kidney = df_labelK[df_labelK['label'] == 'E'].index.values
    cEG = df_common[df_common['label']=='E'].index.values
    cs_EG_kidney = np.setdiff1d(EG_kidney, cEG)
    print(cs_EG_kidney)
    with open("csEG_Kidney.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(cs_EG_kidney)))


.. parsed-literal::

    Selecting 37 cell-lines


.. parsed-literal::

    100%|██████████| 37/37 [00:00<00:00, 586.83it/s]


.. parsed-literal::

    ['ACTG1' 'ACTR6' 'ARF4' 'ARFRP1' 'ARPC4' 'CDK6' 'CFLAR' 'CHMP7' 'COPS3'
     'DCTN3' 'DDX11' 'DDX52' 'EMC3' 'EXOSC1' 'FERMT2' 'GEMIN7' 'GET3' 'HGS'
     'HNF1B' 'HTATSF1' 'ITGAV' 'KIF4A' 'MCM10' 'MDM2' 'METAP2' 'MLST8'
     'NCAPH2' 'NDOR1' 'NHLRC2' 'OXA1L' 'PAX8' 'PFN1' 'PIK3C3' 'PPIE' 'PPP1CA'
     'PPP4R2' 'PTK2' 'RAB7A' 'RAD1' 'RBM42' 'RBMX2' 'RTEL1' 'SEPHS2' 'SNAP23'
     'SNRPB2' 'SPTLC1' 'SRSF10' 'TAF1D' 'TMED10' 'TMED2' 'TRIM37' 'UBA5' 'UBC'
     'UBE2D3' 'USP10' 'VPS33A' 'VPS52' 'WDR25' 'YWHAZ' 'ZNG1B']


Visualizing the obtained results
''''''''''''''''''''''''''''''''

Show the supervenn plot of pan-tissue EGs, Kidney EGs and Kidney csEGs.

.. code:: ipython3

    from help.visualization.plot import svenn_intesect
    svenn_intesect([set(cs_EG_kidney),set(EG_kidney), set(cEG)], labels=['kidney csEGs', 'kidney EGs', 'common EGs'], ylabel='EGs', figsize=(8,4))



.. image:: output_24_0.png


The plot shows that the Kidney tissue shares 1193 EGs with all the other
tissues (over a total of 1250 cEGs) and has 60 csEGs.

Show the supervenn plot of Kidney csEGs against Lung csEGs.

.. code:: ipython3

    from help.visualization.plot import svenn_intesect
    from help.utility.selection import select_cell_lines
    from help.models.labelling import labelling
    tissueL = 'Lung'
    #a) Identify Lung EGs (as in Example 1)
    cell_linesL = select_cell_lines(df_nonan, df_map, [tissueL])
    print(f"Selecting {len(cell_linesL)} cell-lines")
    df_labelL = labelling(df_nonan, columns = cell_linesL, n_classes=2,
                          labelnames={0: 'E', 1: 'NE'},
                          mode='flat-multi', algorithm='otsu')
    
    #b) Compute Lung csEGs
    np.set_printoptions(threshold=sys.maxsize)
    EG_lung = df_labelL[df_labelL['label'] == 'E'].index.values
    cs_EG_lung = np.setdiff1d(EG_lung, cEG)
    print(cs_EG_lung)
    #with open("csEG_Lung.txt", 'w', encoding='utf-8') as f:
    #    f.write('\n'.join(list(cs_EG_lung)))
    
    #Show the supervenn plot
    svenn_intesect([set(cs_EG_kidney), set(cs_EG_lung)], labels=['kidney', 'lung'], ylabel='csEGs', figsize=(8,4))


.. parsed-literal::

    Selecting 119 cell-lines


.. parsed-literal::

    100%|██████████| 119/119 [00:00<00:00, 680.54it/s]


.. parsed-literal::

    ['ACO2' 'AP2M1' 'ATP5F1D' 'BORA' 'CCDC86' 'CDK2' 'CKS1B' 'DCTN3' 'DDX11'
     'DDX39B' 'DGCR8' 'GEMIN7' 'NCAPH2' 'NFYB' 'NUMA1' 'NUP153' 'OXA1L'
     'PI4KA' 'PPAT' 'PTCD3' 'SCD' 'SLBP' 'SLC25A3' 'TFRC' 'TRPM7' 'YPEL5'
     'YTHDC1' 'ZNF407']



.. image:: output_26_3.png

