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

3. Load the input files
~~~~~~~~~~~~~~~~~~~~~~~

Load the CRISPR data and show the content.

.. code:: ipython3

    import pandas as pd
    import os
    df = pd.read_csv("CRISPRGeneEffect.csv").rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
    print(f'{df.isna().sum().sum()} NaN over {len(df)*len(df.columns)} values')
    df


.. parsed-literal::

    739493 NaN over 20287300 values




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



Example 1.1 two-class labeling of EGs based on tissue information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter the information to be exploited
''''''''''''''''''''''''''''''''''''''

Filter the genes mapped to tissues (``OncotreeLineage`` column in the
mapping file) having less than ``minlines`` cell-lines.

.. code:: ipython3

    from HELPpy.utility.selection import filter_crispr_by_model
    df = filter_crispr_by_model(df, df_map, minlines=10, line_group='OncotreeLineage')
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



Show which are the tissues available from the mapping file:

.. code:: ipython3

    print(df_map[['OncotreeLineage']].value_counts())


.. parsed-literal::

    OncotreeLineage          
    Lung                         249
    Lymphoid                     211
    CNS/Brain                    122
    Skin                         120
    Esophagus/Stomach             95
    Breast                        94
    Bowel                         89
    Head and Neck                 84
    Bone                          77
    Myeloid                       77
    Ovary/Fallopian Tube          75
    Kidney                        73
    Pancreas                      66
    Peripheral Nervous System     56
    Soft Tissue                   55
    Biliary Tract                 44
    Uterus                        41
    Fibroblast                    41
    Bladder/Urinary Tract         39
    Normal                        39
    Pleura                        35
    Liver                         29
    Cervix                        25
    Eye                           21
    Thyroid                       18
    Prostate                      15
    Testis                         7
    Vulva/Vagina                   5
    Muscle                         5
    Ampulla of Vater               4
    Hair                           2
    Other                          1
    Embryonal                      1
    Adrenal Gland                  1
    Name: count, dtype: int64


Select only cell-lines of a chosen tissue (here ``Kidney``) and remove
cell-lines having more than a certain percentage of NaN values (here
95%):

.. code:: ipython3

    tissue = 'Kidney'
    from HELPpy.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from HELPpy.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, [tissue])
    print(f"Selecting {len(cell_lines)} cell-lines")
    # remove rows with more than perc NaNs
    df_nonan = delrows_with_nan_percentage(df, perc=95)


.. parsed-literal::

    Selecting 37 cell-lines
    Removed 512 rows from 18443 with at least 95% NaN


Apply two-class HELP labelling
''''''''''''''''''''''''''''''

Compute the two-class labeling (``mode='flat-multi'``) using the Otsu
algorithm (``algorithm='otsu'``) and save the results in a csv file
(``Kidney_HELP_twoClasses.csv``):

.. code:: ipython3

    df_label2 = labelling(df_nonan, columns = cell_lines, n_classes=2,
                          labelnames={0: 'E', 1: 'NE'},
                          mode='flat-multi', algorithm='otsu', verbose=True)
    # save the result
    df_label2.to_csv(f"{tissue}_HELP_twoClasses.csv")


.. parsed-literal::

    performing flat mode on 2-class labelling (flat-multi).
    [flat-multi]: 1. multi-class labelling:


.. parsed-literal::

    100%|██████████| 37/37 [00:00<00:00, 598.29it/s]


.. parsed-literal::

    label
    NE       16678
    E         1253
    Name: count, dtype: int64


Example 1.2 three-class labeling of EGs based on tissue information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Genes have already been filtered according to tissue information for
Example 1.1, so we only need to: 

Apply three-class HELP labelling 
''''''''''''''''''''''''''''''''

Compute the three-class labeling (``mode='two-by-two'``) using
the Otsu algorithm (``algorithm='otsu'``) and save the results in a csv
file (``'Kidney_HELP_threeClasses.csv'``):

.. code:: ipython3

    df_label3 = labelling(df_nonan, columns = cell_lines, n_classes=2,
                          labelnames={0: 'E', 1: 'aE', 2: 'sNE'},
                          mode='two-by-two', algorithm='otsu', verbose=True)
    # save the result
    df_label3.to_csv(f"{tissue}_HELP_threeClasses.csv")


.. parsed-literal::

    performing flat mode on 3-class labelling (two-by-two).
    [two-by-two]: 1. Two-class labelling:


.. parsed-literal::

    100%|██████████| 37/37 [00:00<00:00, 572.91it/s]


.. parsed-literal::

    (17931,)
    [two-by-two]: 2. Two-class labelling on 1-label rows:


.. parsed-literal::

    100%|██████████| 37/37 [00:00<00:00, 737.77it/s]


.. parsed-literal::

    (16678,)
    label
    sNE      13457
    aE        3221
    E         1253
    Name: count, dtype: int64


Example 1.3 two-class labeling of EGs based on disease-related information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter the information to be exploited
''''''''''''''''''''''''''''''''''''''

Show which are the diseases available from the mapping file
(``OncotreePrimaryDisease`` column):

.. code:: ipython3

    print(df_map[['OncotreePrimaryDisease']].value_counts())


.. parsed-literal::

    OncotreePrimaryDisease                                       
    Non-Small Cell Lung Cancer                                       161
    Non-Cancerous                                                    131
    Mature B-Cell Neoplasms                                          113
    Melanoma                                                         107
    Diffuse Glioma                                                    94
                                                                    ... 
    Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma      1
    Myelodysplastic Syndromes                                          1
    Mixed Cervical Carcinoma                                           1
    Hereditary Spherocytosis                                           1
    Acute Leukemias of Ambiguous Lineage                               1
    Name: count, Length: 86, dtype: int64


Select only cell-lines mapped (via the ``OncotreePrimaryDisease`` column
of the mapping file) to a chosen disease (here
``Acute Myeloid Leukemia``\ ’) and remove cell-lines having more than a
certain percentage of NaN values (here 95%):

.. code:: ipython3

    disease = 'Acute Myeloid Leukemia'
    from HELPpy.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from HELPpy.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, [disease], line_group='OncotreePrimaryDisease')
    
    print(f"Selecting {len(cell_lines)} cell-lines")
    # remove rows with more than perc NaNs
    df_nonan = delrows_with_nan_percentage(df[cell_lines], perc=95)


.. parsed-literal::

    Selecting 24 cell-lines
    Removed 512 rows from 18443 with at least 95% NaN


Apply two-class HELP labelling
''''''''''''''''''''''''''''''

Compute the two-class labeling (mode=‘flat-multi’) using the Otsu
algorithm (algorithm=‘otsu’), save the results in a csv file (‘Acute
Myeloid Leukemia_HELP_twoClasses.csv’) and print their summary:

.. code:: ipython3

    df_label2 = labelling(df_nonan, columns = cell_lines, n_classes=2,
                          labelnames={0: 'E', 1: 'NE'},
                          mode='flat-multi', algorithm='otsu', verbose=True)
    # save the result
    df_label2.to_csv(f"{disease}_HELP_twoClasses.csv")
    # print the number of NaNs
    df_label2.value_counts(normalize=False)


.. parsed-literal::

    performing flat mode on 2-class labelling (flat-multi).
    [flat-multi]: 1. multi-class labelling:


.. parsed-literal::

    100%|██████████| 24/24 [00:00<00:00, 492.87it/s]


.. parsed-literal::

    label
    NE       16609
    E         1322
    Name: count, dtype: int64




.. parsed-literal::

    label
    NE       16609
    E         1322
    Name: count, dtype: int64


