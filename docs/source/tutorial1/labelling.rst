Install HELP from GitHub
========================

Skip this cell if you alread have installed HELP.

.. code:: ipython3

    !pip install git+https://github.com/giordamaug/HELP.git

Download the input files
========================

In this cell we download from GitHub repository the label file and the
attribute files. Skip this step if you already have these input files
locally.

.. code:: ipython3

    !wget -c https://figshare.com/ndownloader/files/43346616 -O CRISPRGeneEffect.csv
    !wget -c https://figshare.com/ndownloader/files/43746708 -O Model.csv

Load the CRISPR data file
=========================

.. code:: ipython3

    import pandas as pd
    import os
    df = pd.read_csv("CRISPRGeneEffect.csv").rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
    print(f'{df.isna().sum().sum()} NaN over {len(df)*len(df.columns)} values')
    df


.. parsed-literal::

    739493 NaN over 20287300 values




.. raw:: html

    
      <div id="df-96b856ba-40cd-4f40-bc51-de98c98cb478" class="colab-df-container">
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
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-96b856ba-40cd-4f40-bc51-de98c98cb478')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-96b856ba-40cd-4f40-bc51-de98c98cb478 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-96b856ba-40cd-4f40-bc51-de98c98cb478');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-b52dc1c1-fcd9-4c6a-9d88-ee520a73c63f">
      <button class="colab-df-quickchart" onclick="quickchart('df-b52dc1c1-fcd9-4c6a-9d88-ee520a73c63f')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-b52dc1c1-fcd9-4c6a-9d88-ee520a73c63f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




Load the map between cell lines and tissues
===========================================

.. code:: ipython3

    df_map = pd.read_csv("Model.csv")
    print(df_map[['OncotreeLineage']].value_counts())
    df_map


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
    dtype: int64




.. raw:: html

    
      <div id="df-364386d7-22ea-4946-a637-2f766cb60b07" class="colab-df-container">
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
          <td>None</td>
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
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-364386d7-22ea-4946-a637-2f766cb60b07')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-364386d7-22ea-4946-a637-2f766cb60b07 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-364386d7-22ea-4946-a637-2f766cb60b07');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-7a24c151-30c4-496f-82e2-690109ff81f9">
      <button class="colab-df-quickchart" onclick="quickchart('df-7a24c151-30c4-496f-82e2-690109ff81f9')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-7a24c151-30c4-496f-82e2-690109ff81f9 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




.. code:: ipython3

    from help.utility.selection import filter_crispr_by_model
    df = filter_crispr_by_model(df, df_map, minlines=10, line_group='OncotreeLineage')
    df




.. raw:: html

    
      <div id="df-68fb011e-3c77-44db-96de-2aa825812a16" class="colab-df-container">
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
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-68fb011e-3c77-44db-96de-2aa825812a16')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-68fb011e-3c77-44db-96de-2aa825812a16 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-68fb011e-3c77-44db-96de-2aa825812a16');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-e5cfc771-e368-4284-a192-75b546f442e5">
      <button class="colab-df-quickchart" onclick="quickchart('df-e5cfc771-e368-4284-a192-75b546f442e5')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-e5cfc771-e368-4284-a192-75b546f442e5 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




Select some tissues
===================

In this section we select only cell-lines of a specific tissue. We check
that, once CRISPR datafile is reduced to a subset of total cell-lines,
that there is no row (gene) in the datafile with all NaN as cell values.
Inthat case we remove those rows (genes) before applying the labelling
algorithm.

We start labelling genes for the ``Kidney`` tissue…

.. code:: ipython3

    tissue = 'Kidney'
    from help.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from help.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, [tissue])
    print(f"Selecting {len(cell_lines)} cell-lines")
    # remove rows with all nans
    df_nonan = delrows_with_nan_percentage(df[cell_lines], perc=95)
    df_label1 = labelling(df_nonan, columns = cell_lines, n_classes=2,
                          labelnames={0: 'E', 1: 'aE', 2: 'sNE'},
                          mode='two-by-two', algorithm='otsu')
    df_label1.to_csv(f"{tissue}_HELP.csv")
    df_label1.value_counts(normalize=False), f"Nan: {df_label1['label'].isna().sum()}"


.. parsed-literal::

    Selecting 37 cell-lines
    Removed 512 rows from 18443 with at least 95% NaN


.. parsed-literal::

    100%|██████████| 37/37 [00:01<00:00, 29.77it/s]
    100%|██████████| 37/37 [00:00<00:00, 207.00it/s]




.. parsed-literal::

    (label
     sNE      13457
     aE        3221
     E         1253
     dtype: int64,
     'Nan: 0')



… then we do it for the ``Lung`` tissue …

.. code:: ipython3

    tissue = 'Lung'
    from help.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from help.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, [tissue])
    print(f"Selecting {len(cell_lines)} cell-lines")
    # remove rows with all nans
    df_nonan = delrows_with_nan_percentage(df[cell_lines], perc=95)
    df_label2 = labelling(df_nonan, columns = [cell_lines], n_classes=2,
                          labelnames={2: 'sNE', 1: 'aE', 0:'E'},
                          mode='two-by-two', algorithm='otsu')
    df_label2.to_csv(f"{tissue}_HELP.csv")
    df_label2.value_counts(), f"Nan: {df_label2['label'].isna().sum()}"


.. parsed-literal::

    Selecting 119 cell-lines
    Removed 512 rows from 18443 with at least 95% NaN


.. parsed-literal::

    100%|██████████| 119/119 [00:00<00:00, 124.62it/s]
    100%|██████████| 119/119 [00:01<00:00, 95.06it/s] 




.. parsed-literal::

    (label
     sNE      13847
     aE        2849
     E         1235
     dtype: int64,
     'Nan: 0')



Working on diseases …
=====================

In the same way we can make gene essentiality labelling based on disease
related information, by allowing the labelling algorith to focus on
CRISPR cell-lines related so spcific disease. In order to work on
disease-cells association, we use the same selection functions as before
but using a different Model column as selector
(``OncotreePrimaryDisease``).

.. code:: ipython3

    disease = 'Acute Myeloid Leukemia'
    from help.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from help.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, [disease], line_group='OncotreePrimaryDisease')  # change default from 'OncotreeLineage'
    print(f"Selecting {len(cell_lines)} cell-lines")
    # remove rows with all nans
    df_nonan = delrows_with_nan_percentage(df[cell_lines], perc=100)
    df_label = labelling(df_nonan, columns = cell_lines, n_classes=2, mode='flat-multi', algorithm='otsu')
    df_label.to_csv(f"{disease}_HELP.csv")
    df_label.value_counts(), f"Nan: {df_label['label'].isna().sum()}"


.. parsed-literal::

    Selecting 24 cell-lines
    Removed 512 rows from 18443 with at least 100% NaN


.. parsed-literal::

    100%|██████████| 24/24 [00:00<00:00, 210.68it/s]




.. parsed-literal::

    (label
     NE       16609
     E         1322
     dtype: int64,
     'Nan: 0')


