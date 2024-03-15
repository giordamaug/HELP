.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/giordamaug/HELP/blob/main/help/notebooks/csegs.ipynb
.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/notebooks/welcome?src=https://github.com/giordamaug/HELP/blob/main/help/notebooks/csegs.ipynb

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
    pd.set_option('display.max_columns', 10)
    import os
    df_orig = pd.read_csv("CRISPRGeneEffect.csv").rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
    print(f'{df_orig.isna().sum().sum()} NaN over {len(df_orig)*len(df_orig.columns)} values')
    df_orig


.. parsed-literal::

    739493 NaN over 20287300 values




.. raw:: html

    
      <div id="df-dfbcf913-fd17-4eb1-b07a-5a9f56f661c3" class="colab-df-container">
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
          <th>...</th>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        </tr>
        <tr>
          <th>ZYG11A</th>
          <td>-0.289724</td>
          <td>0.032983</td>
          <td>-0.201273</td>
          <td>-0.100344</td>
          <td>-0.112703</td>
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        <button class="colab-df-convert" onclick="convertToInteractive('df-dfbcf913-fd17-4eb1-b07a-5a9f56f661c3')"
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
            document.querySelector('#df-dfbcf913-fd17-4eb1-b07a-5a9f56f661c3 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-dfbcf913-fd17-4eb1-b07a-5a9f56f661c3');
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
    
    
    <div id="df-44d8ba97-0ecd-4940-a285-7352ca5e7c6a">
      <button class="colab-df-quickchart" onclick="quickchart('df-44d8ba97-0ecd-4940-a285-7352ca5e7c6a')"
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
            document.querySelector('#df-44d8ba97-0ecd-4940-a285-7352ca5e7c6a button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




Load the map between cell lines and tissues
===========================================

In this code we load the file that maps cell-lines identifiers (column
``ModelID``) to tissue, disease or tumor type (respectively:
``OncotreeLineage``, ``OncotreePrimaryDisease``, ``OncotreeSubtype``).
As a second step, we print the counts of lines for each tissue.

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

    
      <div id="df-0710dc5d-1dd9-4f78-b687-ed6f79c73b0b" class="colab-df-container">
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
          <th>...</th>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        </tr>
        <tr>
          <th>1916</th>
          <td>ACH-003157</td>
          <td>PT-QDEP9D</td>
          <td>ABM-T0822</td>
          <td>ABMT0822</td>
          <td>ZIMMMPLC</td>
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        <button class="colab-df-convert" onclick="convertToInteractive('df-0710dc5d-1dd9-4f78-b687-ed6f79c73b0b')"
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
            document.querySelector('#df-0710dc5d-1dd9-4f78-b687-ed6f79c73b0b button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-0710dc5d-1dd9-4f78-b687-ed6f79c73b0b');
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
    
    
    <div id="df-67c9961d-466a-411a-9c41-93b94804e957">
      <button class="colab-df-quickchart" onclick="quickchart('df-67c9961d-466a-411a-9c41-93b94804e957')"
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
            document.querySelector('#df-67c9961d-466a-411a-9c41-93b94804e957 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




Common Egs (pan-tissue labelling)
=================================

We can apply the labelling algorithm to all cell lines present in the
CRISPR data file to compute the common essential genes, i.e. the
*pan-tissue* (or pan-cancer, pan-disease, etc). This set of labels will
be used in the following to get the context-specific essential genes for
single tissue.

Ignoring in Model tissue with few cell-lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

in this code we select in the CRISPR data file only cell lines belonging
to tissue reported in the Model such that the number of lines is greater
(or equal) to a specific amount.

.. code:: ipython3

    from help.utility.selection import filter_crispr_by_model
    df = filter_crispr_by_model(df_orig, df_map, minlines=10, line_group='OncotreeLineage')
    df




.. raw:: html

    
      <div id="df-0d7b1529-24ec-435d-a474-d8436f289d5d" class="colab-df-container">
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
          <th>...</th>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        </tr>
        <tr>
          <th>ZYG11A</th>
          <td>-0.289724</td>
          <td>0.032983</td>
          <td>-0.201273</td>
          <td>-0.100344</td>
          <td>-0.112703</td>
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
          <td>...</td>
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
        <button class="colab-df-convert" onclick="convertToInteractive('df-0d7b1529-24ec-435d-a474-d8436f289d5d')"
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
            document.querySelector('#df-0d7b1529-24ec-435d-a474-d8436f289d5d button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-0d7b1529-24ec-435d-a474-d8436f289d5d');
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
    
    
    <div id="df-f66d0981-6b03-4c86-9ae4-a49c5ff69c3e">
      <button class="colab-df-quickchart" onclick="quickchart('df-f66d0981-6b03-4c86-9ae4-a49c5ff69c3e')"
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
            document.querySelector('#df-f66d0981-6b03-4c86-9ae4-a49c5ff69c3e button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
        </div>
      </div>




labelling EGs across tissues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we compute common EGs by applying the labelling
algorithm within each tissue-specufic cell lines. Then the common
essentiality label is computed by making the mode of previously-computed
labels across tissue. In order to do that, we with need to select
cell-lines form the ``Model.csv`` as a nested list of lists of
cell-lines. THis is obtained by properly calling the
``select-cell-Lines`` function.

.. code:: ipython3

    from help.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from help.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, tissue_list='all', nested=True)
    print(f"Selecting {len(cell_lines)} tissue for a total of {sum([len(x) for x in cell_lines])} cell-lines")


.. parsed-literal::

    Selecting 24 tissue for a total of 1091 cell-lines


Once selected the tissues as a nested list of cell-lines, we can apply
the labelling algorithm with the following coee. Note that we remove
from the CRIPSR matrix those genes with a percentage of missing values
greater than 80%.

.. code:: ipython3

    from help.utility.selection import delrows_with_nan_percentage
    from help.models.labelling import labelling
    # remove rows with all nans
    df_nonan = delrows_with_nan_percentage(df, perc=80)
    df_label = labelling(df_nonan, columns=cell_lines, n_classes=2, labelnames={1: 'NE', 0: 'E'},
                         mode='flat-multi', algorithm='otsu')
    df_label.to_csv("label_PanTissue_group.csv")
    df_label.value_counts(), f"Nan: {df_label['label'].isna().sum()}"


.. parsed-literal::

    Removed 512 rows from 18443 with at least 80% NaN


.. parsed-literal::

    100%|██████████| 34/34 [00:00<00:00, 260.19it/s]
    /usr/local/lib/python3.10/dist-packages/help/models/labelling.py:239: UserWarning: There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.
      warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
    100%|██████████| 32/32 [00:00<00:00, 424.17it/s]
    100%|██████████| 37/37 [00:00<00:00, 433.34it/s]
    100%|██████████| 59/59 [00:00<00:00, 413.31it/s]
    100%|██████████| 48/48 [00:00<00:00, 418.10it/s]
    100%|██████████| 86/86 [00:00<00:00, 432.36it/s]
    /usr/local/lib/python3.10/dist-packages/help/models/labelling.py:239: UserWarning: There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.
      warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
    100%|██████████| 18/18 [00:00<00:00, 417.12it/s]
    100%|██████████| 65/65 [00:00<00:00, 428.64it/s]
    /usr/local/lib/python3.10/dist-packages/help/models/labelling.py:239: UserWarning: There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.
      warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
    100%|██████████| 15/15 [00:00<00:00, 213.07it/s]
    100%|██████████| 72/72 [00:00<00:00, 270.59it/s]
    100%|██████████| 37/37 [00:00<00:00, 428.57it/s]
    /usr/local/lib/python3.10/dist-packages/help/models/labelling.py:239: UserWarning: There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.
      warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
    100%|██████████| 24/24 [00:00<00:00, 428.91it/s]
    100%|██████████| 119/119 [00:00<00:00, 426.87it/s]
    100%|██████████| 81/81 [00:00<00:00, 422.94it/s]
    100%|██████████| 37/37 [00:00<00:00, 434.84it/s]
    100%|██████████| 59/59 [00:00<00:00, 243.62it/s]
    100%|██████████| 47/47 [00:00<00:00, 409.77it/s]
    100%|██████████| 41/41 [00:00<00:00, 399.38it/s]
    100%|██████████| 19/19 [00:00<00:00, 395.83it/s]
    100%|██████████| 10/10 [00:00<00:00, 360.85it/s]
    100%|██████████| 71/71 [00:00<00:00, 428.74it/s]
    100%|██████████| 36/36 [00:00<00:00, 419.54it/s]
    /usr/local/lib/python3.10/dist-packages/help/models/labelling.py:239: UserWarning: There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.
      warnings.warn("There are rows with all NaNs, please remove them using the function 'rows_with_all_nan()' and re-apply the labelling. Otherwise you will have NaN labels in your output.")
    100%|██████████| 11/11 [00:00<00:00, 298.77it/s]
    100%|██████████| 33/33 [00:00<00:00, 433.54it/s]




.. parsed-literal::

    (label
     NE       16681
     E         1250
     dtype: int64,
     'Nan: 0')



For comparison, let’s compute the common EGs by applying the labelling
to all tissue in a flat way. In this case all cell-lines contribute in
the same way to the labelling criterium regardless of the related
tissue.

.. code:: ipython3

    from help.utility.selection import select_cell_lines, delrows_with_nan_percentage
    from help.models.labelling import labelling
    cell_lines = select_cell_lines(df, df_map, tissue_list='all', nested=False)
    print(f"Selecting {len([len(x) for x in cell_lines])} cell-lines")


.. parsed-literal::

    Selecting 1091 cell-lines


Once selected all cell-lines as a flat list of identifiers, we can apply
the labelling algorithm with the following code. Also in this case we
remove from the CRIPSR matrix those genes with a percentage of missing
values greater than 80%.

.. code:: ipython3

    from help.utility.selection import delrows_with_nan_percentage
    from help.models.labelling import labelling
    # remove rows with all nans
    df_nonan = delrows_with_nan_percentage(df, perc=80)
    df_label = labelling(df_nonan, columns=cell_lines, n_classes=2,
                         labelnames={1: 'NE', 0: 'E'},
                         mode='flat-multi', algorithm='otsu')
    df_label.to_csv("label_PanTissue.csv")
    df_label.value_counts(), f"Nan: {df_label['label'].isna().sum()}"


.. parsed-literal::

    Removed 512 rows from 18443 with at least 80% NaN


.. parsed-literal::

    100%|██████████| 1091/1091 [00:02<00:00, 372.72it/s]




.. parsed-literal::

    (label
     NE       16668
     E         1263
     dtype: int64,
     'Nan: 0')



Context specific EGs
====================

In this section we make the difference between pan-tissue EGs (common
EGs) and tissue EGs to get the context-specific EGs for the tissue
(csEGs). The EGs for Kidney and Lung tissues (previously generated) are
loaded from csv data files (downloaded from the GitHub repository). In
this forst code/plot we show the csEGs of Lung and Kidney by subtracting
the common EGs computed by grouping tissue cell-lines
(``label_PanTissue_group.csv``). In the second code/plot we show the
csEGs of the two tissues by subtracting the common EGs computed on all
cell-lines in flat mode (``label_PanTissue.csv``)

.. code:: ipython3

    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/Kidney_HELP.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/Lung_HELP.csv


.. parsed-literal::

    --2024-03-08 12:02:24--  https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/Kidney_HELP.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 180249 (176K) [text/plain]
    Saving to: ‘Kidney_HELP.csv.1’
    
    
Kidney_HELP.csv.1     0%[                    ]       0  --.-KB/s               
Kidney_HELP.csv.1   100%[===================>] 176.02K  --.-KB/s    in 0.03s   
    
    2024-03-08 12:02:24 (6.74 MB/s) - ‘Kidney_HELP.csv.1’ saved [180249/180249]
    
    --2024-03-08 12:02:24--  https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/Lung_HELP.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 180657 (176K) [text/plain]
    Saving to: ‘Lung_HELP.csv’
    
    Lung_HELP.csv       100%[===================>] 176.42K  --.-KB/s    in 0.02s   
    
    2024-03-08 12:02:25 (6.93 MB/s) - ‘Lung_HELP.csv’ saved [180657/180657]
    


.. code:: ipython3

    from help.utility.selection import EG_tissues_intersect
    csEGs, overlap, diffs = EG_tissues_intersect(tissues = {'Kidney': pd.read_csv("Kidney_HELP.csv", index_col=0),
                                                            'Lung': pd.read_csv("Lung_HELP.csv", index_col=0)
                                                           },
                            common_df = pd.read_csv("label_PanTissue_group.csv", index_col=0),
                            display=True, verbose=True)


.. parsed-literal::

    Subtracting 1250 common EGs...
    Overlapping of 5 genes between ['Kidney', 'Lung']
    55 genes only in Kidney
    23 genes only in Lung



.. image:: output_23_1.png


.. code:: ipython3

    from help.utility.selection import EG_tissues_intersect
    csEGs, overlap, diffs = EG_tissues_intersect(tissues = {'Kidney': pd.read_csv("Kidney_HELP.csv", index_col=0),
                                                            'Lung': pd.read_csv("Lung_HELP.csv", index_col=0)
                                                           },
                            #common_df = pd.read_csv(os.path.join(datapath, "PanTissue_HELP.csv"), index_col=0),
                            display=True, verbose=True)


.. parsed-literal::

    Overlapping of 1168 genes between ['Kidney', 'Lung']
    85 genes only in Kidney
    67 genes only in Lung



.. image:: output_24_1.png


As we ca see, the overlapping of cdEgs between the two tissues differs
in two genes. Then we save the csEGs into a file (the last one).

.. code:: ipython3

    with open("csEG_Kidney.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(csEGs['Kidney'])))
    with open("csEG_Lung.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(csEGs['Lung'])))
