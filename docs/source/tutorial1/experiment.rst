.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/giordamaug/HELP/blob/main/help/notebooks/experiment.ipynb
.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
   :target: https://www.kaggle.com/notebooks/welcome?src=https://github.com/giordamaug/HELP/blob/main/help/notebooks/experiment.ipynb

Install HELP from GitHub
========================

Skip this cell if you already have installed HELP.

.. code:: ipython3

    !pip install git+https://github.com/giordamaug/HELP.git

Download the input files
========================

In this cell we download from GitHub repository the label file and the
attribute files. Skip this step if you already have these input files
locally.

.. code:: ipython3

    tissue='Kidney'
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_HELP.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_BIO.csv
    for i in range(5):
      !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_CCcfs_{i}.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_EmbN2V_128.csv
    #!wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_CCBeder.csv
    #for i in range(15):
    #  !wget https://raw.githubusercontent.com/giordamaug/HELP/main/help/datafinal/{tissue}_BPBeder_{i}.csv

Run script for experiment
=========================

This is the batch script for EG prediction used for the experiments. the
manual page iof EG_prediction.py is the follwing:

.. code:: ipython3

    !python EG_prediction.py -h


.. parsed-literal::

    usage: EG_prediction.py [-h] -i <inputfile> [<inputfile> ...]
                            [-X <excludelabels> [<excludelabels> ...]]
                            [-L <labelname>] -l <labelfile> [-A <aliases>]
                            [-b <seed>] [-r <repeat>] [-f <folds>] [-j <jobs>]
                            [-B] [-sf <subfolds>] [-P] [-ba] [-fx]
                            [-n <normalize>] [-o <outfile>] [-s <scorefile>]
    
    PLOS COMPBIO
    
    options:
      -h, --help            show this help message and exit
      -i <inputfile> [<inputfile> ...], --inputfile <inputfile> [<inputfile> ...]
                            input attribute filename list
      -X <excludelabels> [<excludelabels> ...], --excludelabels <excludelabels> [<excludelabels> ...]
                            labels to exclude (default NaN, values any list)
      -L <labelname>, --labelname <labelname>
                            label name (default label)
      -l <labelfile>, --labelfile <labelfile>
                            label filename
      -A <aliases>, --aliases <aliases>
                            the dictionary for label renaming (es: {"oldlabel1":
                            "newlabel1", ..., "oldlabelN": "newlabelN"})
      -b <seed>, --seed <seed>
                            random seed (default: 1)
      -r <repeat>, --repeat <repeat>
                            n. of iteration (default: 10)
      -f <folds>, --folds <folds>
                            n. of cv folds (default: 5)
      -j <jobs>, --jobs <jobs>
                            n. of parallel jobs (default: -1)
      -B, --batch           enable batch mode (no output)
      -sf <subfolds>, --subfolds <subfolds>
                            n. of folds for subsampling (default: 0 - no
                            subsampling)
      -P, --proba           enable probability mode output (default disabled)
      -ba, --balanced       enable balancing in classifier (default disabled)
      -fx, --fixna          enable fixing NaN (default disabled)
      -n <normalize>, --normalize <normalize>
                            normalization mode (default None)
      -o <outfile>, --outfile <outfile>
                            output file for performance measures sumup
      -s <scorefile>, --scorefile <scorefile>
                            output file reporting all measurements


E vs NE experiments for kidney
==============================

This cell’s code reproduce results of Table 3 (A) in the reference
paper.

.. code:: ipython3

    datapath = "../datafinal"
    tissue = "Kidney"                               # or 'Lung'
    labelfile = f"{tissue}_HELP.csv"                # label filename
    aliases = "-A \"{'aE': 'NE', 'sNE':'NE'}\""     # dictionary for renaming labels before prediction: es. {'oldlabel': 'newlabel'}
    #aliases = ""
    #excludeflags = "-X aE"                         # label to remove: es. -X aE (for E vs sNE problem)
    excludeflags = ""                               
    njobs = "-1"                                    # parallelism level: -1 = all cpus, 1 = sequential
    sfolds = "4"                                    # dataset subsampling factor: es: 4 for 1:4 ratio of <minority-class>:<majority-class>
    nchunks = "-c 1 5 1"                            # no. of chunks for each input attribute file: es. 1 5 (Bio is one chunk, CCcfs split in 5 chunks)
    !python EG_prediction.py -i {datapath}/{tissue}_BIO.csv \
                                {datapath}/{tissue}_CCcfs.csv \
                                {datapath}/{tissue}_EmbN2V_128.csv \
                                {nchunks} \
                                -l {datapath}/{labelfile} \
                                {aliases} {excludeflags}  \
                                -n std -ba -sf {sfolds} \
                                -j {njobs} -P


.. parsed-literal::

    METHOD: LGBM	MODE: prob	BALANCE: yes
    PROBL: E vs NE
    INPUT: Kidney_BIO.csv Kidney_CCcfs.csv
    LABEL: Kidney_HELP.csv DISTRIB: E : 1242, NE: 4809
    SUBSAMPLE: 1:4
    +-------------+-------------------------------+
    |             | measure                       |
    |-------------+-------------------------------|
    | ROC-AUC     | 0.9500±0.0067                 |
    | Accuracy    | 0.9038±0.0077                 |
    | BA          | 0.8616±0.0123                 |
    | Sensitivity | 0.7900±0.0238                 |
    | Specificity | 0.9332±0.0078                 |
    | MCC         | 0.7110±0.0225                 |
    | CM          | [[9812, 2608], [3212, 44878]] |
    +-------------+-------------------------------+


E vs sNE experiments for kidney
===============================

This cell’s code reproduce results of Table 4(A) in the reference paper.

.. code:: ipython3

    datapath = "../datafinal"
    tissue = "Kidney"                               # or 'Lung'
    labelfile = f"{tissue}_HELP.csv"                # label filename
    aliases = ""                                    # dictionary for renaming labels before prediction: es. {'oldlabel': 'newlabel'}
    excludeflags = "-X aE"                          # label to remove: es. -X aE (for E vs sNE problem)
    njobs = "-1"                                    # parallelism level: -1 = all cpus, 1 = sequential
    sfolds = "4"                                    # dataset subsampling factor: es: 4 for 1:4 ratio of <minority-class>:<majority-class>
    nchunks = "-c 1 5 1"                            # no. of chunks for each input attribute file: es. 1 5 (Bio is one chunk, CCcfs split in 5 chunks)
    !python EG_prediction.py -i {datapath}/{tissue}_BIO.csv \
                                {datapath}/{tissue}_CCcfs.csv \
                                {datapath}/{tissue}_EmbN2V_128.csv \
                                {nchunks} \
                                -l {datapath}/{labelfile} \
                                {aliases} {excludeflags}  \
                                -n std -ba -sf {sfolds} \
                                -j {njobs} -P -B


.. parsed-literal::

    METHOD: LGBM	MODE: prob	BALANCE: yes
    PROBL: E vs sNE
    INPUT: Kidney_BIO.csv Kidney_CCcfs.csv Kidney_EmbN2V_128.csv
    LABEL: Kidney_HELP.csv DISTRIB: E : 1242, sNE: 4810
    SUBSAMPLE: 1:4
    +-------------+--------------------------------+
    |             | measure                        |
    |-------------+--------------------------------|
    | ROC-AUC     | 0.9701±0.0062                  |
    | Accuracy    | 0.9354±0.0070                  |
    | BA          | 0.9020±0.0120                  |
    | Sensitivity | 0.8454±0.0231                  |
    | Specificity | 0.9587±0.0065                  |
    | MCC         | 0.8026±0.0215                  |
    | CM          | [[10500, 1920], [1988, 46112]] |
    +-------------+--------------------------------+

