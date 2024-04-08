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
``Kidney_EmbN2V_128.csv``). Skip this step if you already have these
input files locally.

.. code:: ipython3

    tissue='Kidney'
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_HELP.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_BIO.csv
    for i in range(5):
      !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_CCcfs_{i}.csv
    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_EmbN2V_128.csv
    #!wget https://raw.githubusercontent.com/giordamaug/HELP/main/data/{tissue}_CCBeder.csv

Other attribute files (CCBeder) are shown but commented to help the user
experiment with different data.

3. Download the script for the experiments and show the man page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the batch script for EG prediction used for the experiments and
show its manual page:

.. code:: ipython3

    !wget https://raw.githubusercontent.com/giordamaug/HELP/main/HELPpy/notebooks/EG_prediction.py
    !python EG_prediction.py -h


.. parsed-literal::

    usage: EG_prediction.py [-h] -i <inputfile> [<inputfile> ...]
                            [-c <chunks> [<chunks> ...]]
                            [-X <excludelabels> [<excludelabels> ...]]
                            [-L <labelname>] -l <labelfile> [-A <aliases>]
                            [-b <seed>] [-r <repeat>] [-f <folds>] [-j <jobs>]
                            [-B] [-v <voters>] [-ba] [-fx] [-n <normalize>]
                            [-o <outfile>] [-s <scorefile>] [-p <predfile>]
    
    PLOS COMPBIO
    
    options:
      -h, --help            show this help message and exit
      -i <inputfile> [<inputfile> ...], --inputfile <inputfile> [<inputfile> ...]
                            input attribute filename list
      -c <chunks> [<chunks> ...], --chunks <chunks> [<chunks> ...]
                            no of chunks for attribute filename list
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
      -v <voters>, --voters <voters>
                            n. of voter predictors (default: 1 - one classifier)
      -ba, --balanced       enable balancing in classifier (default disabled)
      -fx, --fixna          enable fixing NaN (default disabled)
      -n <normalize>, --normalize <normalize>
                            normalization mode (default None)
      -o <outfile>, --outfile <outfile>
                            output file for performance measures sumup
      -s <scorefile>, --scorefile <scorefile>
                            output file reporting all measurements
      -p <predfile>, --predfile <predfile>
                            output file reporting predictions


4. Run the E vs NE experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This cell’s code reproduces the results for Kidney reported in Table 3
(A) of the HELP paper.

.. code:: ipython3

    datapath = "."
    tissue = "Kidney"                               # or 'Lung'
    labelfile = f"{tissue}_HELP.csv"                # label filename
    aliases = "-A \"{'aE': 'NE', 'sNE':'NE'}\""     # dictionary for renaming labels before prediction: es. {'oldlabel': 'newlabel'}
    excludeflags = ""                               # label to remove (none for E vs NE problem)
    njobs = "-1"                                    # parallelism level: -1 = all cpus, 1 = sequential
    nchunks = "-c 1 5 1"                            # no. of chunks for each input attribute file: es. 1 5 (Bio is one chunk, CCcfs split in 5 chunks)
    voters = "-v 10"                                # no. of voters on classifier ensemble
    repeats = "-r 10"                               # no. of iterations for experiments 
    !python EG_prediction.py -i {datapath}/{tissue}_BIO.csv \
                                {datapath}/{tissue}_CCcfs.csv \
                                {datapath}/{tissue}_EmbN2V_128.csv \
                                {nchunks} \
                                -l {datapath}/{labelfile} \
                                {aliases} {excludeflags}  \
                                {voters} {repeats} \
                                -n std -ba \
                                -j -1 -B


.. parsed-literal::

    METHOD: LGBM	VOTERS: 10	BALANCE: yes
    PROBL: E vs NE
    INPUT: Kidney_BIO.csv Kidney_CCcfs.csv Kidney_EmbN2V_128.csv
    LABEL: Kidney_HELP.csv DISTRIB: E : 1242, NE: 15994
    +-------------+----------------------------------+
    |             | measure                          |
    |-------------+----------------------------------|
    | ROC-AUC     | 0.9572±0.0057                    |
    | Accuracy    | 0.8939±0.0037                    |
    | BA          | 0.8904±0.0089                    |
    | Sensitivity | 0.8862±0.0190                    |
    | Specificity | 0.8945±0.0044                    |
    | MCC         | 0.5483±0.0114                    |
    | CM          | [[11007, 1413], [16876, 143064]] |
    +-------------+----------------------------------+


5. Run the E vs sNE experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This cell’s code reproduces the results for Kidney reported in Table 4
(A) of the HELP paper, removing the ``aE`` flags
(``excludeflags = "-X aE"``).

.. code:: ipython3

    datapath = "."
    tissue = "Kidney"                               # or 'Lung'
    labelfile = f"{tissue}_HELP.csv"                # label filename
    aliases = ""                                    # dictionary for renaming labels before prediction: es. {'oldlabel': 'newlabel'}
    excludeflags = "-X aE"                          # label to remove: es. -X aE (for E vs sNE problem)
    njobs = "-1"                                    # parallelism level: -1 = all cpus, 1 = sequential
    nchunks = "-c 1 5 1"                            # no. of chunks for each input attribute file: es. 1 5 (Bio is one chunk, CCcfs split in 5 chunks)
    voters = "-v 8"                                 # no. of voters on classifier ensemble
    repeats = "-r 10"                               # no. of iterations for experiments 
    !python EG_prediction.py -i {datapath}/{tissue}_BIO.csv \
                                {datapath}/{tissue}_CCcfs.csv \
                                {datapath}/{tissue}_EmbN2V_128.csv \
                                {nchunks} \
                                -l {datapath}/{labelfile} \
                                {aliases} {excludeflags}  \
                                -n std -ba \
                                {voters} {repeats} \
                                -j {njobs} -B


.. parsed-literal::

    METHOD: LGBM	VOTERS: 8	BALANCE: yes
    PROBL: E vs sNE
    INPUT: Kidney_BIO.csv Kidney_CCcfs.csv Kidney_EmbN2V_128.csv
    LABEL: Kidney_HELP.csv DISTRIB: E : 1242, sNE: 12886
    +-------------+---------------------------------+
    |             | measure                         |
    |-------------+---------------------------------|
    | ROC-AUC     | 0.9724±0.0039                   |
    | Accuracy    | 0.9221±0.0044                   |
    | BA          | 0.9129±0.0085                   |
    | Sensitivity | 0.9018±0.0186                   |
    | Specificity | 0.9241±0.0053                   |
    | MCC         | 0.6579±0.0135                   |
    | CM          | [[11200, 1220], [9779, 119081]] |
    +-------------+---------------------------------+


Please be aware that this will take a while in sequential execution.
