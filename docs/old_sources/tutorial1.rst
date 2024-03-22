
EG detection for a tissue 
=========================

In notebook we show how to make essentiality labelling of genes by considering cell-line scores for a specific tissue.
In more details, the workflow implements the following steps 

#. load CRISPR data file and the Model file that maps cell-lines to tissues:
#. select from the CRISPR data file only cell-lines related to the chosen tissue, by using the mapping in the Model
#. in the redced CRIPSR data matrix, remove all genes having a percentage of missing values grater than 80%
#. apply the labelling process to get Essential/Non-Essential gene labels for the tissue

.. toctree::
   :maxdepth: 1

   tutorial1/labelling

Tissue's PPI network embedding 
==============================

In this notebook we show how to calculate embedding vectors for gene (nodes) in the PPI network 
of a tissue. The embedding matrix (gene x embedding vector components) can be used in successive processing,
such as prediction of gene essentiality base on gene attributes.

The notebook has few cells:

#. load the PPI network cvs file and apply mebdding by the [!Node2Vec](https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/neighbourhood/node2vec.html) embedding technique.
#. display the embedding matrix and save it in a csv file.

.. toctree::
   :maxdepth: 1

   tutorial1/embedding


Context-Specific EG for tissues 
===============================

In notebook we show how

#. starting from the CRISPR file we apply the HELP labelling algorithm to detect essential genes (EGs) and non-essentials
genes (NEGs) in two tissues and in all tissues (pan-tissue).
#. by subtracting pan-tissue EGs from a tissue EGs list we get the context-specific EGs (csEGs)
#. by using tissue specific attributes (genomic, functional, PPI structural, etc.) we develop a LighGBM prediction model for EGs. 

.. toctree::
   :maxdepth: 1

   tutorial1/csegs


EG prediction for a tissue 
==========================

In notebook we show how

#. starting from the CRISPR file we apply the HELP labelling algorithm to detect essential genes (EGs) and non-essentials
genes (NEGs) in two tissues and in all tissues (pan-tissue).
#. by subtracting pan-tissue EGs from a tissue EGs list we get the context-specific EGs (csEGs)
#. by using tissue specific attributes (genomic, functional, PPI structural, etc.) we develop a LighGBM prediction model for EGs. 

.. toctree::
   :maxdepth: 1

   tutorial1/prediction


Experiments 
===========

In this notebook we show how we conducted the experiments in the work published on these topics:

#. starting from downloading the input files used to build EG prediction odels, 
#. by configuring a batch script to iterate cross-validation of a LightGBM model to address binary classification 
in a specific problem (E vs NE, E vs sNE, e vs aE, aE vs sNE)
#. we obtain all measures and their statistics of the experiments.


.. toctree::
   :maxdepth: 1

   tutorial1/experiment
