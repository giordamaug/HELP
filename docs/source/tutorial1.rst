EG detection and predition for two tissues 
==========================================

In notebook we show how

#. starting from the CRISPR file we apply the HELP labelling algorithm to detect essential genes (EGs) and non-essentials
genes (NEGs) in two tissues and in all tissues (pan-tissue).
#. by subtracting pan-tissue EGs from a tissue EGs list we get the context-specific EGs (csEGs)
#. by using tissue specific attributes (genomic, functional, PPI structural, etc.) we develop a LighGBM prediction model for EGs. 

.. toctree::
   :maxdepth: 1

   tutorial1/Workflow_1