
Example 1: Identification of context-specific EGs 
=================================================

This example shows how to use HELP for computing two-class and three-class labelling of EGs based on tissue or disease-related information. 
The workflow involves the following steps 

#. Install HELP from GitHub
#. Download the input files
#. Load the input files
#. Filter the information to be exploited
#. Apply two-class or three-class labelling

Steps 1.-3. are needed only once, while steps 4.-5. are differentiated and executed three times to compute:

* Example 1.1 two-class labelling of EGs based on tissue information
* Example 1.2 three-class labelling of EGs based on tissue information
* Example 1.3 two-class labelling of EGs based on disease-related information

.. toctree::
   :maxdepth: 1

   examples/labelling

Example 2: Identification of uncommon context-specific EGs
==========================================================

This example shows how to use HELP for identifying uncommon tissue-specific EGs. 
The workflow involves the following steps

#. Install HELP from GitHub
#. Download the input files
#. Load the input files
#. Filter the information to be exploited
#. Compute EGs common to all tissues (pan-tissue EGs)
#. Subtract pan-tissue EGs from those of the chosen tissue

Finally, the example shows various ways of visualizing the obtained results.

.. toctree::
   :maxdepth: 1

   examples/csegs

Example 3: Prediction of EGs for a tissue
=========================================

This example shows how to use HELP to estimate the performance of EG prediction for a tissue.  
The workflow involves the following steps 

#. Install HELP from GitHub
#. Download the input files
#. Load the input files and process the tissue attributes 
#. Estimate the performance of EGs prediction

As a further step, this example shows how to compute the True Positive Rate (TPR) for ucsEGs and csEGs and show their bar plot. 

.. toctree::
   :maxdepth: 1

   examples/prediction

Example 5: Reproduce the experiments reported in the HELP paper
===============================================================

This example shows how to use HELP to reproduce the experiments reported in the HELP paper.
The workflow involves the following steps

#. Install HELP from GitHub
#. Download the input files
#. Download the script for the experiments and show its man page 
#. Run the E vs NE experiments
#. Run the E vs sNE experiments

.. toctree::
   :maxdepth: 1

   examples/experiment
