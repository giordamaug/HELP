
Example 1: Identification of EGs 
================================

This example shows how to use HELP for computing two-class and three-class labeling of EGs based on tissue or disease-related information. 
The workflow involves the following steps 

#. Install HELP from GitHub
#. Download the input files
#. Load the input files
#. Filter the information to be exploited
#. Apply two-class or three-class labeling

Steps 1.-3. are needed only once, while steps 4.-5. are differentiated and executed three times to compute:

* Example 1.1 two-class labeling of EGs based on tissue information
* Example 1.2 three-class labeling of EGs based on tissue information
* Example 1.3 two-class labeling of EGs based on disease-related information

.. toctree::
   :maxdepth: 1

   examples/labelling

Example 2: Identification of tissue-specific EGs
================================================

This example shows how to use HELP for identifying tissue-specific EGs. 
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

Example 3: Prediction of tissue-specific EGs
============================================

This example shows how to use HELP for identifying tissue-specific EGs. 
The workflow involves the following steps

#. Install HELP from GitHub
#. Download the input files
#. Processing tissue-specific attributes
#. Prediction
#. True Positive rates of context-specific EGs

Finally, the example shows various ways of visualizing the obtained results.

.. toctree::
   :maxdepth: 1

   examples/prediction