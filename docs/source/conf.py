# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
#import sphinx_pdj_theme
#sys.setrecursionlimit(10000)
sys.path.insert(0, os.path.abspath('../..'))

project = 'HELP'
copyright = '2024, National Research Council of Italy'
author = 'Maurizio Giordano'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
html_static_path = ['_static']
extensions = [
    "sphinx.ext.githubpages", "nbsphinx",
    'sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 'sphinx.ext.intersphinx', 'sphinx.ext.viewcode',
]

exclude_patterns = ['modules.rst', 'notebook.rst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "sphinx_pdj_theme"
#html_theme_path = sphinx_pdj_theme.get_html_theme_path()
#html_theme = "pydata_sphinx_theme"
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/HELP_logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
