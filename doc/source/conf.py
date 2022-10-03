# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
import jupytext_process


jupytext_process.process()

project = 'fenicsx-pctools'
copyright = '2022, FEniCSx-pctools Authors'
author = 'FEniCSx-pctools Authors'
release = '0.5.0.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']

myst_enable_extensions = ["dollarmath"]

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'imported-members': True,
    'undoc-members': True,
}
autosummary_generate = True
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Bibliography configuration  ---------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'alpha'
bibtex_reference_style = "author_year"
