# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import fenicsx_pctools
import jupytext_process

jupytext_process.process()

project = "FEniCSx-pctools"
copyright = "2022-2023, FEniCSx-pctools Authors"
author = "FEniCSx-pctools Authors"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = fenicsx_pctools.__version__
# The full version, including alpha/beta/rc tags.
release = fenicsx_pctools.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md"]

myst_enable_extensions = ["dollarmath"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "imported-members": True,
    "undoc-members": True,
}
autosummary_generate = True
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_use_admonition_for_notes = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Bibliography configuration  ---------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
# bibtex_reference_style = "author_year"
