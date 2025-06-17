# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import fenicsx_pctools

sys.path.insert(0, os.path.abspath("."))

import jupytext_process  # noqa: I001


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
    "sphinx_autodoc_typehints",  # NOTE: Must come after Napoleon extension!
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
    "private-members": True,
    "special-members": False,
    "ignore-module-all": True,  # TODO: Not working - missing private functions (unless in __all__)!
}
autosummary_generate = True
autosummary_ignore_module_all = False
autoclass_content = "both"

napoleon_google_docstring = True
napoleon_use_admonition_for_notes = False

# -- Options for Intersphinx  ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "dolfinx": ("https://docs.fenicsproject.org/dolfinx/main/python", None),
    "petsc4py": ("https://petsc.org/release/petsc4py", None),
    "ufl": ("https://docs.fenicsproject.org/ufl/main", None),
}

# -- Options for Napoleon  ---------------------------------------------------
napoleon_use_rtype = False

# -- Options for typehints  --------------------------------------------------
# https://github.com/tox-dev/sphinx-autodoc-typehints

always_use_bars_union = True
typehints_fully_qualified = True
typehints_defaults = "braces-after"
typehints_document_rtype = True
typehints_document_rtype_none = False
typehints_use_rtype = False
typehints_use_signature = False
typehints_use_signature_return = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Bibliography configuration  ---------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
# bibtex_reference_style = "author_year"


def skip_member(app, what, name, obj, skip, opts):
    # Skip private members from abc
    if name in [
        "_abc_cache",
        "_abc_impl",
        "_abc_negative_cache",
        "_abc_registry",
    ]:
        return True
    else:
        return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
