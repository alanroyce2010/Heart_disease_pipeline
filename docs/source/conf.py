# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Heart Disease MLOps'
copyright = '2026, Alan'
author = 'Alan'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # for Google/Numpy docstrings
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"
import os
import sys
sys.path.insert(0, os.path.abspath("../../pipeline/src"))
