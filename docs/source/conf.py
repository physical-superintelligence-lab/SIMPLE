"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add src/ to Python path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SIMPLE'
copyright = '2025, Songlin Wei'
author = 'Songlin Wei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/Numpy docstrings
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxemoji.sphinxemoji",
]

# Theme
html_theme = "furo"

# Optional: control sidebar behavior
html_theme_options = {
    "sidebar_hide_name": False,  # show project name at top
    # Furo auto-expands to current section by default
}

# MyST markdown config
myst_enable_extensions = [
    "dollarmath",  # For inline LaTeX
    "colon_fence"  # ::: fenced blocks
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']


autodoc_mock_imports = ["omni", "carb", "isaacsim"]
