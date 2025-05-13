# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SuperDARN'
copyright = '2025, Stephen Mander, Maria Walach'
author = 'Stephen Mander, Maria Walach'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    # "sphinx_multiversion",

]
autoclass_content = 'both'



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']
# html_sidebars = {
#     "**": [
#         "sidebar/brand.html",
#         "sidebar/search.html",
#         "sidebar/scroll-start.html",
#         "sidebar/navigation.html",
#         "sidebar/versions.html",
#         "sidebar/scroll-end.html",
#     ],
# }