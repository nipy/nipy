# -*- coding: utf-8 -*-
#
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
import os
import sys

sys.path.append(os.path.abspath('../sphinxext'))

execfile('../conf.py')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of documents that shouldn't be included in the build.
unused_docs = []


# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'nipy.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'NIPY Home'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = True

# Output file base name for HTML help builder.
htmlhelp_basename = project


# Options for LaTeX output
# ------------------------
latex_documents = []
latex_preamble = ""
