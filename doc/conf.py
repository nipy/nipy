# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# sampledoc documentation build configuration file, created by
# sphinx-quickstart on Tue Jun  3 12:40:24 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import os
import sys
from importlib import import_module

import sphinx
import sphinx.ext.doctest

# Doc generation depends on being able to import project
project = 'nipy'
try:
    project_module = import_module(project)
except ImportError:
    raise RuntimeError(f'Cannot import {project}, please investigate')

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('sphinxext'))

# General configuration
# ---------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'texext.mathcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'texext.math_dollar',
    'numpydoc',
    'sphinx.ext.inheritance_diagram',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Autosummary on
autosummary_generate=True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# copyright = ':ref:`2005-2018, Neuroimaging in Python team.
# <nipy-software-license>`'
copyright = '2005-2023, Neuroimaging in Python team'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
version = project_module.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# List of directories, relative to source directories, that shouldn't
# be searched for source files.
# exclude_trees = []

# what to put into API doc (just class doc, just init, or both)
autoclass_content = 'class'

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Options for HTML output
# -----------------------
#
# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'sphinxdoc'

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'nipy.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'NIPY Documentation'

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Content template for the index page.
html_index = 'index.html'

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {'index': 'indexsidebar.html'}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = project

# Options for LaTeX output
# ------------------------

# Additional stuff for the LaTeX preamble.
_latex_preamble = r"""
   \usepackage{amsmath}
   \usepackage{amssymb}
   % Uncomment these two if needed
   %\usepackage{amsfonts}
   %\usepackage{txfonts}
"""

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
    'preamble': _latex_preamble,
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class
# [howto/manual]).

latex_documents = [
  ('documentation', 'nipy.tex', 'Neuroimaging in Python Documentation',
   'Neuroimaging in Python team.','manual'),
  ]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

if sphinx.version_info[:2] < (1, 4):
    # For "manual" documents, if this is true, then toplevel headings are parts,
    # not chapters.
    latex_use_parts = True
else:  # Sphinx >= 1.4
    latex_toplevel_sectioning = 'part'

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

# Doctesting helpers
doctest_global_setup = """\
import numpy as np
from numpy import array

try:
    import vtk
except ImportError:
    vtk = None
"""

_sedd = sphinx.ext.doctest.doctest
doctest_default_flags = (_sedd.ELLIPSIS | _sedd.IGNORE_EXCEPTION_DETAIL |
                         _sedd.DONT_ACCEPT_TRUE_FOR_1 |
                         _sedd.NORMALIZE_WHITESPACE)

# Numpy extensions
# ----------------
# Worked out by Steven Silvester in
# https://github.com/scikit-image/scikit-image/pull/1356
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
