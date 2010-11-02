# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
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

import sys, os

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('sphinxext'))

# We load the nipy release info into a dict by explicit execution
rel = {}
execfile('../nipy/info.py', rel)

# Import support for ipython console session syntax highlighting (lives
# in the sphinxext directory defined above)
import ipython_console_highlighting

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'ipython_console_highlighting', 
              'inheritance_diagram', 
              ]

# Current version (as of 11/2010) of numpydoc is only compatible with sphinx >
# 1.0.  We keep copies of this version in 'numpy_ext'.  For a while we will also
# keep a copy of the older numpydoc version to allow compatibility with sphinx
# 0.6
try:
    # With older versions of sphinx, this causes a crash
    import numpy_ext.numpydoc
except ImportError:
    # Older version of sphinx
    extensions.append('numpy_ext_old.numpydoc')
else: # probably sphinx >= 1.0
    extensions.append('numpy_ext.numpydoc')
    autosummary_generate=True

# Matplotlib sphinx extensions
# ----------------------------

# Currently we depend on some matplotlib extentions that are only in
# the trunk, so we've added copies of these files to fall back on,
# since most people install releases.  Once theses extensions have
# been released for a while we should remove this hack.  I'm assuming
# any modifications to these extensions will be done upstream in
# matplotlib!  The matplotlib trunk will have more bug fixes and
# feature updates so we'll try to use that one first.
try:
    import matplotlib.sphinxext
    extensions.append('matplotlib.sphinxext.mathmpl')
    extensions.append('matplotlib.sphinxext.only_directives')
    extensions.append('matplotlib.sphinxext.plot_directive')
except ImportError:
    extensions.append('mathmpl')
    extensions.append('only_directives')
    extensions.append('plot_directive')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General substitutions.
project = 'nipy'
#copyright = ':ref:`2005-2010, Neuroimaging in Python team. <nipy-software-license>`' 
copyright = '2005-2010, Neuroimaging in Python team' 

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
version = rel['__version__']
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

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class
# [howto/manual]).

latex_documents = [
  ('documentation', 'nipy.tex', 'Neuroimaging in Python Documentation',
   ur'Neuroimaging in Python team.','manual'),
  ]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = True

# Additional stuff for the LaTeX preamble.
latex_preamble = """
   \usepackage{amsmath}
   \usepackage{amssymb}
   % Uncomment these two if needed
   %\usepackage{amsfonts}
   %\usepackage{txfonts}
"""

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True
