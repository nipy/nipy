""" This file contains defines parameters for nipy that we use to fill
settings in setup.py, the nipy top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nipy
"""

# nipy version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 2
_version_micro = 0
_version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'A python package for analysis of neuroimaging data'

# Note: this long_description is actually a copy/paste from the top-level
# README.rst, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = \
"""
====
NIPY
====

Neuroimaging tools for Python.

The aim of NIPY is to produce a platform-independent Python environment for the
analysis of functional brain imaging data using an open development model.

In NIPY we aim to:

1. Provide an open source, mixed language scientific programming
    environment suitable for rapid development.

2. Create sofware components in this environment to make it easy
    to develop tools for MRI, EEG, PET and other modalities.

3. Create and maintain a wide base of developers to contribute to
    this platform.

4. To maintain and develop this framework as a single, easily
    installable bundle.

NIPY is the work of many people. We list the main authors in the file ``AUTHOR``
in the NIPY distribution, and other contributions in ``THANKS``.

Website
=======

Current information can always be found at the NIPY website::

    http://nipy.org/nipy

Mailing Lists
=============

Please see the developer's list::

    http://projects.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nipy
.. _Documentation: http://nipy.org/nipy
.. _current trunk: http://github.com/nipy/nipy/archives/master
.. _available releases: http://github.com/nipy/nipy/downloads

Dependencies
============

To run NIPY, you will need:

* python_ >= 2.5.  We don't yet run on python 3, sad to say.
* numpy_ >= 1.2
* scipy_ >= 0.7.0
* sympy_ >= 0.6.6
* nibabel_ >= 1.2

You will probably also like to have:

* ipython_ for interactive work
* matplotlib_ for 2D plotting
* mayavi_ for 3D plotting

.. _python: http://python.org
.. _numpy: http://numpy.scipy.org
.. _scipy: http://www.scipy.org
.. _sympy: http://sympy.org
.. _nibabel: http://nipy.org/nibabel
.. _ipython: http://ipython.scipy.org
.. _matplotlib: http://matplotlib.sourceforge.net
.. _mayavi: http://code.enthought.com/projects/mayavi/

License
=======

We use the 3-clause BSD license; the full license is in the file ``LICENSE`` in
the nipy distribution.
"""

NAME                = 'nipy'
MAINTAINER          = "nipy developers"
MAINTAINER_EMAIL    = "nipy-devel@neuroimaging.scipy.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://nipy.org/nipy"
DOWNLOAD_URL        = "http://github.com/nipy/nipy/archives/master"
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "nipy developmers"
AUTHOR_EMAIL        = "nipy-devel@neuroimaging.scipy.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
REQUIRES            = ["numpy", "scipy", "sympy"]
STATUS              = 'alpha'

# versions
NUMPY_MIN_VERSION='1.2'
SCIPY_MIN_VERSION = '0.5'
NIBABEL_MIN_VERSION = '1.2'
SYMPY_MIN_VERSION = '0.6.6'
MAYAVI_MIN_VERSION = '3.0'
CYTHON_MIN_VERSION = '0.12.1'

# Versions and locations of optional data packages
NIPY_DATA_URL= 'http://nipy.sourceforge.net/data-packages/'
DATA_PKGS = {'nipy-data': {'min version':'0.2',
                           'relpath':'nipy/data'},
             'nipy-templates': {'min version':'0.2',
                                'relpath':'nipy/templates'}
            }
NIPY_INSTALL_HINT = \
"""You can download and install the package from:

%s

Check the instructions in the ``doc/users/install_data.rst`` file in the nipy
source tree, or online at http://nipy.org/nipy/stable/users/install_data.html

If you have the package, have you set the path to the package correctly?"""

for key, value in DATA_PKGS.items():
    url = '%s%s-%s.tar.gz' % (NIPY_DATA_URL,
                              key,
                              value['min version'])
    value['name'] = key
    value['install hint'] = NIPY_INSTALL_HINT % url

del key, value, url

