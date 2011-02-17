""" This file contains defines parameters for nipy that we use to fill
settings in setup.py, the nipy top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nipy
"""

# nipy version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 1
_version_micro = 2
_version_extra = '.dev'

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
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = \
"""
====
NIPY
====

Neuroimaging tools for Python

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of brain imaging data using an open development model.

The project is still in its initial stages, but we have packages for file I/O,
script support as well as single subject fMRI and random effects group
comparisons models.

In NIPY we aim to:

   1. Provide an open source, mixed language scientific programming
      environment suitable for rapid development.

   2. Create sofware components in this environment to make it easy
      to develop tools for MRI, EEG, PET and other modalities.

   3. Create and maintain a wide base of developers to contribute to
      this platform.

   4. To maintain and develop this framework as a single, easily
      installable bundle.
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
NUMPY_MIN_VERSION='1.0'
SCIPY_MIN_VERSION = '0.5'
NIBABEL_MIN_VERSION = '1.0'
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

Check the instructions in the INSTALL file in the nipy source tree, or online at
http://nipy.org/nipy/stable/devel/development_quickstart.html#optional-data-packages

If you have the package, have you set the path to the package correctly?"""

for key, value in DATA_PKGS.items():
    url = '%s%s-%s.tar.gz' % (NIPY_DATA_URL,
                              key,
                              value['min version'])
    value['name'] = key
    value['install hint'] = NIPY_INSTALL_HINT % url

del key, value, url

