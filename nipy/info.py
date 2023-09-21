""" This file contains defines parameters for nipy that we use to fill
settings in setup.py, the nipy top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nipy
"""

# nipy version information.  An empty _version_extra corresponds to a
# full release.  '.devN' as a _version_extra string means this is a development
# version
# See: https://www.python.org/dev/peps/pep-0440
_version_major = 0
_version_minor = 5
_version_micro = 1
_version_extra = '.dev1' # For development
# _version_extra = 'rc1' # For release candidate
# _version_extra = '' # For release

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

# Note: this long_description is the canonical place to edit this text.
# It also appears in README.rst, but it should get there by running
# ``tools/refresh_readme.py`` which pulls in this version.
long_description = \
"""
====
NIPY
====

Neuroimaging tools for Python.

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of functional brain imaging data using an open development model.

In NIPY we aim to:

1. Provide an open source, mixed language scientific programming environment
   suitable for rapid development.

2. Create software components in this environment to make it easy to develop
   tools for MRI, EEG, PET and other modalities.

3. Create and maintain a wide base of developers to contribute to this
   platform.

4. To maintain and develop this framework as a single, easily installable
   bundle.

NIPY is the work of many people. We list the main authors in the file
``AUTHOR`` in the NIPY distribution, and other contributions in ``THANKS``.

Website
=======

Current information can always be found at the `NIPY project website
<http://nipy.org/nipy>`_.

Mailing Lists
=============

For questions on how to use nipy or on making code contributions, please see
the ``neuroimaging`` mailing list:

    https://mail.python.org/mailman/listinfo/neuroimaging

Please report bugs at github issues:

    https://github.com/nipy/nipy/issues

You can see the list of current proposed changes at:

    https://github.com/nipy/nipy/pulls

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github;
* Documentation_ for all releases and current development tree;
* Download the `current development version`_ as a tar/zip file;
* Downloads of all `available releases`_.

.. _main repository: https://github.com/nipy/nipy
.. _Documentation: http://nipy.org/nipy
.. _current development version: https://github.com/nipy/nipy/archive/main.zip
.. _available releases: http://pypi.python.org/pypi/nipy

Tests
=====

To run nipy's tests, you will need to install the nose_ Python testing
package.  If you are using Python 2.7, you will also need to install the mock_
testing package - e.g.::

    pip install nose mock

Then::

    python -c "import nipy; nipy.test()"

You can also run nipy's tests with the ``nipnost`` script in the ``tools``
directory of the nipy distribution::

    ./tools/nipnost nipy

``nipnost`` is a thin wrapper around the standard ``nosetests`` program that
is part of the nose package.  Try ``nipnost --help`` to see a large number of
command-line options.

Installation
============

See the latest `installation instructions`_.

License
=======

We use the 3-clause BSD license; the full license is in the file ``LICENSE`` in
the nipy distribution.

.. links:
.. _python: http://python.org
.. _numpy: http://numpy.scipy.org
.. _scipy: http://www.scipy.org
.. _sympy: http://sympy.org
.. _nibabel: http://nipy.org/nibabel
.. _ipython: http://ipython.org
.. _matplotlib: http://matplotlib.org
.. _nose: http://nose.readthedocs.org/en/latest
.. _mock: https://pypi.python.org/pypi/mock
.. _six: https://six.readthedocs.io
.. _installation instructions: http://nipy.org/nipy/users/installation.html
"""

# minimum versions
# Update in readme text above
# Update in .travis.yml
# Update in requirements.txt
NUMPY_MIN_VERSION='1.14'
SCIPY_MIN_VERSION = '1.0.0'
NIBABEL_MIN_VERSION = '2.0'
SYMPY_MIN_VERSION = '1.0'
MAYAVI_MIN_VERSION = '3.0'
CYTHON_MIN_VERSION = '0.12.1'
SIX_MIN_VERSION = '1.9'

NAME                = 'nipy'
MAINTAINER          = "nipy developers"
MAINTAINER_EMAIL    = "neuroimaging@python.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://nipy.org/nipy"
DOWNLOAD_URL        = "http://github.com/nipy/nipy/archives/main"
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "nipy developmers"
AUTHOR_EMAIL        = "neuroimaging@python.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
REQUIRES            = ["numpy", "scipy", "sympy(<1.6)", "nibabel", "six"]
STATUS              = 'beta'

# Versions and locations of optional data packages
NIPY_DATA_URL= 'http://nipy.org/data-packages/'
DATA_PKGS = {'nipy-data': {'min version':'0.2',
                           'relpath':'nipy/data'},
             'nipy-templates': {'min version':'0.2',
                                'relpath':'nipy/templates'}
            }
NIPY_INSTALL_HINT = \
"""You can download and install the package from:

%s

Check the instructions in the ``doc/users/install_data.rst`` file in the nipy
source tree, or online at http://nipy.org/nipy/users/install_data.html

If you have the package, have you set the path to the package correctly?"""

for key, value in DATA_PKGS.items():
    url = '%s%s-%s.tar.gz' % (NIPY_DATA_URL,
                              key,
                              value['min version'])
    value['name'] = key
    value['install hint'] = NIPY_INSTALL_HINT % url

del key, value, url

