# -*- coding: utf-8 -*-
"""
Neuroimaging tools for Python (NIPY).

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of brain imaging data using an open development model.

While
the project is still in its initial stages, packages for file I/O, script
support as well as single subject fMRI and random effects group comparisons
model are currently available.

Specifically, we aim to:

   1. Provide an open source, mixed language scientific programming
      environment suitable for rapid development.

   2. Create sofware components in this environment to make it easy
      to develop tools for MRI, EEG, PET and other modalities.

   3. Create and maintain a wide base of developers to contribute to
      this platform.

   4. To maintain and develop this framework as a single, easily
      installable bundle.

Package Organization 
==================== 
The neuroimaging package contains the following subpackages and modules: 

.. packagetree:: 
   :style: UML  
"""


from version import version as __version__
__status__   = 'alpha'
__url__     = 'http://neuroimaging.scipy.org'

# We require numpy 1.2 for our test suite.  If Tester fails to import,
# check the version of numpy the user has and inform them they need to
# upgrade.
try:
    from neuroimaging.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # If the user has an older version of numpy which does not have
    # the nose test framework, fail gracefully and prompt them to
    # upgrade.
    import numpy as np
    npver = np.__version__.split('.')
    npver = '.'.join((npver[0], npver[1]))
    npver = float(npver)
    if npver < 1.2:
        raise ImportError('Nipy requires numpy version 1.2 or greater. '
                          '\n    You have numpy version %s installed.'
                          '\n    Please upgrade numpy:  '
                          'http://www.scipy.org/NumPy' 
                          % np.__version__)


def _test_local_install():
    """ Warn the user that running with neuroimaging being
        imported locally is a bad idea.
    """
    import os
    if os.getcwd() == os.sep.join(
                            os.path.abspath(__file__).split(os.sep)[:-2]):
        import warnings
        warnings.warn('Running the tests from the install directory may '
                     'trigger some failures')

_test_local_install()
