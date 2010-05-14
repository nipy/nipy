# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set fileencoding=utf-8 ft=python sts=4 ts=4 sw=4 et:
from .info import long_description as __doc__

from .version import version as __version__
__status__   = 'alpha'
__url__     = 'http://neuroimaging.scipy.org'

# We require numpy 1.2 for our test suite.  If Tester fails to import,
# check the version of numpy the user has and inform them they need to
# upgrade.
try:
    from nipy.testing import Tester
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
    """ Warn the user that running with nipy being
        imported locally is a bad idea.
    """
    import os
    if os.getcwd() == os.sep.join(
                            os.path.abspath(__file__).split(os.sep)[:-2]):
        import warnings
        warnings.warn('Running the tests from the install directory may '
                     'trigger some failures')

_test_local_install()


# Add to top-level namespace
from nipy.io.api import load_image, save_image, as_image
from nipy.core.api import is_image

# Cleanup namespace
del _test_local_install
# If this file is exec after being imported, the following lines will
# fail
try:
    del version
    del Tester
except:
    pass


