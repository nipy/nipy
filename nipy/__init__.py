
import os

from .info import LONG_DESCRIPTION as __doc__
from .info import STATUS as __status__
from .info import URL as __url__
from .info import __version__


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
from nipy.core.api import is_image
from nipy.io.api import as_image, load_image, save_image

# Set up package information function
from .pkg_info import get_pkg_info as _get_pkg_info

get_info = lambda : _get_pkg_info(os.path.dirname(__file__))

# Cleanup namespace
del _test_local_install
# If this file is exec after being imported, the following lines will
# fail
try:
    del version
    del Tester
except:
    pass
