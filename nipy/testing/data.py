"""
Nipy uses a set of test data that is installed separately.  The test
data should be located in the directory ``~/.nipy/tests/data``.

If the test data is not installed the user should be prompted with the
option to download and install it when they run the examples.  The
module nipy/utils/get_data.py performs this.

Alternatively one could install the data from the svn repository, but
this is much slower::

  $ mkdir -p .nipy/tests/data
  $ svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri .nipy/tests/data

"""
import os

from nipy.utils.get_data import get_data, data_dir


def datapjoin(filename):
    ''' Return result of os.path.join of `filename` to NIPY data path '''
    # If the data directory does not exist, download it.
    if not os.path.exists(data_dir):
        get_data()
    return os.path.join(data_dir, filename)

