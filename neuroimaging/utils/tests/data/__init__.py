"""
Nipy uses a set of test data that is installed separately.  The test
data should be located in the directory ``~/.nipy/tests/data``.

If the test data is not installed the user should be prompted with the
option to download and install it when they run the examples.  The
module neuroimaging/utils/get_data.py performs this.

Alternatively one could install the data from the svn repository, but
this is much slower::

  $ mkdir -p .nipy/tests/data
  $ svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri .nipy/tests/data

"""

import os

from neuroimaging.io.datasource import Repository
from neuroimaging.utils.get_data import get_data

# data directory should be: $HOME/.nipy/tests/data
datapath = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))

# If the data directory does not exist, download it.
if not os.path.exists(datapath):
    get_data()

repository = Repository(datapath)
