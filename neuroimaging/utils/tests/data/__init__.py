"""Information used for locating nipy test data.

Nipy uses a set of test data that is installed separately.  The test
data should be located in the directory ``~/.nipy/tests/data``.

Install the data in your home directory from the data repository::
  $ mkdir -p .nipy/tests/data
  $ svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri .nipy/tests/data

"""
from os.path import expanduser, exists, join

from neuroimaging.data_io.datasource import Repository

# data directory should be: $HOME/.nipy/tests/data
datapath = expanduser(join('~', '.nipy', 'tests', 'data'))

if not exists(datapath):
    raise IOError, 'Nipy data directory is not found!'

repository = Repository(datapath)

