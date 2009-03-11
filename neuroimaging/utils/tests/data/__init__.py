"""
Nipy uses a set of test data that is installed separately.  The test
data should be located in the directory ``~/.nipy/tests/data``.

Install the data in your home directory from the data repository::
  $ mkdir -p .nipy/tests/data
  $ svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri .nipy/tests/data

"""

# Fernando pointed out that we should wrap the test data into a
# tarball and write a pure python function to grab the data for people
# instead of using svn.  Users may not have svn and the mkdir may not
# work on Windows.

from os.path import expanduser, exists, join

from neuroimaging.io.datasource import Repository

# data directory should be: $HOME/.nipy/tests/data
datapath = expanduser(join('~', '.nipy', 'tests', 'data'))

if not exists(datapath):
    msg = 'Nipy data directory is not found!\n%s' % __doc__
    raise IOError(msg)

repository = Repository(datapath)
