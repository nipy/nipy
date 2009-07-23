"""
Utilities to download data files from the web and store them in a local
cache.

Core set of data files
------------------------

The nipy data files are downloaded from the cirl.berkeley.edu server as a
tarball and stored in a local cache.

The data is stored in subversion, at::

    $ svn co http://neuroimaging.scipy.org/svn/ni/data/trunk/fmri


Example files
--------------

Example files can be dowloaded on the fly from an url and stored locally.
"""

import os
from os.path import join as pjoin
import sys
import tarfile
import urllib2
import warnings
import ConfigParser

from nipy.__config__ import nipy_info

# Constants
BLOCK_SIZE = int(512e3)

NIPY_URL= 'https://cirl.berkeley.edu/mb312/nipy-data/'

def get_data_path():
    ''' Return specified or guessed location of NIPY data files '''
    try:
        return os.environ['NIPY_DATA_PATH']
    except KeyError:
        pass
    try:
        return nipy_info['data_path']
    except KeyError:
        pass
    # Now we have to guess
    if sys.platform == 'win32':
        prefixes = ['C:\\']
    else:
        midfixes = [
            '/usr/share',
            '/usr/local/share',
            '/opt/share',
            '/opt/local/share']
        prefixes = midfixes[:] # system paths
        # add user version of these paths
        for midfix in midfixes + ['/share']:
            prefixes.append(
                os.path.expanduser('~' + midfix))
    for prefix in prefixes:
        pth = pjoin(prefix, 'nipy')
        if os.path.isdir(pth):
            return pth
    

class Repository(object):
    ''' Simple class to add base path to relative path '''
    def __init__(self, base_path):
        ''' Initialize repository

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths

        Examples
        --------
        >>> repo = Repository('/some/path')
        >>> repo.full_path('afile.txt')
        '/some/path/afile.txt'
        '''
        self.base_path = base_path

    def full_path(self, relative_path):
        ''' Prepend base path to ``relative_path`` '''
        return pjoin(self.base_path, relative_path)


class VersionedRepository(Repository):
    ''' Simple repository with version information in config file '''
    def __init__(self, base_path):
        Repository.__init__(self, base_path)
        self.config = ConfigParser.SafeConfigParser()
        self.config.read(self.full_path('config.ini'))
        self.version = self.config.get('DEFAULT', 'version')
        major, minor = self.version.split('.')
        self.major_version = int(major)
        self.minor_version = int(minor)


def make_repositories(root_path, names, repo_maker=Repository):
    repos = [None] * len(names)
    if not root_path:
        return repos
    for i, name in enumerate(names):
        pth = pjoin(root_path, name)
        if os.path.isdir(pth):
            repos[i] = repo_maker(pth)
    return repos


template_repo, data_repo = make_repositories(
    get_data_path(),
    ['templates', 'data'],
    VersionedRepository)


def data_warn():
    if template_repo is None:
        warnings.warn(
"""
Cannot find template data files; are they installed?

We use template files for standard spatial processing and visualization

Please go to %s; download and install the latest nipy-templates package
""" % NIPY_URL)
    if data_repo is None:
        warnings.warn(
"""
Cannot find example data files; are they installed?

We use the data files to run more extensive tests and for several examples.

Please go to %s; download and install the latest nipy-data package
""" % NIPY_URL)

    
################################################################################
# Utilities

def extract_tarfile(filename, dstdir):
    """Extract tarfile to the destination directory."""
    sys.stdout.write('\nExtracting tarfile: %s\n' % filename)
    tar = tarfile.open(filename)
    tar.extractall(dstdir)
    tar.close()


def read_chunk(fp):
    while True:
        chunk = fp.read(BLOCK_SIZE)
        if not chunk:
            break
        yield chunk


def download(url, filename):
    """ Download a file from a URL to a local filename

    Parameters
    ----------
    url: string
        URL to download from, eg
        'https://cirl.berkeley.edu/nipy/nipy_data.tar.gz'
    filename: string
        path to the local file to download to.
    """
    fp = urllib2.urlopen(url)
    finfo = fp.info()
    fsize = finfo.getheader('Content-Length')
    print 'Downloading %(filename)s from %(url)s (size: %(fsize)s byte)' \
                    % locals()
    if os.sep in filename and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # read file a chunk at a time so we can provide some feedback
    local_file = open(filename, 'w')
    for chunk in read_chunk(fp):
        local_file.write(chunk)
        sys.stdout.write('.')
        sys.stdout.flush()
    local_file.close()
    

################################################################################
# Core data 

def check_fetch_data(data_dir, data_url, tar_filename, descrip):
    """ Check for, if necessary fetch given data from the nipy website

    This utility asks the user if they would like to download the file and
    if so it:
    - makes the data directory according to the site.cfg 
    - downloads the tarball
    - extracts the tarball
    - removes the tarball

    """
    if os.path.isdir(data_dir):
        return
    tar_url = os.path.join(data_url, tar_filename)
    dest_file = os.path.join(data_dir, tar_filename)
    fp = urllib2.urlopen(tar_url)
    finfo = fp.info()
    fsize = finfo.getheader('Content-Length')
    msg = 'Nipy %s was not found.\n' % descrip
    msg += 'Would you like to download the %s byte file now ([Y]/N)? ' % fsize
    answer = raw_input(msg).lower()
    if not answer or answer == 'y':
        if_download = True
    else:
        if_download = False
    if if_download:
        download(tar_url, data_dir)
        # extract the tarball
        extract_tarfile(dest_file, data_dir)
        os.remove(dest_file)


def get_template_file(filename):
    """ Return the path to the NIPY data `filename` if available, and
    offer downloading it if not.

    Nipy uses a set of data that is installed separately.  The test
    data should be located in the directory specified at install time
    in the site.cfg (default: ``~/.nipy/data``).

    If the data is not installed the user should be prompted with the
    option to download and install it when they run the examples.
    """
    # If the data directories do not exist, download it.
    check_fetch_data(TEMPLATE_DIR, NIPY_URL,
                     TEMPLATE_TAR, 'templates')
    return os.path.join(data_dir, filename)


################################################################################
# Example data files downloaded on the fly


def get_example_file(filename, url=False):
    """ Retrieve the full filename of an example dataset. If the file does not
        exist, and an url is given, it is automatically downloaded from the
        web.

        Parameters
        ----------
        filename: string
            filename of the data file of interest
        url: string or False
            if url is not false, this is the url to automatically
            download the data from.

        Return
        ------
        The full path to the file.

        Notes
        ------
        If url is False, and the file cannot be found, raises an OSError
        The file is stored to a path defined at build time in the
        site.cfg.
    """
    check_fetch_data(EXAMPLE_DATA_DIR, NIPY_URL,
                     EXAMPLE_DATA_TAR, 'example data')
    full_path = os.path.join(EXAMPLE_DATA_DIR, filename)
    if not os.path.exists(full_path):
        if url is False:
            raise OSError
        download(url, full_path)
    return full_path
 
