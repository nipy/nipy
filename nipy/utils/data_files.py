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
import sys
import tarfile
import urllib2

from nipy.__config__ import nipy_info

# Constants
block_size = int(512e3)

core_data_url = 'https://cirl.berkeley.edu/nipy/'
core_data_filename = 'nipy_data.tar.gz'

data_dir = nipy_info['data_dir']
example_data_dir = nipy_info['example_data_dir']

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
        chunk = fp.read(block_size)
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

def fetch_core_data():
    """ Fetch the core data from the nipy website.

        This utility asks the user if they would like to download the file and
        if so it:
        - makes the data directory according to the site.cfg 
        - downloads the tarball
        - extracts the tarball
        - remove the tarball

    """
    core_datafile = os.path.join(core_data_url, core_data_filename)
    dest_file = os.path.join(data_dir, core_data_filename)
    fp = urllib2.urlopen(core_data_filename)
    finfo = fp.info()
    fsize = finfo.getheader('Content-Length')
    msg = 'Nipy example data was not found.\n'
    msg += 'Would you like to download the %s byte file now ([Y]/N)? ' % fsize
    answer = raw_input(msg).lower()
    if not answer or answer == 'y':
        if_download = True
    else:
        if_download = False
    if if_download:
        download(core_data_filename, data_dir)
        # extract the tarball
        extract_tarfile(dest_file, data_dir)
        os.remove(dest_file)


def get_data_file(filename):
    """ Return the path to the NIPY data `filename` if available, and
    offer downloading it if not.

    Nipy uses a set of data that is installed separately.  The test
    data should be located in the directory specified at install time
    in the site.cfg (default: ``~/.nipy/data``).

    If the data is not installed the user should be prompted with the
    option to download and install it when they run the examples.
    """
    # If the data directory does not exist, download it.
    if not os.path.exists(data_dir):
        fetch_core_data()
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
    full_path = os.path.join(example_data_dir, filename)
    if not os.path.exists(full_path):
        if url is False:
            raise OSError
        download(url, full_path)
    return full_path
 
