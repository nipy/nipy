"""
Utilities to download data from the web.

Default test data
------------------

Download the nipy test/example data from the cirl.berkeley.edu server.

This utility asks the user if they would like to download the file and
if so it:
  - makes the data directory ~/.nipy/tests/data
  - downloads the tarball
  - extracts the tarball

"""

import os
import sys
import tarfile
import urllib2

block_size = int(512e3)

url = 'https://cirl.berkeley.edu/nipy/'
filename = 'nipy_data.tar.gz'
datafile = os.path.join(url, filename)

dest_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
dest_file = os.path.join(dest_dir, filename)

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
    


def get_data():
    fp = urllib2.urlopen(datafile)
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
        download(datafile, dest_file)
        # extract the tarball
        extract_tarfile(dest_file, dest_dir)
