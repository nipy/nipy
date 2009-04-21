"""
Download the nipy test/example data from the cirl.berkeley.edu server.

This utility asks the user if they would like to download the file and
if so it:
  - makes the data directory ~/.nipy/tests/data
  - downloads the tarball
  - extracts the tarball

"""

import os
import subprocess
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


def get_data():
    # If urlopen fails it should give the user a useful traceback
    fp = urllib2.urlopen(datafile)
    finfo = fp.info()
    fsize = finfo.getheader('Content-Length')
    msg = 'Nipy example data was not found.\n'
    msg += 'Would you like to download the %s byte file now ([Y]/N)? ' % fsize
    answer = raw_input(msg).lower()
    if not answer or answer == 'y':
        download = True
    else:
        download = False
    if download:
        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            assert os.path.exists(dest_dir)
        local_file = open(dest_file, 'w')
        # read file a chunk at a time so we can provide some feedback
        for chunk in read_chunk(fp):
            local_file.write(chunk)
            sys.stdout.write('.')
            sys.stdout.flush()
        local_file.flush()
        local_file.close()
        # extract the tarball
        extract_tarfile(dest_file, dest_dir)
