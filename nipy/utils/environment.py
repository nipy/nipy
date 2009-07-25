'''
Settings from the system environment
'''

import os
from os.path import join as pjoin
import sys

def get_home_dir():
    """Return the closest possible equivalent to a 'home' directory."""
    dir = os.path.expanduser('~')
    if not os.path.isdir(dir):
        raise OSError('Found HOME directory %s but it does not exist' % dir)
    return dir


def get_nipy_dir():
    """Get the NIPY local directory for this platform and user.
    
    This uses the logic in `get_home_dir` to find the home directory
    and the adds either .nipy or _nipy to the end of the path.

    The code is from the IPython distribution, with thanks
    """
    if os.name == 'posix':
         nipy_dir_def = '.nipy'
    else:
         nipy_dir_def = '_nipy'
    home_dir = get_home_dir()
    nipy_dir = os.path.abspath(os.environ.get('NIPY_DIR',
                                              pjoin(home_dir, nipy_dir_def)))
    return nipy_dir.decode(sys.getfilesystemencoding())


def get_etc_dir():
    ''' Get systemwide configuration file directory '''
    if os.name == 'posix':
        return '/etc'
    raise NotImplementedError

    
