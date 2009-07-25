"""
Utilities to find files from NIPY data packages

"""

import os
from os.path import join as pjoin
import glob
import sys
import tarfile
import urllib2
import warnings
import ConfigParser

from .environment import get_nipy_dir, get_etc_dir

NIPY_URL= 'https://cirl.berkeley.edu/mb312/nipy-data/'


def _cfg_value(fname, section='DATA', value='path'):
    configp =  ConfigParser.ConfigParser()
    readfiles = configp.read(fname)
    if not readfiles:
        return ''
    try:
        return configp.get(section, value)
    except ConfigParser.Error:
        return ''


def get_data_path():
    ''' Return specified or guessed locations of NIPY data files '''
    try:
        var = os.environ['NIPY_DATA_PATH']
    except KeyError:
        paths = []
    else:
        paths = var.split(os.path.pathsep)
    np_cfg = pjoin(get_nipy_dir(), 'config.ini')
    nipy_etc = pjoin(get_etc_dir(), 'nipy')
    config_files = sorted(glob.glob(pjoin(nipy_etc, '*.ini')))
    for fname in [np_cfg] + config_files:
        var = _cfg_value(fname)
        if var:
            paths += var.split(os.path.pathsep)
    if not paths:
        paths = [pjoin(sys.prefix, 'share', 'nipy')]
    return paths
    

class Datasource(object):
    ''' Simple class to add base path to relative path '''
    def __init__(self, base_path):
        ''' Initialize datasource

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths

        Examples
        --------
        >>> from os.path import join as pjoin
        >>> repo = Datasource(pjoin('a', 'path'))
        >>> fname = repo.get_filename('somedir', 'afile.txt')
        >>> fname == pjoin('a', 'path', 'somedir', 'afile.txt')
        True
        '''
        self.base_path = base_path

    def get_filename(self, *path_parts):
        ''' Prepend base path to ``*path_parts`` '''
        return pjoin(self.base_path, *path_parts)


class VersionedDatasource(Datasource):
    ''' Datasource with version information in config file '''
    def __init__(self, base_path):
        Datasource.__init__(self, base_path)
        self.config = ConfigParser.SafeConfigParser()
        self.config.read(self.get_filename('config.ini'))
        self.version = self.config.get('DEFAULT', 'version')
        major, minor = self.version.split('.')
        self.major_version = int(major)
        self.minor_version = int(minor)


def find_data_dir(root_dirs, *names):
    repo_relative = pjoin(*names)
    for path in root_dirs:
        pth = pjoin(path, repo_relative)
        if os.path.isdir(pth):
            return pth
    raise OSError('Could not find datasource %s in data path %s' %
                  (repo_relative,
                   os.path.pathsep.join(root_dirs)))


def make_datasource(*names):
    root_dirs = get_data_path()
    pth = find_data_dir(root_dirs, *names)
    return VersionedDatasource(pth)


