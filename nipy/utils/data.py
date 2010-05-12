# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to find files from NIPY data packages

"""
import os
from os.path import join as pjoin
import glob
import sys
import ConfigParser

from .environment import get_nipy_user_dir, get_nipy_system_dir

NIPY_URL= 'http://nipy.sourceforge.net/data-packages/'


class DataError(OSError):
    pass


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
        ''' Prepend base path to `*path_parts`

        We make no check whether the returned path exists.

        Parameters
        ----------
        *path_parts : sequence of strings

        Returns
        -------
        fname : str
           result of ``os.path.join(*path_parts), with
           ``self.base_path`` prepended

        '''
        return pjoin(self.base_path, *path_parts)

    def list_files(self, relative=True):
        ''' Recursively list the files in the data source directory.

            Parameters
            ----------
            relative: bool, optional
                If True, path returned are relative to the base paht of
                the data source.

            Returns
            -------
            file_list: list of strings
                List of the paths of all the files in the data source.

        '''
        out_list = list()
        for base, dirs, files in os.walk(self.base_path):
            if relative:
                base = base[len(self.base_path)+1:]
            for filename in files:
                out_list.append(pjoin(base, filename))
        return out_list


class VersionedDatasource(Datasource):
    ''' Datasource with version information in config file

    '''
    def __init__(self, base_path, config_filename=None):
        ''' Initialize versioned datasource

        We assume that there is a configuration file with version
        information in datasource directory tree.

        The configuration file contains an entry like::
        
           [DEFAULT]
           version = 0.3

        The version should have at least a major and a minor version
        number in the form above. 

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths
        config_filaname : None or str
           relative path to configuration file containing version

        '''
        Datasource.__init__(self, base_path)
        if config_filename is None:
            config_filename = 'config.ini'
        self.config = ConfigParser.SafeConfigParser()
        cfg_file = self.get_filename(config_filename)
        readfiles = self.config.read(cfg_file)
        if not readfiles:
            raise DataError('Could not read config file %s' % cfg_file)
        try:
            self.version = self.config.get('DEFAULT', 'version')
        except ConfigParser.Error:
            raise DataError('Could not get version from %s' % cfg_file)
        version_parts = self.version.split('.')
        self.major_version = int(version_parts[0])
        self.minor_version = int(version_parts[1])
        self.version_no = float('%d.%d' % (self.major_version,
                                           self.minor_version))


def _cfg_value(fname, section='DATA', value='path'):
    """ Utility function to fetch value from config file """
    configp =  ConfigParser.ConfigParser()
    readfiles = configp.read(fname)
    if not readfiles:
        return ''
    try:
        return configp.get(section, value)
    except ConfigParser.Error:
        return ''


def get_data_path():
    ''' Return specified or guessed locations of NIPY data files

    The algorithm is to return paths, extracted from strings, where
    strings are found in the following order:

    #. The contents of environment variable ``NIPY_DATA_PATH`` 
    #. Any section = ``DATA``, key = ``path`` value in a ``config.ini``
       file in your nipy user directory (found with
       ``get_nipy_user_dir()``)
    #. Any section = ``DATA``, key = ``path`` value in any files found
       with a ``sorted(glob.glob(os.path.join(sys_dir, '*.ini')))``
       search, where ``sys_dir`` is found with ``get_nipy_system_dir()``
    #. The result of ``os.path.join(sys.prefix, 'share', 'nipy')``
    #. The result of ``get_nipy_user_dir()``

    Therefore, any paths found in ``NIPY_DATA_PATH`` will be searched
    before paths found in the user directory ``config.ini``

    Parameters
    ----------
    None

    Returns
    -------
    paths : sequence of paths

    Examples
    --------
    >>> pth = get_data_path()
    '''
    paths = []
    try:
        var = os.environ['NIPY_DATA_PATH']
    except KeyError:
        pass
    else:
        if var:
            paths = var.split(os.path.pathsep)
    np_cfg = pjoin(get_nipy_user_dir(), 'config.ini')
    np_etc = get_nipy_system_dir()
    config_files = sorted(glob.glob(pjoin(np_etc, '*.ini')))
    for fname in [np_cfg] + config_files:
        var = _cfg_value(fname)
        if var:
            paths += var.split(os.path.pathsep)
    paths.append(pjoin(sys.prefix, 'share', 'nipy'))
    paths.append(pjoin(get_nipy_user_dir()))
    return paths
    

def find_data_dir(root_dirs, *names):
    ''' Find relative path given path prefixes to search

    We raise a DataError if we can't find the relative path
    
    Parameters
    ----------
    root_dirs : sequence of strings
       sequence of paths in which to search for data directory
    *names : sequence of strings
       sequence of strings naming directory to find. The name to search
       for is given by ``os.path.join(*names)``

    Returns
    -------
    data_dir : str
       full path (root path added to `*names` above)

    '''
    ds_relative = pjoin(*names)
    for path in root_dirs:
        pth = pjoin(path, ds_relative)
        if os.path.isdir(pth):
            return pth
    raise DataError('Could not find datasource "%s" in data path "%s"' %
                   (ds_relative,
                    os.path.pathsep.join(root_dirs)))


def make_datasource(*names, **kwargs):
    ''' Return datasource `*names` as found in `data_path`

    `data_path` is the only allowed keyword argument.
    
    The relative path of the directory we are looking for is given by
    ``os.path.join(*names)``.  We search for this path in the list of
    paths given by `data_path`.  By default `data_path` is given by
    ``get_data_path()`` in this module.

    If we can't find the relative path, raise a DataError

    Parameters
    ----------
    *names : sequence of strings
       The relative path to search for is given by
       ``os.path.join(*names)``
    data_path : sequence of strings or None, optional
       sequence of paths in which to search for data.  If None (the
       default), then use ``get_data_path()``
       
    Returns
    -------
    datasource : ``VersionedDatasource``
       An initialized ``VersionedDatasource`` instance

    '''
    if any(key for key in kwargs if key != 'data_path'):
        raise ValueError('Unexpected keyword argument(s)')
    data_path = kwargs.get('data_path')
    if data_path is None:
        data_path = get_data_path()
    try:
        pth = find_data_dir(data_path, *names)
    except DataError, exception:
        pth = [pjoin(this_data_path, *names) 
                for this_data_path in data_path]
        msg = '''%s;
Is it possible you have not installed a data package?
From the names, maybe you need data package "%s"?
If you have the package, have you set the path to the package correctly?
If you don't have the data, you can download it from 
    %s 
and extract it in one of the following path:
    %s''' % (
            exception,
            '-'.join(names),
            NIPY_URL,
            '\n    '.join(pth),
            )
        raise DataError(msg)
    return VersionedDatasource(pth)


class Bomber(object):
    ''' Class to raise an informative error when used '''
    def __init__(self, name, msg):
        self.name = name
        self.msg = msg
        
    def __getattr__(self, attr_name):
        ''' Raise informative error accessing not-found attributes '''
        raise DataError(
            'Trying to access attribute "%s" '
            'of non-existent data "%s"\n\n%s\n' %
            (attr_name, self.name, self.msg))


def _datasource_or_bomber(*names, **options):
    ''' Return a viable datasource or a Bomber

    This is to allow module level creation of datasource objects.  We
    create the objects, so that, if the data exists, the objects are
    valid datasources, and if they don't, they raise an error on access,
    warning about the lack of data.

    The parameters are as for ``make_datasource`` in this module.

    Parameters
    ----------
    *names : sequence of strings
    data_path : sequence of strings or None, optional

    Returns
    -------
    ds : datasource or ``Bomber`` instance
    '''
    try:
        return make_datasource(*names, **options)
    except DataError, exception:
        pass
    name = os.path.sep.join(names)
    return Bomber(name, exception)


# Module level datasource instances for convenience
templates = _datasource_or_bomber('nipy', 'templates')
example_data = _datasource_or_bomber('nipy', 'data')

