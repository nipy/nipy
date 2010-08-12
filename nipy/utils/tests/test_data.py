# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Tests for data module '''
from __future__ import with_statement
import os
from os.path import join as pjoin
from os import environ as env
import sys
import tempfile

from nipy.utils.data import get_data_path, find_data_dir, \
    DataError, _cfg_value, make_datasource, \
    Datasource, VersionedDatasource, Bomber, \
    _datasource_or_bomber

from nipy.utils.tmpdirs import TemporaryDirectory

import nipy.utils.data as nud

from nose import with_setup
from nose.tools import assert_equal, \
    assert_raises, raises


GIVEN_ENV = {}
DATA_KEY = 'NIPY_DATA_PATH'
USER_KEY = 'NIPY_USER_DIR'


def setup_environment():
    """Setup test environment for some functions that are tested 
    in this module. In particular this functions stores attributes
    and other things that we need to stub in some test functions.
    This needs to be done on a function level and not module level because 
    each testfunction needs a pristine environment.
    """
    global GIVEN_ENV
    GIVEN_ENV['env'] = env.copy()
    GIVEN_ENV['sys_dir_func'] = nud.get_nipy_system_dir
    GIVEN_ENV['path_func'] = nud.get_data_path

def teardown_environment():
    """Restore things that were remebered by the setup_environment function
    """
    orig_env = GIVEN_ENV['env']
    for key in env.keys():
        if key not in orig_env:
            del env[key]
    env.update(orig_env)
    nud.get_nipy_system_dir = GIVEN_ENV['sys_dir_func']
    nud.get_data_path = GIVEN_ENV['path_func']
    
# decorator to use setup, teardown environment
with_environment = with_setup(setup_environment, teardown_environment)


def test_datasource():
    # Tests for DataSource
    pth = pjoin('some', 'path')
    ds = Datasource(pth)
    yield assert_equal, ds.get_filename('unlikeley'), pjoin(pth, 'unlikeley')
    yield (assert_equal, ds.get_filename('un','like','ley'),
           pjoin(pth, 'un','like','ley'))


def test_versioned():
    with TemporaryDirectory() as tmpdir:
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        tmpfile = pjoin(tmpdir, 'config.ini')
        # ini file, but wrong section
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[SOMESECTION]\n')
            fobj.write('version = 0.1\n')
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        # ini file, but right section, wrong key
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('somekey = 0.1\n')
        yield (assert_raises,
               DataError,
               VersionedDatasource,
               tmpdir)
        # ini file, right section and key
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1\n')
        vds = VersionedDatasource(tmpdir)
        yield assert_equal, vds.version, '0.1'
        yield assert_equal, vds.version_no, 0.1
        yield assert_equal, vds.major_version, 0
        yield assert_equal, vds.minor_version, 1        
        yield assert_equal, vds.get_filename('config.ini'), tmpfile
        # ini file, right section and key, funny value
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1.2.dev\n')
        vds = VersionedDatasource(tmpdir)
        yield assert_equal, vds.version, '0.1.2.dev'
        yield assert_equal, vds.version_no, 0.1
        yield assert_equal, vds.major_version, 0
        yield assert_equal, vds.minor_version, 1        


def test__cfg_value():
    # no file, return ''
    yield assert_equal, _cfg_value('/implausible_file'), ''
    # try files
    try:
        fd, tmpfile = tempfile.mkstemp()
        fobj = os.fdopen(fd, 'wt')
        # wrong section, right key
        fobj.write('[strange section]\n')
        fobj.write('path = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), ''
        # right section, wrong key
        fobj.write('[DATA]\n')
        fobj.write('funnykey = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), ''
        # right section, right key
        fobj.write('path = /some/path\n')
        fobj.flush()
        yield assert_equal, _cfg_value(tmpfile), '/some/path'
        fobj.close()
    finally:
        try:
            os.unlink(tmpfile)
        except:
            pass


@with_environment
def test_data_path():
    # wipe out any sources of data paths
    if DATA_KEY in env:
        del env[DATA_KEY]
    if USER_KEY in env:
        del os.environ[USER_KEY]
    nud.get_nipy_system_dir = lambda : ''
    # now we should only have the default
    old_pth = get_data_path()
    # We should have only sys.prefix and, iff sys.prefix == /usr,
    # '/usr/local'.  This last to is deal with Debian patching to
    # distutils.
    def_dirs = [pjoin(sys.prefix, 'share', 'nipy')]
    if sys.prefix == '/usr':
        def_dirs.append(pjoin('/usr/local', 'share', 'nipy'))
    home_nipy = pjoin(os.path.expanduser('~'), '.nipy')
    yield assert_equal, old_pth, def_dirs + [home_nipy]
    # then we'll try adding some of our own
    tst_pth = '/a/path' + os.path.pathsep + '/b/ path'
    tst_list = ['/a/path', '/b/ path']
    # First, an environment variable
    os.environ[DATA_KEY] = tst_list[0]
    yield assert_equal, get_data_path(), tst_list[:1] + old_pth
    os.environ[DATA_KEY] = tst_pth
    yield assert_equal, get_data_path(), tst_list + old_pth
    del os.environ[DATA_KEY]
    # Next, make a fake user directory, and put a file in there
    with TemporaryDirectory() as tmpdir:
        tmpfile = pjoin(tmpdir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s' % tst_pth)
        os.environ[USER_KEY] = tmpdir
        yield assert_equal, get_data_path(), tst_list + def_dirs + [tmpdir]
    del os.environ[USER_KEY]
    yield assert_equal, get_data_path(), old_pth
    # with some trepidation, the system config files
    with TemporaryDirectory() as tmpdir:
        nud.get_nipy_system_dir = lambda : tmpdir
        tmpfile = pjoin(tmpdir, 'an_example.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s\n' % tst_pth)
        tmpfile = pjoin(tmpdir, 'another_example.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DATA]\n')
            fobj.write('path = %s\n' % '/path/two')
        yield (assert_equal, get_data_path(),
               tst_list + ['/path/two'] + old_pth)
    

def test_find_data_dir():
    here, fname = os.path.split(__file__)
    # here == '<rootpath>/nipy/utils/tests'
    under_here, subhere = os.path.split(here)
    # under_here == '<rootpath>/nipy/utils'
    # subhere = 'tests'
    # fails with non-existant path
    yield (assert_raises,
           DataError,
           find_data_dir,
           [here],
           'implausible',
           'directory')
    # fails with file, when directory expected
    yield (assert_raises,
           DataError,
           find_data_dir,
           [here],
           fname)
    # passes with directory that exists
    dd = find_data_dir([under_here], subhere)
    yield assert_equal, dd, here
    # and when one path in path list does not work
    dud_dir = pjoin(under_here, 'implausible')
    dd = find_data_dir([dud_dir, under_here], subhere)
    yield assert_equal, dd, here


@with_environment
def test_make_datasource():
    with TemporaryDirectory() as tmpdir:
        nud.get_data_path = lambda : [tmpdir]
        yield (assert_raises,
           DataError,
           make_datasource,
           'pkg')
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        yield (assert_raises,
           DataError,
           make_datasource,
           'pkg')
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.1\n')
        ds = make_datasource('pkg', data_path=[tmpdir])
        yield assert_equal, ds.version, '0.1'


@raises(DataError)
def test_bomber():
    b = Bomber('bomber example', 'a message')
    res = b.any_attribute


@with_environment
def test__datasource_or_bomber():
    with TemporaryDirectory() as tmpdir:
        nud.get_data_path = lambda : [tmpdir]
        ds = _datasource_or_bomber('pkg')
        yield (assert_raises,
               DataError,
               getattr,
               ds,
               'get_filename')
        pkg_dir = pjoin(tmpdir, 'pkg')
        os.mkdir(pkg_dir)
        tmpfile = pjoin(pkg_dir, 'config.ini')
        with open(tmpfile, 'wt') as fobj:
            fobj.write('[DEFAULT]\n')
            fobj.write('version = 0.2\n')
        ds = _datasource_or_bomber('pkg')
        fn = ds.get_filename('some_file.txt')
        # check that versioning works
        ds = _datasource_or_bomber('pkg', version='0.2') # OK
        fn = ds.get_filename('some_file.txt')
        ds = _datasource_or_bomber('pkg', version='0.3') # not OK
        yield (assert_raises,
               DataError,
               getattr,
               ds,
               'get_filename')
        
