''' Tests for data module '''

import os
import tempfile

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from nipy.utils.data import get_data_path, make_repositories

DATA_PATH = None
DATA_KEY = 'NIPY_DATA_PATH'

def setup_module():
    try:
        DATA_PATH = os.environ[DATA_KEY]
    except KeyError:
        pass

def teardown_module():
    if DATA_PATH is None:
        try:
            del os.environ[DATA_KEY]
        except KeyError:
            pass
    else:
        os.environ[DATA_KEY] = DATA_PATH
        
def test_data_path():
    # First, an environment variable
    os.environ[DATA_KEY] = '/a/path'
    yield assert_equal, get_data_path(), '/a/path'
    

def test_make_repositories():
    here, fname = os.path.split(__file__)
    under_here, subhere = os.path.split(here)
    repos = make_repositories(None, [subhere])
    yield assert_equal, repos, [None]
    repos = make_repositories(under_here, [subhere])
    yield assert_equal, len(repos), 1
    yield assert_equal, repos[0].full_path(fname), __file__
