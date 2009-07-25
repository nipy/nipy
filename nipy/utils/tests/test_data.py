''' Tests for data module '''

import os
import tempfile

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from nipy.utils.data import get_data_path, find_data_dir

DATA_PATH = None
DATA_KEY = 'NIPY_DATA_PATH'

def setup_module():
    global DATA_PATH, DATA_KEY
    try:
        DATA_PATH = os.environ[DATA_KEY]
    except KeyError:
        pass

def teardown_module():
    global DATA_PATH, DATA_KEY
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
    yield assert_equal, get_data_path(), ['/a/path']
    os.environ[DATA_KEY] = '/a/path' + os.path.pathsep + '/b/ path'
    yield assert_equal, get_data_path(), ['/a/path', '/b/ path']
    

def test_find_data_dir():
    here, fname = os.path.split(__file__)
    under_here, subhere = os.path.split(here)
    dd = find_data_dir([under_here], subhere)
    yield assert_equal, dd, here
