''' Tests for data module '''

import os

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from nipy.utils.data import get_data_path

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
    
