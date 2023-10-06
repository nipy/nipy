# Control testing
import os
import tempfile
from pathlib import Path

import numpy
import pytest


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    numpy.set_printoptions(legacy="1.13")
    doctest_namespace["np"] = numpy


@pytest.fixture(scope='session')
def mpl_imports():
    """ Force matplotlib to use agg backend for tests
    """
    try:
        import matplotlib as mpl
    except ImportError:
        pass
    else:
        mpl.use('agg')


@pytest.fixture
def in_tmp_path():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield Path(newpath)
        os.chdir(old_cwd)
