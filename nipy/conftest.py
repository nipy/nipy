# Control testing
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
