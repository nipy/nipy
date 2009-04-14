"""
Test the hierarchical clustering bootstrap, with a special attention to 
performance as these procedures are 
"""


import numpy as np
from numpy.random import rand, permutation

from nipy.neurospin.clustering.bootstrap_hc import _bootstrap_cols, \
     _compare_list_of_arrays

def test_bootstrap_cols():
    """ Unit test _bootstrap_cols.
    """
    a = rand(100, 100)
    b = _bootstrap_cols(a)
    for col in b.T:
        assert col in a


def test_compare_list_of_arrays():
    a = rand(100, 100)
    b = list()
    for index in permutation(100):
        b.append(a[index, permutation(100)])
    assert np.all(_compare_list_of_arrays(a, b))


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
    

