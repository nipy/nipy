"""
Test the hierarchical clustering bootstrap, with a special attention to 
performance as these procedures are 
"""

from textwrap import dedent
import sys
import timeit

import numpy as np
from numpy.random import rand, permutation

from neuroimaging.neurospin.clustering.bootstrap_hc import _bootstrap_cols, ward_msb, \
            _compare_list_of_arrays


def test_bootstrap_cols_perf(nb_repeat=200):
    """ Timing procedure for _bootstrap_cols.
    """
    # First do the timing test
    setup=dedent("""
    from neuroimaging.neurospin.clustering.bootstrap_hc import _bootstrap_cols
    from numpy.random import rand
    a = rand(100, 100)
    """)
    time = max(timeit.Timer('_bootstrap_cols(a)', setup).repeat(2, nb_repeat))
    print >>sys.__stderr__, "_bootstrap_cols: %f ms per call" % (1000*time/float(nb_repeat))


def test_bootstrap_cols():
    """ Unit test _bootstrap_cols.
    """
    a = rand(100, 100)
    b = _bootstrap_cols(a)
    for col in b.T:
        assert col in a


def profile_ward_msb_perf(nb_repeat=2,verbose=0):
    """ Timing procedure for ward_msb. 
    """
    # First do the timing test
    setup=dedent("""
    from neuroimaging.neurospin.clustering.bootstrap_hc import ward_msb 
    from numpy.random import rand
    a = rand(100, 100)
    """)
    time = max(timeit.Timer('ward_msb(a, niter=10)', 
                                    setup).repeat(2, nb_repeat))
    if verbose:
        print >>sys.__stderr__, "ward_msb: %f ms per call" % \
                                    (1000*time/float(nb_repeat))


def test_compare_list_of_arrays():
    a = rand(100, 100)
    b = list()
    for index in permutation(100):
        b.append(a[index, permutation(100)])
    assert np.all(_compare_list_of_arrays(a, b))


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
    
    # Profile ward_msb
    import hotshot
    from tempfile import mktemp
    profile_file = mktemp()
    prof = hotshot.Profile(profile_file)
    prof.runcall(profile_ward_msb_perf)
    prof.close()
    stats = hotshot.stats.load(profile_file)
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(40)
    import os
    os.unlink(profile_file)

