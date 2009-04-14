import os
import hotshot
import timeit
import sys
from tempfile import mktemp
from textwrap import dedent

from nipy.neurospin.clustering.bootstrap_hc import ward_msb

def bench_bootstrap_cols_perf(nb_repeat=200):
    """ Timing procedure for _bootstrap_cols.
    """
    # First do the timing test
    setup=dedent("""
    from nipy.neurospin.clustering.bootstrap_hc import _bootstrap_cols
    from numpy.random import rand
    a = rand(100, 100)
    """)
    time = max(timeit.Timer('_bootstrap_cols(a)', setup).repeat(2, nb_repeat))
    print >>sys.__stderr__, "_bootstrap_cols: %f ms per call" % \
        (1000*time/float(nb_repeat))


def profile_ward_msb_perf(nb_repeat=2,verbose=0):
    """ Timing procedure for ward_msb. 
    """
    # First do the timing test
    setup=dedent("""
    from nipy.neurospin.clustering.bootstrap_hc import ward_msb 
    from numpy.random import rand
    a = rand(100, 100)
    """)
    time = max(timeit.Timer('ward_msb(a, niter=10)', 
                                    setup).repeat(2, nb_repeat))
    if verbose:
        print >>sys.__stderr__, "ward_msb: %f ms per call" % \
                                    (1000*time/float(nb_repeat))
"""
# This hangs when run as a benchmark.  Don't know why....
def bench_ward_msb():
    # Profile ward_msb
    profile_file = mktemp()
    prof = hotshot.Profile(profile_file)
    prof.runcall(profile_ward_msb_perf)
    prof.close()
    stats = hotshot.stats.load(profile_file)
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(40)
    os.unlink(profile_file)
"""
