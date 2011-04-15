# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nose

import numpy as np

from ..ortho_slicer import demo_ortho_slicer, _edge_detect, \
        _fast_abs_percentile

from ..anat_cache import find_mni_template

try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    do_test = True
except ImportError:
    do_test = False

################################################################################
# First some smoke testing for graphics-related code
if do_test:
    def test_demo_ortho_slicer():
        # This is only a smoke test
        # conditioned on presence of MNI templated
        if not find_mni_template():
            raise nose.SkipTest("MNI Template is absent for the smoke test")
        mp.use('svg', warn=False)
        import pylab as pl
        pl.switch_backend('svg')
        demo_ortho_slicer()



################################################################################
# Actual unit tests
def test_fast_abs_percentile():
    data = np.arange(1, 100)
    for p in range(10, 100, 10):
        yield nose.tools.assert_equal, _fast_abs_percentile(data, p-1), p


def test_edge_detect():
    img = np.zeros((10, 10))
    img[:5] = 1
    _, edge_mask = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
