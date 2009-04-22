#!/usr/bin/env python

from neuroimaging.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_raises
import numpy as np

from nipy.neurospin.register.realign4d import Image4d


"""
r = Image4d(np.zeros([64,64,20]), np.diag([3.,3.,3.,1.]), tr=2., slice_order='ascending', interleaved=False)

r.to_time()
"""


if __name__ == "__main__":
	import nose
	nose.run(argv=['', __file__])
