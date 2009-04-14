"""
Test surrogate data generation.
"""

import numpy as np

from nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset import \
    make_surrogate_array

def test_surrogate_array():
    """ Check that with no noise, the surrogate activation correspond to
        the ones that we specify.
    """
    # We can't use random positions, as the positions have to be
    # far-enough not to overlap.
    pos   = np.array([[ 2, 10],
                      [10,  4],
                      [80, 30],
                      [40, 60],
                      [90, 70]])
    ampli = np.random.random(5)
    data = make_surrogate_array(nbsubj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, dimx=100,
                                dimy=100, ampli=ampli).squeeze()
    x, y = pos.T
    np.testing.assert_array_equal(data[x, y], ampli)

