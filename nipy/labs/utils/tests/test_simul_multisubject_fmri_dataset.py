# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test surrogate data generation.
"""

import numpy as np

from ..simul_multisubject_fmri_dataset import \
    surrogate_2d_dataset, surrogate_3d_dataset 

def test_surrogate_array():
    """ Check that with no noise, the surrogate activation correspond to
        the ones that we specify. 2D version
    """
    # We can't use random positions, as the positions have to be
    # far-enough not to overlap.
    pos   = np.array([[ 2, 10],
                      [10,  4],
                      [80, 30],
                      [40, 60],
                      [90, 70]])
    ampli = np.random.random(5)
    data = surrogate_2d_dataset(nbsubj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, dimx=100,
                                dimy=100, ampli=ampli).squeeze()
    x, y = pos.T
    np.testing.assert_array_equal(data[x, y], ampli)

def test_surrogate_array_3d():
    """ Check that with no noise, the surrogate activation correspond to
        the ones that we specify. 3D version
    """
    # We can't use random positions, as the positions have to be
    # far-enough not to overlap.
    pos   = np.array([[ 2, 10, 2],
                      [10,  4, 4],
                      [18, 13, 18],
                      [13, 18, 25],
                      [25, 18, 18]])
    ampli = np.random.random(5)
    data = surrogate_3d_dataset(nbsubj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, shape=(32,32,32),
                                ampli=ampli).squeeze()
    x, y, z = pos.T
    np.testing.assert_array_equal(data[x, y, z], ampli)

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
