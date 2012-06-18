""" Testing brain segmentation module
"""

import numpy as np

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..segmentation import Segmentation
from ..brain_segmentation import BrainT1Segmentation

from ....io.files import load as load_image
from ....testing import anatfile


def test_bseg():
    # Very crude smoke test
    anat_img = load_image(anatfile)
    mask = anat_img.get_data() > 0
    S = BrainT1Segmentation(anat_img.get_data(), mask=mask, model='4kpv')
    S.run(niters=3, beta=0.2)
    assert_equal(S.ppm.ndim, 4)
    assert_equal(S.label.ndim, 3)


def test_segmentation_3d():
    data = np.random.rand(21, 22, 23)
    S = Segmentation(data, mu=[0.25, 0.75], sigma=[1, 1])
    S.run()


def test_segmentation_3d_with_MRF():
    data = np.random.rand(21, 22, 23)
    S = Segmentation(data, mu=[0.25, 0.75], sigma=[1, 1], beta=.2)
    S.run()


def test_segmentation_3d_with_mask():
    data = np.random.rand(21, 22, 23)
    mask = data > .1
    if mask[0].size < 1:
        return
    S = Segmentation(data, mu=[0.25, 0.75], sigma=[1, 1], mask=mask)
    S.run()
