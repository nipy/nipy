""" Testing brain segmentation module
"""

import numpy as np

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..brain_segmentation import brain_segmentation
from ..segmentation import Segmentation

from ....io.files import load as load_image
from ....testing import anatfile


def test_bseg():
    # Very crude smoke test
    anat_img = load_image(anatfile)
    ppm_img, label_img = brain_segmentation(anat_img)
    assert_equal(ppm_img.ndim, 4)
    assert_equal(label_img.ndim, 3)


def test_segmentation_3d():
    data = np.random.rand(21, 22, 23)
    S = Segmentation(data, [0.25, 0.75], [1, 1])
    S.run()


def test_segmentation_3d_with_MRF():
    data = np.random.rand(21, 22, 23)
    S = Segmentation(data, [0.25, 0.75], [1, 1], beta=.2)
    S.run()


def test_segmentation_3d_with_mask():
    data = np.random.rand(21, 22, 23)
    mask = np.where(data > .1)
    if mask[0].size < 1:
        return
    S = Segmentation(data, [0.25, 0.75], [1, 1], mask=mask)
    S.run()
