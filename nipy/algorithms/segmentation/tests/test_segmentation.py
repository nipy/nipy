""" Testing brain segmentation module
"""

import numpy as np

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..brain_segmentation import brain_segmentation


from ....io.files import load as load_image
from ....testing import anatfile


def test_bseg():
    # Very crude smoke test
    anat_img = load_image(anatfile)
    ppm_img, label_img = brain_segmentation(anat_img)
    assert_equal(ppm_img.ndim, 4)
    assert_equal(label_img.ndim, 3)
