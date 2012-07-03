""" Testing brain segmentation module
"""

import numpy as np

from nose.tools import assert_equal

from ..segmentation import Segmentation
from ..brain_segmentation import BrainT1Segmentation

from ....io.files import load as load_image
from ....testing import anatfile


def test_brain_seg():
    anat_img = load_image(anatfile)
    mask = anat_img.get_data() > 0
    S = BrainT1Segmentation(anat_img.get_data(), mask=mask, model='3k')
    S.run(niters=3, beta=0.)
    assert_equal(S.ppm.ndim, 4)
    assert_equal(S.ppm.shape[3], 3)
    assert_equal(S.label.ndim, 3)
    S = BrainT1Segmentation(anat_img.get_data(), mask=mask,
                            model=np.array([[1., 0., 0.],
                                            [1., 0., 0.],
                                            [0., 1., 0.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]))
    S.run(niters=3, beta=0.5, ngb_size=6, convert=False)
    assert_equal(S.ppm.ndim, 4)
    assert_equal(S.ppm.shape[3], 5)
    assert_equal(S.label.ndim, 3)
    S.run(niters=3, beta=0.5, ngb_size=26, convert=False)
    assert_equal(S.ppm.ndim, 4)
    assert_equal(S.ppm.shape[3], 5)
    assert_equal(S.label.ndim, 3)


def _test_segmentation(S, nchannels=1):
    assert_equal(S.nchannels, nchannels)
    S.run(niters=5)
    label = S.map()
    assert_equal(label.ndim, 3)
    assert_equal(label.dtype, 'uint8')
    assert isinstance(S.free_energy(), float)


def test_segmentation_3d():
    data = np.random.rand(21, 22, 23)
    _test_segmentation(Segmentation(data, mu=[0.25, 0.75], sigma=[1, 1]))


def test_segmentation_3d_with_MRF():
    data = np.random.rand(21, 22, 23)
    _test_segmentation(Segmentation(data, mu=[0.25, 0.75],
                                    sigma=[1, 1], beta=.2))


def test_segmentation_3d_with_mask():
    data = np.random.rand(21, 22, 23)
    mask = data > .1
    if mask[0].size < 1:
        return
    _test_segmentation(Segmentation(data, mu=[0.25, 0.75],
                                    sigma=[1, 1], mask=mask))


def test_segmentation_3d_multichannel():
    data = np.random.rand(21, 22, 23, 2)
    mask = data[..., 0] > .1
    if mask[0].size < 1:
        return
    _test_segmentation(Segmentation(data,
                                    mu=[[0.25, 0.25], [0.75, 0.75]],
                                    sigma=[np.eye(2), np.eye(2)],
                                    mask=mask),
                       nchannels=2)
