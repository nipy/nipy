# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import pytest

from ....core.reference.coordinate_map import (
    AffineTransform,
    AxisError,
    CoordinateSystem,
)
from ....io.api import load_image
from ....testing import (
    assert_almost_equal,
    funcfile,
)
from ..image import Image
from ..image_list import ImageList, iter_axis

FIMG = load_image(funcfile)


def test_il_init():
    images = list(iter_axis(FIMG, 't'))
    imglst = ImageList(images)
    assert len(imglst) == 20
    element = imglst[1]
    assert element.shape == (17, 21, 3)
    assert element.coordmap == FIMG[...,1].coordmap
    # Test bad construction
    bad_images = images + [np.zeros((17, 21, 3))]
    pytest.raises(ValueError, ImageList, bad_images)
    a = np.arange(10)
    pytest.raises(ValueError, ImageList, a)
    # Test empty ImageList
    emplst = ImageList()
    assert len(emplst.list) == 0


def test_il_from_image():
    exp_shape = (17, 21, 3, 20)
    assert FIMG.shape == exp_shape
    # from_image construction
    imglst = ImageList.from_image(FIMG, axis=-1)
    # Test axis must be specified
    pytest.raises(ValueError, ImageList.from_image, FIMG)
    pytest.raises(ValueError, ImageList.from_image, FIMG, None)
    # check all the axes
    for i in range(4):
        order = list(range(4))
        order.remove(i)
        order.insert(0,i)
        img_re_i = FIMG.reordered_reference(order).reordered_axes(order)
        imglst_i = ImageList.from_image(FIMG, axis=i)
        assert imglst_i.list[0].shape == img_re_i.shape[1:]
        # check the affine as well
        assert_almost_equal(imglst_i.list[0].affine, img_re_i.affine[1:,1:])
    # length of image list should match number of frames
    assert len(imglst) == FIMG.shape[3]
    # check the affine
    A = np.identity(4)
    A[:3,:3] = FIMG.affine[:3,:3]
    A[:3,-1] = FIMG.affine[:3,-1]
    assert_almost_equal(imglst.list[0].affine, A)
    # Check other ways of naming axis
    assert len(ImageList.from_image(FIMG, axis='t')) == 20
    # Input and output axis names work
    new_cmap = AffineTransform(CoordinateSystem('ijkl'),
                               FIMG.coordmap.function_range,
                               FIMG.coordmap.affine)
    fimg2 = Image(FIMG.get_fdata(), new_cmap)
    assert len(ImageList.from_image(fimg2, axis='t')) == 20
    assert len(ImageList.from_image(fimg2, axis='l')) == 20
    pytest.raises(AxisError, ImageList.from_image, FIMG, 'q')
    # Check non-dropping case
    ndlist = ImageList.from_image(FIMG, axis='t', dropout=False)
    element = ndlist[1]
    assert element.coordmap == FIMG[...,1].coordmap


def test_il_slicing_dicing():
    imglst = ImageList.from_image(FIMG, -1)
    # Slicing an ImageList should return an ImageList
    sublist = imglst[2:5]
    assert isinstance(sublist, ImageList)
    # Except when we're indexing one element
    assert isinstance(imglst[0], Image)
    # Verify array interface
    # test __array__
    assert isinstance(sublist.get_list_data(axis=0), np.ndarray)
    # Test __setitem__
    sublist[2] = sublist[0]
    assert (sublist[0].get_fdata().mean() ==
                 sublist[2].get_fdata().mean())
    # Test iterator
    for x in sublist:
        assert isinstance(x, Image)
        assert x.shape == FIMG.shape[:3]

    # Test image_list.get_list_data(axis = an_axis)
    funcim = load_image(funcfile)
    ilist = ImageList.from_image(funcim, axis='t')

    # make sure that we pass an axis
    pytest.raises(ValueError, ImageList.get_list_data, ilist, None)
    pytest.raises(ValueError, ImageList.get_list_data, ilist)

    # make sure that axis that don't exist makes the function fail
    pytest.raises(ValueError, ImageList.get_list_data, ilist, 4)
    pytest.raises(ValueError, ImageList.get_list_data, ilist, -5)

    # make sure that axis is put in the right place in the result array
    # image of ilist have dimension (17,21,3), length(ilist) = 20.
    data = ilist.get_list_data(axis=0)
    assert data.shape == (20, 17, 21, 3)

    data = ilist.get_list_data(axis=1)
    assert data.shape == (17, 20, 21, 3)

    data = ilist.get_list_data(axis=2)
    assert data.shape == (17, 21, 20, 3)

    data = ilist.get_list_data(axis=3)
    assert data.shape == (17, 21, 3, 20)

    data = ilist.get_list_data(axis=-1)
    assert data.shape == (17, 21, 3, 20)

    data = ilist.get_list_data(axis=-2)
    assert data.shape == (17, 21, 20, 3)
