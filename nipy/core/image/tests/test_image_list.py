# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..image_list import ImageList, iter_axis
from ..image import Image
from ....io.api import load_image
from ....core.reference.coordinate_map import (AxisError, CoordinateSystem,
                                               AffineTransform)

from ....testing import (funcfile, assert_true, assert_equal, assert_raises,
                        assert_almost_equal)


FIMG = load_image(funcfile)


def test_il_init():
    images = list(iter_axis(FIMG, 't'))
    imglst = ImageList(images)
    assert_equal(len(imglst), 20)
    element = imglst[1]
    assert_equal(element.shape, (17, 21, 3))
    assert_equal(element.coordmap, FIMG[...,1].coordmap)
    # Test bad construction
    bad_images = images + [np.zeros((17, 21, 3))]
    assert_raises(ValueError, ImageList, bad_images)
    a = np.arange(10)
    assert_raises(ValueError, ImageList, a)
    # Test empty ImageList
    emplst = ImageList()
    assert_equal(len(emplst.list), 0)


def test_il_from_image():
    exp_shape = (17, 21, 3, 20)
    assert_equal(FIMG.shape, exp_shape)
    # from_image construction
    imglst = ImageList.from_image(FIMG, axis=-1)
    # Test axis must be specified
    assert_raises(ValueError, ImageList.from_image, FIMG)
    assert_raises(ValueError, ImageList.from_image, FIMG, None)
    # check all the axes
    for i in range(4):
        order = range(4)
        order.remove(i)
        order.insert(0,i)
        img_re_i = FIMG.reordered_reference(order).reordered_axes(order)
        imglst_i = ImageList.from_image(FIMG, axis=i)
        assert_equal(imglst_i.list[0].shape, img_re_i.shape[1:])
        # check the affine as well
        assert_almost_equal(imglst_i.list[0].affine, img_re_i.affine[1:,1:])
    # length of image list should match number of frames
    assert_equal(len(imglst), FIMG.shape[3])
    # check the affine
    A = np.identity(4)
    A[:3,:3] = FIMG.affine[:3,:3]
    A[:3,-1] = FIMG.affine[:3,-1]
    assert_almost_equal(imglst.list[0].affine, A)
    # Check other ways of naming axis
    assert_equal(len(ImageList.from_image(FIMG, axis='t')), 20)
    # Input and output axis names work
    new_cmap = AffineTransform(CoordinateSystem('ijkl'),
                               FIMG.coordmap.function_range,
                               FIMG.coordmap.affine)
    fimg2 = Image(FIMG.get_data(), new_cmap)
    assert_equal(len(ImageList.from_image(fimg2, axis='t')), 20)
    assert_equal(len(ImageList.from_image(fimg2, axis='l')), 20)
    assert_raises(AxisError, ImageList.from_image, FIMG, 'q')
    # Check non-dropping case
    ndlist = ImageList.from_image(FIMG, axis='t', dropout=False)
    element = ndlist[1]
    assert_equal(element.coordmap, FIMG[...,1].coordmap)


def test_il_slicing_dicing():
    imglst = ImageList.from_image(FIMG, -1)
    # Slicing an ImageList should return an ImageList
    sublist = imglst[2:5]
    assert_true(isinstance(sublist, ImageList))
    # Except when we're indexing one element
    assert_true(isinstance(imglst[0], Image))
    # Verify array interface
    # test __array__
    assert_true(isinstance(sublist.get_list_data(axis=0), np.ndarray))
    # Test __setitem__
    sublist[2] = sublist[0]
    assert_equal(sublist[0].get_data().mean(),
                 sublist[2].get_data().mean())
    # Test iterator
    for x in sublist:
        assert_true(isinstance(x, Image))
        assert_equal(x.shape, FIMG.shape[:3])

    # Test image_list.get_list_data(axis = an_axis)
    funcim = load_image(funcfile)
    ilist = ImageList.from_image(funcim, axis='t')

    # make sure that we pass an axis
    assert_raises(ValueError, ImageList.get_list_data, ilist, None)
    assert_raises(ValueError, ImageList.get_list_data, ilist)

    # make sure that axis that don't exist makes the function fail
    assert_raises(ValueError, ImageList.get_list_data, ilist, 4)
    assert_raises(ValueError, ImageList.get_list_data, ilist, -5)

    # make sure that axis is put in the right place in the result array
    # image of ilist have dimension (17,21,3), lenght(ilist) = 20.
    data = ilist.get_list_data(axis=0)
    assert_equal(data.shape, (20, 17, 21, 3))

    data = ilist.get_list_data(axis=1)
    assert_equal(data.shape, (17, 20, 21, 3))

    data = ilist.get_list_data(axis=2)
    assert_equal(data.shape, (17, 21, 20, 3))

    data = ilist.get_list_data(axis=3)
    assert_equal(data.shape, (17, 21, 3, 20))

    data = ilist.get_list_data(axis=-1)
    assert_equal(data.shape, (17, 21, 3, 20))

    data = ilist.get_list_data(axis=-2)
    assert_equal(data.shape, (17, 21, 20, 3))
