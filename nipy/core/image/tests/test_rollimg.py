# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This test basically just plays around with image.rollimg.

It has three examples

* image_reduce:  this takes an Image having, say, an axis 't' and returns another
  Image having reduced over 't'

* need_specific_axis_reduce: this takes an Image and a specific
  axis name, like 't' and produces an Image reduced over 't'. raises an
  exception if Image has no axis 't'

* image_call: this takes an Image having, say, an axis 't'
  and does something along this axis -- like fits a regression model? and
  outputs a new Image with the 't' axis replaced by 'new'

* image_modify_copy: this takes an Image and an axis specification,
  such as 'x+LR', 'l', or 2, modifies a copy of the data by iterating over this
  axis, and returns an Image with the same axes


Note
----

In these loaded Images, 't' is both an axis name and a world coordinate name so
it is not ambiguous to say 't' axis. It is slightly ambiguous to say 'x+LR' axis
if the axisnames are ['slice', 'frequency', 'phase'] but image.rollimg
identifies 'x+LR' == 'slice' == 0.
"""

import numpy as np

from ..image import (Image, rollimg, synchronized_order)
from ...reference.coordinate_map import (AffineTransform as AT, drop_io_dim,
                                         AxisError)
from ...reference.coordinate_system import CoordinateSystem as CS
from ...reference.spaces import mni_csm
from ...image.image_spaces import xyz_affine

from nose.tools import (assert_raises, assert_equal)

from numpy.testing import assert_almost_equal, assert_array_equal


MNI3 = mni_csm(3)
MNI4 = mni_csm(4)


def image_reduce(img, reduce_op, axis='t'):
    """
    Take an Image, perform some reduce operation on it, over
    a specified axis, and return a new Image.

    For the sake of testing things out, we will assume that
    the operation reduces over the first axis only.

    Parameters
    ----------
    image : Image
    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)
    axis : str or int
        Specification of axis of Image

    Returns
    -------
    newim : Image, missing axis
    """
    img = rollimg(img, axis)
    axis_name = img.axes.coord_names[0]
    output_axes = list(img.axes.coord_names)
    output_axes.remove(axis_name)
    newdata = reduce_op(img.get_data())
    return Image(newdata, drop_io_dim(img.coordmap, axis))


def need_specific_axis_reduce(img, reduce_op):
    """
    Take an Image, perform some reduce operation on it, over the axis named
    'specific', and return a new Image.

    For the sake of testing things out, we will assume that the operation
    reduces over the first axis only.

    Parameters
    ----------
    img : Image
    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)

    Returns
    -------
    newim : Image, missing axis
    """
    return image_reduce(img, reduce_op, 'specific')


def image_call(img, function, inaxis='t', outaxis='new'):
    """
    Take an Image, perform some operation on it, over a specified axis, and
    return a new Image.

    For the sake of testing things out, we will assume that the operation can
    only operate on the first axis of the array.

    Parameters
    ----------
    img : Image
    function : callable
        An operation that does something over the first axis,
        such as lambda x: x[::2]
    inaxis : str or int
        Specification of axis of Image
    outaxis : str
        Name of new axis in new Image

    Returns
    -------
    newim : Image
        with axis `inaxis` replaced with `outaxis`
    """
    rolled_img = rollimg(img, inaxis)
    inaxis = rolled_img.axes.coord_names[0] # now it's a string
    newdata = function(rolled_img.get_data())
    new_coordmap = rolled_img.coordmap.renamed_domain({inaxis: outaxis})
    new_image = Image(newdata, new_coordmap)
    # we have to roll the axis back
    axis_index = img.axes.index(inaxis)
    return rollimg(new_image, 0, axis_index)


def image_modify(img, modify, axis='y+PA'):
    """
    Take an Image, perform some operation on it, over a specified axis, and
    return a new Image.

    For this operation, we are allowed to iterate over spatial axes.

    For the sake of testing things out, we will assume that the operation modify
    can only operate by iterating over the first axis of an array.

    Parameters
    ----------
    img : Image
    modify : callable
        An operation that modifies an array.  Something like::

            def f(x):
                x[:] = x.mean()

    axis : str or int
        Specification of axis of Image

    Returns
    -------
    newim : Image
        with a modified copy of img._data.
    """
    rolled_img = rollimg(img, axis)
    data = rolled_img.get_data().copy()
    for d in data:
        modify(d)
    import copy
    new_image = Image(data, copy.copy(rolled_img.coordmap))
    # Now, we have to put the data back to same order as img
    return synchronized_order(new_image, img)


def test_reduce():
    shape = (3, 5, 7, 9)
    x = np.random.standard_normal(shape)
    im = Image(x, AT(CS('ijkq'), MNI4, np.diag([3, 4, 5, 6, 1])))
    newim = image_reduce(im, lambda x: x.sum(0), 'q')
    assert_array_equal(xyz_affine(im), xyz_affine(newim))
    assert_equal(newim.axes.coord_names, tuple('ijk'))
    assert_equal(newim.shape, (3, 5, 7))
    assert_almost_equal(newim.get_data(), x.sum(3))
    im_nd = Image(x, AT(CS('ijkq'), MNI4, np.array(
        [[0, 1, 2, 0, 10],
         [3, 4, 5, 0, 11],
         [6, 7, 8, 0, 12],
         [0, 0, 0, 9, 13],
         [0, 0, 0, 0, 1]])))
    for i, o, n in zip('ijk', MNI3.coord_names, range(3)):
        for axis_id in (i, o, n):
            # Non-diagonal reduce raise an error
            assert_raises(AxisError, image_reduce, im_nd,
                          lambda x: x.sum(0), axis_id)
            # Diagonal reduces are OK
            newim = image_reduce(im, lambda x: x.sum(0), axis_id)


def test_specific_reduce():
    shape = (3, 5, 7, 9)
    x = np.random.standard_normal(shape)
    im = Image(x, AT(CS('ijkq'), MNI4, np.diag([3, 4, 5, 6, 1])))
    # we have to rename the axis before we can call the function
    # need_specific_axis_reduce on it
    assert_raises(AxisError, need_specific_axis_reduce, im, lambda x: x.sum(0))
    im = im.renamed_axes(q='specific')
    newim = need_specific_axis_reduce(im, lambda x: x.sum(0))
    assert_array_equal(xyz_affine(im), xyz_affine(newim))
    assert_equal(newim.axes.coord_names, tuple('ijk'))
    assert_equal(newim.shape, (3, 5, 7))
    assert_almost_equal(newim.get_data(), x.sum(3))


def test_call():
    shape = (3, 5, 7, 12)
    x = np.random.standard_normal(shape)
    affine = np.eye(5)
    affine[:3, :3] = np.random.standard_normal((3, 3))
    affine[:4, 4] = np.random.standard_normal((4,))
    im = Image(x, AT(CS('ijkq'), MNI4, affine))
    newim = image_call(im, lambda x: x[::2], 'q', 'out')
    assert_array_equal(xyz_affine(im), xyz_affine(newim))
    assert_equal(newim.axes.coord_names, tuple('ijk') + ('out',))
    assert_equal(newim.shape, (3, 5, 7, 6))
    assert_almost_equal(newim.get_data(), x[:,:,:,::2])


def test_modify():
    shape = (3, 5, 7, 12)
    x = np.random.standard_normal(shape)
    affine = np.eye(5)
    affine[:3, :3] = np.random.standard_normal((3, 3))
    affine[:4, 4] = np.random.standard_normal((4,))
    im = Image(x, AT(CS('ijkq'), MNI4, affine))

    def nullmodify(d):
        pass

    def meanmodify(d):
        d[:] = d.mean()

    for i, o, n in zip('ijkq', MNI3.coord_names + ('q',), range(4)):
        for a in i, o, n:
            nullim = image_modify(im, nullmodify, a)
            meanim = image_modify(im, meanmodify, a)
            assert_array_equal(nullim.get_data(), im.get_data())
            assert_array_equal(xyz_affine(im), xyz_affine(nullim))
            assert_equal(nullim.axes, im.axes)
            # yield assert_equal, nullim, im
            assert_array_equal(xyz_affine(im), xyz_affine(meanim))
            assert_equal(meanim.axes, im.axes)
        # Make sure that meanmodify works as expected
        d = im.get_data()
        d = np.rollaxis(d, n)
        meand = meanim.get_data()
        meand = np.rollaxis(meand, n)
        for i in range(d.shape[0]):
            assert_almost_equal(meand[i], d[i].mean())
