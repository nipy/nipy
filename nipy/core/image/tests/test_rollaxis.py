# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This test basically just plays around with image.rollaxis.

It has three examples

image_reduce:  this takes an XYZImage having, say, an axis 't'
               and returns another XYZImage
               having reduced over 't'

image_call: this takes an XYZImage having, say, an axis 't'
            and does something along this axis -- like fits
            a regression model? and outputs a new XYZImage
            with the 't' axis replaced by 'new'

image_modify_copy: this takes an XYZImage and an axis specification,
                   such as 'x+LR', 'l', or 2, modifies a copy of the data
                   by iterating over this axis, and returns an
                   XYZImage with the same axes

need_specific_axis_reduce: this takes an XYZImage and a specific 
                           axis name, like 't' and produces an XYZImage
                           reduced over 't'. raises an exception if
                           XYZImage has no axis 't'

Note
----

Because these are XYZImages, 't' is both an axis name and a world coordinate
name so it is not ambiguous to say 't' axis. It is slightly 
ambiguous to say 'x+LR' axis if the axisnames are ['slice', 'frequency', 'phase']
but image.rollaxis identifies 'x+LR' == 'slice' == 0.

"""

import numpy as np

from nipy.testing import *

from nipy.core.image.image import Image, rollaxis as image_rollaxis, \
    synchronized_order
from nipy.core.image.xyz_image import XYZImage

def image_reduce(xyz_image, reduce_op, axis='t'):
    """
    Take an XYZImage, perform some reduce operation on it, over
    a specified axis, and return a new XYZImage.

    For the sake of testing things out, we will assume that
    the operation reduces over the first axis only.

    Parameters
    ----------

    xyz_image : XYZImage

    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)

    axis : str or int
        Specification of axis of XYZImage

    Returns
    -------

    newim : XYZImage, missing axis 

    """

    if axis in xyz_image.reference.coord_names + \
            xyz_image.axes.coord_names[:3] + tuple(range(3)):
        raise ValueError('cannot reduce over a spatial axis' +
                         'or we will not be able to output XYZImages')

    image = xyz_image.to_image()
    image = image_rollaxis(image, axis)

    axis_name = image.axes.coord_names[0]
    output_axes = list(xyz_image.axes.coord_names)
    output_axes.remove(axis_name)

    newdata = reduce_op(image.get_data())
    return XYZImage(newdata, xyz_image.affine, output_axes)

def need_specific_axis_reduce(xyz_image, reduce_op):
    """
    Take an XYZImage, perform some reduce operation on it, over
    the axis 'specific', and returns a new XYZImage.

    For the sake of testing things out, we will assume that
    the operation reduces over the first axis only.

    Parameters
    ----------

    xyz_image : XYZImage

    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)

    Returns
    -------

    newim : XYZImage, missing axis 

    """

    image = xyz_image.to_image()
    image = image_rollaxis(image, 'specific')

    axis_name = image.axes.coord_names[0]
    output_axes = list(xyz_image.axes.coord_names)
    output_axes.remove(axis_name)

    newdata = reduce_op(image.get_data())
    return XYZImage(newdata, xyz_image.affine, output_axes)


def image_call(xyz_image, function, inaxis='t', outaxis='new'):
    """
    Take an XYZImage, perform some operation on it, over
    a specified axis, and return a new XYZImage.

    For the sake of testing things out, we will assume that
    the operation can only operate on the first
    axis of the array.

    Parameters
    ----------

    xyz_image : XYZImage

    function : callable
        An operation that does something over the first axis,
        such as lambda x: x[::2]

    inaxis : str or int
        Specification of axis of XYZImage

    outaxis : str
        Name of new axis in new XYZImage

    Returns
    -------

    newim : XYZImage with axis inaxis replaced with outaxis

    """

    if inaxis in xyz_image.reference.coord_names + \
            xyz_image.axes.coord_names[:3] + tuple(range(3)):
        raise ValueError('cannot reduce over a spatial axis' +
                         'or we will not be able to output XYZImages')

    image = xyz_image.to_image()
    image = image_rollaxis(image, inaxis)
    inaxis = image.axes.coord_names[0] # now it's a string

    newdata = function(image.get_data())
    newcoordmap = image.coordmap.renamed_domain({inaxis:outaxis})
    newimage = Image(newdata, newcoordmap)

    # we have to roll the axis back

    axis_index = xyz_image.axes.index(inaxis)
    newimage = image_rollaxis(newimage, axis_index, inverse=True)

    return XYZImage(newimage.get_data(), xyz_image.affine, 
                    newimage.axes.coord_names)

def image_modify(xyz_image, modify, axis='y+PA'):
    """
    Take an XYZImage, perform some operation on it, over
    a specified axis, and return a new XYZImage.

    For this operation, we are allowed to iterate over
    spatial axes.

    For the sake of testing things out, we will assume that
    the operation modify can only operate by iterating
    over the first axis of an array.

    Parameters
    ----------

    xyz_image : XYZImage

    modify : callable
        An operation that does modifies an array.
        Something like 

        def f(x):
           x[:] = x.mean()

    axis : str or int
        Specification of axis of XYZImage

    Returns
    -------

    newim : XYZImage with a modified copy of xyz_image._data.

    """

    image = Image(xyz_image.get_data(), xyz_image.coordmap)
    image = image_rollaxis(image, axis)

    data = image.get_data().copy()
    for d in data:
        modify(d)

    import copy
    newimage = Image(data, copy.copy(image.coordmap))

    # Now, we have to put the data back in order
    # with XYZImage

    newimage = synchronized_order(newimage, xyz_image)
    return XYZImage.from_image(newimage)


def test_specific_reduce():
    x = np.random.standard_normal((3,5,7,9))
    im = XYZImage(x, np.diag([3,4,5,1]), 'ijkq')

    yield assert_raises, ValueError, need_specific_axis_reduce, im, lambda x: x.sum(0)
    
    # we have to rename the axis before we can call the
    # function need_specific_axis_reduce on it

    im = im.renamed_axes(q='specific')

    newim = need_specific_axis_reduce(im, lambda x: x.sum(0))

    yield assert_equal, im.xyz_transform, newim.xyz_transform
    yield assert_equal, newim.axes.coord_names, tuple('ijk')
    yield assert_equal, newim.shape, (3,5,7)
    yield assert_almost_equal, newim.get_data(), x.sum(3)


def test_reduce():
    x = np.random.standard_normal((3,5,7,9))
    im = XYZImage(x, np.diag([3,4,5,1]), 'ijkq')
    newim = image_reduce(im, lambda x: x.sum(0), 'q')
    yield assert_equal, im.xyz_transform, newim.xyz_transform
    yield assert_equal, newim.axes.coord_names, tuple('ijk')
    yield assert_equal, newim.shape, (3,5,7)
    yield assert_almost_equal, newim.get_data(), x.sum(3)

    for i, o, n in zip('ijk', 'xyz', range(3)):
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), i
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), o
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), n


def test_call():
    x = np.random.standard_normal((3,5,7,12))
    affine = np.random.standard_normal((4,4))
    affine[-1] = [0,0,0,1]
    im = XYZImage(x, affine, 'ijkq')
    newim = image_call(im, lambda x: x[::2], 'q', 'out')
    yield assert_equal, im.xyz_transform, newim.xyz_transform
    yield assert_equal, newim.axes.coord_names, tuple('ijk') + ('out',)
    yield assert_equal, newim.shape, (3,5,7,6)
    yield assert_almost_equal, newim.get_data(), x[:,:,:,::2]

    for i, o, n in zip('ijk', 'xyz', range(3)):
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), i
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), o
        yield assert_raises, ValueError, image_reduce, im, lambda x: x.sum(0), n


def test_modify():
    x = np.random.standard_normal((3,5,7,12))
    affine = np.random.standard_normal((4,4))
    affine[-1] = [0,0,0,1]
    im = XYZImage(x, affine, 'ijkq')

    def nullmodify(d):
        pass

    def meanmodify(d):
        d[:] = d.mean()

    for i, o, n in zip('ijkq', ['x+LR', 'y+PA', 'z+SI', 'q'], range(4)):
        for a in i, o, n:
            nullim = image_modify(im, nullmodify, a)
            meanim = image_modify(im, meanmodify, a)

            yield assert_equal, nullim.get_data(), im.get_data()
            yield assert_equal, nullim.xyz_transform, im.xyz_transform
            yield assert_equal, nullim.axes, im.axes
#            yield assert_equal, nullim, im

            yield assert_equal, meanim.xyz_transform, im.xyz_transform
            yield assert_equal, meanim.axes, im.axes

        # Make sure that meanmodify works as expected

        d = im.get_data()
        d = np.rollaxis(d, n)

        meand = meanim.get_data()
        meand = np.rollaxis(meand, n)

        for i in range(d.shape[0]):
            yield assert_almost_equal, meand[i], d[i].mean()
