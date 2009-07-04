"""
This test basically just plays around with image.rollaxis.

It has three examples

image_reduce:  this takes an LPIImage having, say, an axis 't'
               and returns another LPIImage
               having reduced over 't'

image_call: this takes an LPIImage having, say, an axis 't'
            and does something along this axis -- like fits
            a regression model? and outputs a new LPIImage
            with the 't' axis replaced by 'new'

image_modify_copy: this takes an LPIImage and an axis specification,
                   such as 'x', 'l', or 2, modifies a copy of the data
                   by iterating over this axis, and returns an
                   LPIImage with the same axes

need_specific_axis_reduce: this takes an LPIImage and a specific 
                           axis name, like 't' and produces an LPIImage
                           reduced over 't'. raises an exception if
                           LPIImage has no axis 't'

Note
----

Because these are LPIImages, 't' is both an axis name and a world coordinate
name so it is not ambiguous to say 't' axis. It is slightly 
ambiguous to say 'x' axis if the axisnames are ['slice', 'frequency', 'phase']
but image.rollaxis identifies 'x' == 'slice' == 0.

"""

import numpy as np

from nipy.testing import *

from nipy.core.image.image import Image, rollaxis as image_rollaxis, \
    synchronized_order
from nipy.core.image.lpi_image import LPIImage

def image_reduce(lpi_image, reduce_op, axis='t'):
    """
    Take an LPIImage, perform some reduce operation on it, over
    a specified axis, and return a new LPIImage.

    For the sake of testing things out, we will assume that
    the operation reduces over the first axis only.

    Parameters
    ----------

    lpi_image : LPIImage

    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)

    axis : str or int
        Specification of axis of LPIImage

    Returns
    -------

    newim : LPIImage, missing axis 

    """

    if axis in lpi_image.world.coord_names + \
            lpi_image.axes.coord_names[:3] + tuple(range(3)):
        raise ValueError('cannot reduce over a spatial axis' +
                         'or we will not be able to output LPIImages')

    image = Image(lpi_image.get_data(), lpi_image.coordmap)
    image = image_rollaxis(image, axis)

    axis_name = image.axes.coord_names[0]
    output_axes = list(lpi_image.axes.coord_names)
    output_axes.remove(axis_name)

    newdata = reduce_op(image.get_data())
    return LPIImage(newdata, lpi_image.affine, output_axes)

def need_specific_axis_reduce(lpi_image, reduce_op):
    """
    Take an LPIImage, perform some reduce operation on it, over
    the axis 'specific', and returns a new LPIImage.

    For the sake of testing things out, we will assume that
    the operation reduces over the first axis only.

    Parameters
    ----------

    lpi_image : LPIImage

    reduce_op : callable
        An operation that reduces over the first axis,
        such as lambda x: x.sum(0)

    Returns
    -------

    newim : LPIImage, missing axis 

    """

    image = Image(lpi_image.get_data(), lpi_image.coordmap)
    image = image_rollaxis(image, 'specific')

    axis_name = image.axes.coord_names[0]
    output_axes = list(lpi_image.axes.coord_names)
    output_axes.remove(axis_name)

    newdata = reduce_op(image.get_data())
    return LPIImage(newdata, lpi_image.affine, output_axes)


def image_call(lpi_image, function, inaxis='t', outaxis='new'):
    """
    Take an LPIImage, perform some operation on it, over
    a specified axis, and return a new LPIImage.

    For the sake of testing things out, we will assume that
    the operation can only operate on the first
    axis of the array.

    Parameters
    ----------

    lpi_image : LPIImage

    function : callable
        An operation that does something over the first axis,
        such as lambda x: x[::2]

    inaxis : str or int
        Specification of axis of LPIImage

    outaxis : str
        Name of new axis in new LPIImage

    Returns
    -------

    newim : LPIImage with axis inaxis replaced with outaxis

    """

    if inaxis in lpi_image.world.coord_names + \
            lpi_image.axes.coord_names[:3] + tuple(range(3)):
        raise ValueError('cannot reduce over a spatial axis' +
                         'or we will not be able to output LPIImages')

    image = Image(lpi_image.get_data(), lpi_image.coordmap)
    image = image_rollaxis(image, inaxis)
    inaxis = image.axes.coord_names[0] # now it's a string

    newdata = function(image.get_data())
    newcoordmap = image.coordmap.renamed_domain({inaxis:outaxis})
    newimage = Image(newdata, newcoordmap)

    # we have to roll the axis back

    axis_index = lpi_image.axes.index(inaxis)
    newimage = image_rollaxis(newimage, axis_index, inverse=True)

    return LPIImage(newimage.get_data(), lpi_image.affine, 
                    newimage.axes.coord_names)

def image_modify(lpi_image, modify, axis='y'):
    """
    Take an LPIImage, perform some operation on it, over
    a specified axis, and return a new LPIImage.

    For this operation, we are allowed to iterate over
    spatial axes.

    For the sake of testing things out, we will assume that
    the operation modify can only operate by iterating
    over the first axis of an array.

    Parameters
    ----------

    lpi_image : LPIImage

    modify : callable
        An operation that does modifies an array.
        Something like 

        def f(x):
           x[:] = x.mean()

    axis : str or int
        Specification of axis of LPIImage

    Returns
    -------

    newim : LPIImage with a modified copy of lpi_image._data.

    """

    image = Image(lpi_image.get_data(), lpi_image.coordmap)
    image = image_rollaxis(image, axis)

    data = image.get_data().copy()
    for d in data:
        modify(d)

    import copy
    newimage = Image(data, copy.copy(image.coordmap))

    # Now, we have to put the data back in order
    # with LPIImage

    newimage = synchronized_order(newimage, lpi_image)
    return LPIImage.from_image(newimage)


def test_specific_reduce():
    x = np.random.standard_normal((3,5,7,9))
    im = LPIImage(x, np.diag([3,4,5,1]), 'ijkq')

    yield assert_raises, ValueError, need_specific_axis_reduce, im, lambda x: x.sum(0)
    
    # we have to rename the axis before we can call the
    # function need_specific_axis_reduce on it

    # this is a little clunky -- maybe renamed_axes should be a
    # method instead...

    im_renamed = im.renamed_axes(q='specific')
    lpi_renamed = LPIImage(im_renamed._data,
                           im.affine,
                           im_renamed.axes.coord_names)
                           
    newim = need_specific_axis_reduce(lpi_renamed, lambda x: x.sum(0))

    yield assert_equal, im.lpi_transform, newim.lpi_transform
    yield assert_equal, newim.axes.coord_names, tuple('ijk')
    yield assert_equal, newim.shape, (3,5,7)
    yield assert_almost_equal, newim.get_data(), x.sum(3)

def test_reduce():
    x = np.random.standard_normal((3,5,7,9))
    im = LPIImage(x, np.diag([3,4,5,1]), 'ijkq')
    newim = image_reduce(im, lambda x: x.sum(0), 'q')
    yield assert_equal, im.lpi_transform, newim.lpi_transform
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
    im = LPIImage(x, affine, 'ijkq')
    newim = image_call(im, lambda x: x[::2], 'q', 'out')
    yield assert_equal, im.lpi_transform, newim.lpi_transform
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
    im = LPIImage(x, affine, 'ijkq')

    def nullmodify(d):
        pass

    def meanmodify(d):
        d[:] = d.mean()

    for i, o, n in zip('ijkq', 'xyzq', range(4)):
        for a in i, o, n:
            nullim = image_modify(im, nullmodify, a)
            meanim = image_modify(im, meanmodify, a)

            yield assert_equal, nullim.get_data(), im.get_data()
            print nullim.lpi_transform
            print im.lpi_transform
            print im.lpi_transform == nullim.lpi_transform
            yield assert_equal, nullim.lpi_transform, im.lpi_transform
            yield assert_equal, nullim.axes, im.axes
#            yield assert_equal, nullim, im

            yield assert_equal, meanim.lpi_transform, im.lpi_transform
            yield assert_equal, meanim.axes, im.axes

        # Make sure that meanmodify works as expected

        d = im.get_data()
        d = np.rollaxis(d, n)

        meand = meanim.get_data()
        meand = np.rollaxis(meand, n)

        for i in range(d.shape[0]):
            yield assert_almost_equal, meand[i], d[i].mean()
