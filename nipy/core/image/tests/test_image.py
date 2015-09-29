from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings
from itertools import chain

import numpy as np

import nibabel as nib

from .. import image
from ..image import Image, iter_axis, is_image, rollimg
from ...api import parcels, data_generator, write_data
from ...reference.coordinate_system import CoordinateSystem
from ...reference.coordinate_map import AffineTransform, AxisError

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_array_equal)

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()

_data = np.arange(24).reshape((4,3,2))
gimg = Image(_data, AffineTransform('ijk', 'xyz', np.eye(4)))


def test_init():
    data = gimg.get_data()
    new = Image(data, gimg.coordmap)
    assert_array_almost_equal(gimg.get_data(), new.get_data())
    assert_equal(new.coordmap, gimg.coordmap)
    assert_raises(TypeError, Image)
    assert_raises(TypeError, Image, data)


def test_maxmin_values():
    y = gimg.get_data()
    assert_equal(y.shape, tuple(gimg.shape))
    assert_equal(y.max(), 23)
    assert_equal(y.min(), 0.0)


def test_slice_plane():
    x = gimg[1]
    assert_equal(x.shape, gimg.shape[1:])


def test_slice_block():
    x = gimg[1:3]
    assert_equal(x.shape, (2,) + tuple(gimg.shape[1:]))


def test_slice_step():
    s = slice(0,4,2)
    x = gimg[s]
    assert_equal(x.shape, (2,) + tuple(gimg.shape[1:]))
    x = gimg[0:4:2]
    assert_equal(x.shape, (2,) + tuple(gimg.shape[1:]))


def test_slice_type():
    s = slice(0, gimg.shape[0])
    x = gimg[s]
    assert_equal(x.shape, gimg.shape)
    x = gimg[0:]
    assert_equal(x.shape, gimg.shape)


def test_slice_steps():
    dim0, dim1, dim2 = gimg.shape
    slice_z = slice(0, dim0, 2)
    slice_y = slice(0, dim1, 2)
    slice_x = slice(0, dim2, 2)
    x = gimg[slice_z, slice_y, slice_x]
    newshape = tuple(np.floor((np.array(gimg.shape) - 1)/2) + 1)
    assert_equal(x.shape, newshape)
    x = gimg[0:dim0:2,0:dim1:2,0:dim2:2]
    assert_equal(x.shape, newshape)


def test_get_data():
    # get_data always returns an array
    x = gimg.get_data()
    assert_true(isinstance(x, np.ndarray))
    assert_equal(x.shape, gimg.shape)
    assert_equal(x.ndim, gimg.ndim)


def test_generator():
    # User iter_axis to return slices
    gen = iter_axis(gimg, axis=0)
    for img_slice in gen:
        assert_equal(img_slice.shape, (3,2))


def test_iter():
    for img_slice in image.iter_axis(gimg, 0):
        assert_equal(img_slice.shape, (3,2))
    tmp = np.zeros(gimg.shape)
    write_data(tmp, enumerate(iter_axis(gimg, 0, asarray=True)))
    assert_array_almost_equal(tmp, gimg.get_data())
    tmp = np.zeros(gimg.shape)
    g = iter_axis(gimg, 0, asarray=True)
    write_data(tmp, enumerate(g))
    assert_array_almost_equal(tmp, gimg.get_data())


def test_parcels1():
    parcelmap = gimg.get_data().astype(np.int32)
    test = np.zeros(parcelmap.shape)
    v = 0
    for i, d in data_generator(test, parcels(parcelmap)):
        v += d.shape[0]
    assert_equal(v, np.product(test.shape))


def test_parcels3():
    rho = gimg[0]
    parcelmap = rho.get_data().astype(np.int32)
    labels = np.unique(parcelmap)
    test = np.zeros(rho.shape)
    v = 0
    for i, d in data_generator(test, parcels(parcelmap, labels=labels)):
        v += d.shape[0]
    yield assert_equal, v, np.product(test.shape)


def test_slicing_returns_image():
    data = np.ones((2,3,4))
    img = Image(data, AffineTransform('kji', 'zyx', np.eye(4)))
    assert_true(isinstance(img, Image))
    assert_equal(img.ndim, 3)
    # 2D slice
    img2D = img[:,:,0]
    assert_true(isinstance(img2D, Image))
    assert_equal(img2D.ndim, 2)
    # 1D slice
    img1D = img[:,0,0]
    assert_true(isinstance(img1D, Image))
    assert_equal(img1D.ndim, 1)


class ArrayLikeObj(object):
    """The data attr in Image is an array-like object.
    Test the array-like interface that we'll expect to support."""
    def __init__(self):
        self._data = np.ones((2,3,4))

    @property
    def shape(self):
        return self._data.shape

    def __array__(self):
        return self._data


def test_ArrayLikeObj():
    obj = ArrayLikeObj()
    # create simple coordmap
    xform = np.eye(4)
    coordmap = AffineTransform.from_params('xyz', 'ijk', xform)
    # create image form array-like object and coordmap
    img = image.Image(obj, coordmap)
    assert_equal(img.ndim, 3)
    assert_equal(img.shape, (2,3,4))
    assert_array_almost_equal(img.get_data(), 1)
    # Test that the array stays with the image, so we can assign the array
    # in-place, at least in this case
    img.get_data()[:] = 4
    assert_array_equal(img.get_data(), 4)


def test_defaults_ND():
    for arr_shape, in_names, out_names in (
        ((2,3), 'kj', 'yz'),
        ((2,3,4), 'ijk', 'zyx'),
        ((2,3,4,5), 'hijk', 'zyxt')):
        data = np.ones(arr_shape)
        ndim = len(arr_shape)
        img = Image(data, AffineTransform(in_names, out_names, np.eye(ndim+1)))
        assert_true(isinstance(img._data, np.ndarray))
        assert_equal(img.ndim, len(arr_shape))
        assert_equal(img.shape, arr_shape)
        assert_equal(img.affine.shape, (img.ndim+1, img.ndim+1))
        assert_true(img.affine.diagonal().all())
        # img.header deprecated, when removed, test will raise Error
        assert_raises(AttributeError, getattr, img, 'header')


def test_header():
    # Property header interface deprecated
    arr = np.arange(24).reshape((2,3,4))
    coordmap = AffineTransform.from_params('xyz', 'ijk', np.eye(4))
    header = nib.Nifti1Header()
    img = Image(arr, coordmap, metadata={'header': header})
    assert_equal(img.metadata['header'], header)
    # This interface deprecated
    assert_equal(img.header, header)
    hdr2 = nib.Nifti1Header()
    hdr2['descrip'] = 'from fullness of heart'
    assert_not_equal(img.header, hdr2)
    img.header = hdr2
    assert_equal(img.header, hdr2)


def test_from_image():
    # from_image classmethod copies
    arr = np.arange(24).reshape((2,3,4))
    coordmap = AffineTransform.from_params('xyz', 'ijk', np.eye(4))
    img = Image(arr, coordmap, metadata={'field': 'value'})
    img2 = Image.from_image(img)
    assert_array_equal(img.get_data(), img2.get_data())
    assert_equal(img.coordmap, img2.coordmap)
    assert_equal(img.metadata, img2.metadata)
    assert_false(img.metadata is img2.metadata)
    # optional inputs - data
    arr2 = arr + 10
    new = Image.from_image(img, arr2)
    assert_array_almost_equal(arr2, new.get_data())
    assert_equal(new.coordmap, coordmap)
    new = Image.from_image(img, data=arr2)
    assert_array_almost_equal(arr2, new.get_data())
    assert_equal(new.coordmap, coordmap)
    # optional inputs - coordmap
    coordmap2 = AffineTransform.from_params('pqr', 'ijk', np.eye(4))
    new = Image.from_image(img, arr2, coordmap2)
    assert_array_almost_equal(arr2, new.get_data())
    assert_equal(new.coordmap, coordmap2)
    new = Image.from_image(img, coordmap=coordmap2)
    assert_array_almost_equal(arr, new.get_data())
    assert_equal(new.coordmap, coordmap2)
    # Optional inputs - metadata
    assert_equal(new.metadata, img.metadata)
    another_meta = {'interesting': 'information'}
    new = Image.from_image(img, arr2, coordmap2, another_meta)
    assert_array_almost_equal(arr2, new.get_data())
    assert_equal(another_meta, new.metadata)


def test_synchronized_order():
    data = np.random.standard_normal((3,4,7,5))
    im = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    im_scrambled = im.reordered_axes('iljk').reordered_reference('xtyz')
    im_unscrambled = image.synchronized_order(im_scrambled, im)
    assert_equal(im_unscrambled.coordmap, im.coordmap)
    assert_almost_equal(im_unscrambled.get_data(), im.get_data())
    assert_equal(im_unscrambled, im)
    assert_true(im_unscrambled == im)
    assert_false(im_unscrambled != im)
    # the images don't have to be the same shape
    data2 = np.random.standard_normal((3,11,9,4))
    im2 = Image(data2, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    im_scrambled2 = im2.reordered_axes('iljk').reordered_reference('xtyz')
    im_unscrambled2 = image.synchronized_order(im_scrambled2, im)
    assert_equal(im_unscrambled2.coordmap, im.coordmap)
    # or the same coordmap
    data3 = np.random.standard_normal((3,11,9,4))
    im3 = Image(data3, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,9,3,-2,1])))
    im_scrambled3 = im3.reordered_axes('iljk').reordered_reference('xtyz')
    im_unscrambled3 = image.synchronized_order(im_scrambled3, im)
    assert_equal(im_unscrambled3.axes, im.axes)
    assert_equal(im_unscrambled3.reference, im.reference)


def test_iter_axis():
    # axis iteration helper function.  This function also tests rollimg,
    # because iter_axis uses rollimg
    iter_axis = image.iter_axis
    data = np.arange(24).reshape((4,3,2))
    img = Image(data, AffineTransform('ijk', 'xyz', np.eye(4)))
    for ax_id, ax_no in (('i',0), ('j',1), ('k',2),
                        ('x',0), ('y',1), ('z',2),
                        (0,0), (1,1), (2,2),
                        (-1,2)):
        slices = list(iter_axis(img, ax_id))
        expected_shape = list(data.shape)
        g_len = expected_shape.pop(ax_no)
        assert_equal(len(slices), g_len)
        for s in slices:
            assert_equal(list(s.shape), expected_shape)
        # test asarray
        slicer = [slice(None) for i in range(data.ndim)]
        for i, s in enumerate(iter_axis(img, ax_id, asarray=True)):
            slicer[ax_no] = i
            assert_array_equal(s, data[slicer])


def test_rollaxis():
    data = np.random.standard_normal((3,4,7,5))
    im = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    # for the inverse we must specify an integer
    assert_raises(ValueError, image.rollaxis, im, 'i', True)
    # Check that rollaxis preserves diagonal affines, as claimed
    assert_almost_equal(image.rollaxis(im, 1).affine, np.diag([2,1,3,4,1]))
    assert_almost_equal(image.rollaxis(im, 2).affine, np.diag([3,1,2,4,1]))
    assert_almost_equal(image.rollaxis(im, 3).affine, np.diag([4,1,2,3,1]))
    # Check that ambiguous axes raise an exception
    # 'l' appears both as an axis and a reference coord name
    # and in different places
    im_amb = Image(data, AffineTransform.from_params('ijkl', 'xylt', np.diag([1,2,3,4,1])))
    assert_raises(ValueError, image.rollaxis, im_amb, 'l')
    # But if it's unambiguous, then
    # 'l' can appear both as an axis and a reference coord name
    im_unamb = Image(data, AffineTransform.from_params('ijkl', 'xyzl', np.diag([1,2,3,4,1])))
    im_rolled = image.rollaxis(im_unamb, 'l')
    assert_almost_equal(im_rolled.get_data(),
                        im_unamb.get_data().transpose([3,0,1,2]))
    for i, o, n in zip('ijkl', 'xyzt', range(4)):
        im_i = image.rollaxis(im, i)
        im_o = image.rollaxis(im, o)
        im_n = image.rollaxis(im, n)
        assert_almost_equal(im_i.get_data(), im_o.get_data())
        assert_almost_equal(im_i.affine, im_o.affine)
        assert_almost_equal(im_n.get_data(), im_o.get_data())
        for _im in [im_n, im_o, im_i]:
            im_n_inv = image.rollaxis(_im, n, inverse=True)
            assert_almost_equal(im_n_inv.affine, im.affine)
            assert_almost_equal(im_n_inv.get_data(), im.get_data())


def test_is_image():
    # Test is_image check
    arr = np.arange(24).reshape((2,3,4))
    cmap = AffineTransform(CoordinateSystem('ijk'),
                           CoordinateSystem('xyz'),
                           np.eye(4))
    img = Image(arr, cmap)
    assert_true(is_image(img))
    assert_false(is_image(object()))
    class C(object):
        def get_data(self): pass
    c = C()
    assert_false(is_image(c))
    c.coordmap = None
    assert_false(is_image(c))
    c.metadata = None
    assert_true(is_image(c))


def test_rollimg():
    AT = AffineTransform
    data = np.random.standard_normal((3,4,7,5))
    aff = np.diag([1,2,3,4,1])
    im = Image(data, AT('ijkl', 'xyzt', aff))
    # No attempt to preserve the diagonal
    im1 = rollimg(im, 1)
    assert_equal(im1.coordmap, rollimg(im, 'j').coordmap)
    assert_equal(im1.coordmap, rollimg(im, 'y').coordmap)
    assert_equal(im1.coordmap, rollimg(im, -3).coordmap)
    assert_equal(im1.coordmap,
                 AT('jikl', 'xyzt', aff[:, (1, 0, 2, 3, 4)]))
    assert_array_equal(im1.get_data(), np.rollaxis(data, 1))
    im2 = rollimg(im, 2)
    assert_equal(im2.coordmap, rollimg(im, 'k').coordmap)
    assert_equal(im2.coordmap, rollimg(im, 'z').coordmap)
    assert_equal(im2.coordmap, rollimg(im, -2).coordmap)
    assert_equal(im2.coordmap,
                 AT('kijl', 'xyzt', aff[:, (2, 0, 1, 3, 4)]))
    assert_array_equal(im2.get_data(), np.rollaxis(data, 2))
    im3 = rollimg(im, 3)
    assert_equal(im3.coordmap, rollimg(im, 'l').coordmap)
    assert_equal(im3.coordmap, rollimg(im, 't').coordmap)
    assert_equal(im3.coordmap, rollimg(im, -1).coordmap)
    assert_equal(im3.coordmap,
                        AT('lijk', 'xyzt', aff[:, (3, 0, 1, 2, 4)]))
    assert_array_equal(im3.get_data(), np.rollaxis(data, 3))
    # We can roll to before a specified axis
    im31 = rollimg(im, 3, 1)
    assert_equal(im31.coordmap, rollimg(im, 'l', 'j').coordmap)
    assert_equal(im31.coordmap, rollimg(im, 't', 'y').coordmap)
    assert_equal(im31.coordmap, rollimg(im, 't', 'j').coordmap)
    assert_equal(im31.coordmap, rollimg(im, 'l', 'y').coordmap)
    assert_equal(im31.coordmap, rollimg(im, -1, 'y').coordmap)
    assert_equal(im31.coordmap, rollimg(im, -1, -3).coordmap)
    assert_equal(im31.coordmap,
                 AT('iljk', 'xyzt', aff[:, (0, 3, 1, 2, 4)]))
    assert_array_equal(im31.get_data(), np.rollaxis(data, 3, 1))
    # Check that ambiguous axes raise an exception; 'l' appears both as an axis
    # and a reference coord name and in different places
    im_amb = Image(data, AT('ijkl', 'xylt', np.diag([1,2,3,4,1])))
    assert_raises(AxisError, rollimg, im_amb, 'l')
    # But if it's unambiguous, then 'l' can appear both as an axis and a
    # reference coord name
    im_unamb = Image(data, AT('ijkl', 'xyzl', np.diag([1,2,3,4,1])))
    im_rolled = rollimg(im_unamb, 'l')
    assert_array_equal(im_rolled.get_data(),
                       im_unamb.get_data().transpose([3,0,1,2]))
    # Zero row / col means we can't find an axis mapping, when fix0 is false
    aff_z = np.diag([1, 2, 3, 0, 1])
    im_z = Image(data, AT('ijkl', 'xyzt', aff_z))
    assert_raises(AxisError, rollimg, im_z, 't', fix0=False)
    # But we can work it out if we turn on our zero detector
    assert_equal(rollimg(im_z, 't', fix0=True).coordmap,
                 AT('lijk', 'xyzt', aff_z[:, (3, 0, 1, 2, 4)]))
    # That's the default
    assert_equal(rollimg(im_z, 't').coordmap,
                 AT('lijk', 'xyzt', aff_z[:, (3, 0, 1, 2, 4)]))
    # Non square is OK
    aff_r = np.array([[1, 0, 0, 10],
                      [0, 2, 0, 11],
                      [0, 0, 2, 12],
                      [0, 0, 0, 13],
                      [0, 0, 0, 1]])
    im_r = Image(data[:,:,:,0], AT('ijk', 'xyzt', aff_r))
    assert_equal(rollimg(im_r, 'k').coordmap,
                 AT('kij', 'xyzt', aff_r[:, (2, 0, 1, 3)]))
    # Unless you're tring to get at the dropped input dimension of course
    assert_raises(AxisError, rollimg, im_r, 't')
    # Another check for integers, input names, output names, reversing
    for i, o, n in zip('ijkl', 'xyzt', range(4)):
        im_i = rollimg(im, i)
        im_o = rollimg(im, o)
        im_n = rollimg(im, n)
        assert_array_equal(im_i.get_data(), im_o.get_data())
        assert_array_equal(im_i.affine, im_o.affine)
        assert_array_equal(im_n.get_data(), im_o.get_data())
        for _im in [im_n, im_o, im_i]:
            # We're rollimg back.  We want to roll the new axis 0 back to where
            # it started, which was position n
            im_n_inv = rollimg(_im, 0, n + 1)
            assert_array_equal(im_n_inv.affine, im.affine)
            assert_array_equal(im_n_inv.get_data(), im.get_data())


def test_rollimg_rollaxis():
    # Check that rollimg and rollaxis do the same
    AT = AffineTransform
    data = np.random.standard_normal((3,4,7,5))
    aff = np.diag([1,2,3,4,1])
    img = Image(data, AT('ijkl', 'xyzt', aff))
    for axis in chain(range(4), range(-3, -1)):
        rdata = np.rollaxis(data, axis)
        rimg = rollimg(img, axis)
        assert_array_equal(rdata, rimg.get_data())
        for start in chain(range(4), range(-3, -1)):
            rdata = np.rollaxis(data, axis, start)
            rimg = rollimg(img, axis, start)
            assert_array_equal(rdata, rimg.get_data())


def test_rollaxis_inverse():
    # Test deprecated image rollaxis with inverse
    AT = AffineTransform
    data = np.random.standard_normal((3,4,7,5))
    aff = np.diag([1,2,3,4,1])
    img = Image(data, AT('ijkl', 'xyzt', aff))
    for axis in chain(range(4), range(-3, -1)):
        rimg = image.rollaxis(img, axis)
        rdata = np.rollaxis(data, axis)
        assert_array_equal(rdata, rimg.get_data())
        rrimg = image.rollaxis(rimg, axis, inverse=True)
        assert_array_equal(data, rrimg.get_data())
