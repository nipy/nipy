from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from itertools import product

import numpy as np

from nipy.core.api import (CoordinateMap, AffineTransform, Image,
        ArrayCoordMap, vox2mni)
from nipy.core.reference import slices
from nipy.algorithms.resample import resample, resample_img2img
from nipy.io.api import load_image

from nose.tools import assert_true, assert_raises

from numpy.testing import assert_array_almost_equal, assert_array_equal
from nipy.testing import funcfile, anatfile


def test_resample_img2img():
    fimg = load_image(funcfile)
    aimg = load_image(anatfile)
    resimg = resample_img2img(fimg, fimg)
    yield assert_true, np.allclose(resimg.get_data(), fimg.get_data())
    yield assert_raises, ValueError, resample_img2img, fimg, aimg


# Hackish flag for enabling of pylab plots of resamplingstest_2d_from_3d
gui_review = False

def test_rotate2d():
    # Rotate an image in 2d on a square grid, should result in transposed image
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.7,0.5,1]))
    g2 = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.7,1]))
    i = Image(np.ones((100,100)), g)
    # This sets the image data by writing into the array
    i.get_data()[50:55,40:55] = 3.
    a = np.array([[0,1,0],
                  [1,0,0],
                  [0,0,1]], np.float)
    ir = resample(i, g2, a, (100, 100))
    assert_array_almost_equal(ir.get_data().T, i.get_data())


def test_rotate2d2():
    # Rotate an image in 2d on a non-square grid, should result in transposed
    # image
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.7,0.5,1]))
    g2 = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.7,1]))
    i = Image(np.ones((100,80)), g)
    # This sets the image data by writing into the array
    i.get_data()[50:55,40:55] = 3.
    a = np.array([[0,1,0],
                  [1,0,0],
                  [0,0,1]], np.float)
    ir = resample(i, g2, a, (80,100))
    assert_array_almost_equal(ir.get_data().T, i.get_data())


def test_rotate2d3():
    # Another way to rotate/transpose the image, similar to
    # test_rotate2d2 and test_rotate2d, except the world of the
    # output coordmap is the same as the world of the
    # original image. That is, the data is transposed on disk, but the
    # output coordinates are still 'x,'y' order, not 'y', 'x' order as
    # above

    # this functionality may or may not be used a lot. if data is to
    # be transposed but one wanted to keep the NIFTI order of output
    # coords this would do the trick
    g = AffineTransform.from_params('xy', 'ij', np.diag([0.5,0.7,1]))
    i = Image(np.ones((100,80)), g)
    # This sets the image data by writing into the array
    i.get_data()[50:55,40:55] = 3.
    a = np.identity(3)
    g2 = AffineTransform.from_params('xy', 'ij', np.array([[0,0.5,0],
                                                  [0.7,0,0],
                                                  [0,0,1]]))
    ir = resample(i, g2, a, (80,100))
    assert_array_almost_equal(ir.get_data().T, i.get_data())


def test_rotate3d():
    # Rotate / transpose a 3d image on a non-square grid
    g = AffineTransform.from_params('ijk', 'xyz', np.diag([0.5,0.6,0.7,1]))
    g2 = AffineTransform.from_params('ijk', 'xyz', np.diag([0.5,0.7,0.6,1]))
    shape = (100,90,80)
    i = Image(np.ones(shape), g)
    i.get_data()[50:55,40:55,30:33] = 3.
    a = np.array([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1.]])
    ir = resample(i, g2, a, (100,80,90))
    assert_array_almost_equal(np.transpose(ir.get_data(), (0,2,1)),
                              i.get_data())


def test_resample2d():
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.5,1]))
    i = Image(np.ones((100,90)), g)
    i.get_data()[50:55,40:55] = 3.
    # This mapping describes a mapping from the "target" physical
    # coordinates to the "image" physical coordinates.  The 3x3 matrix
    # below indicates that the "target" physical coordinates are related
    # to the "image" physical coordinates by a shift of -4 in each
    # coordinate.  Or, to find the "image" physical coordinates, given
    # the "target" physical coordinates, we add 4 to each "target
    # coordinate".  The resulting resampled image should show the
    # overall image shifted -8,-8 voxels towards the origin
    a = np.identity(3)
    a[:2,-1] = 4.
    ir = resample(i, i.coordmap, a, (100,90))
    assert_array_almost_equal(ir.get_data()[42:47,32:47], 3.)


def test_resample2d1():
    # Tests the same as test_resample2d, only using a callable instead of
    # an AffineTransform instance
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.5,1]))
    i = Image(np.ones((100,90)), g)
    i.get_data()[50:55,40:55] = 3.
    a = np.identity(3)
    a[:2,-1] = 4.
    A = np.identity(2)
    b = np.ones(2)*4
    def mapper(x):
        return np.dot(x, A.T) + b
    ir = resample(i, i.coordmap, mapper, (100,90))
    assert_array_almost_equal(ir.get_data()[42:47,32:47], 3.)


def test_resample2d2():
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.5,1]))
    i = Image(np.ones((100,90)), g)
    i.get_data()[50:55,40:55] = 3.
    a = np.identity(3)
    a[:2,-1] = 4.
    A = np.identity(2)
    b = np.ones(2)*4
    ir = resample(i, i.coordmap, (A, b), (100,90))
    assert_array_almost_equal(ir.get_data()[42:47,32:47], 3.)


def test_resample2d3():
    # Same as test_resample2d, only a different way of specifying
    # the transform: here it is an (A,b) pair
    g = AffineTransform.from_params('ij', 'xy', np.diag([0.5,0.5,1]))
    i = Image(np.ones((100,90)), g)
    i.get_data()[50:55,40:55] = 3.
    a = np.identity(3)
    a[:2,-1] = 4.
    ir = resample(i, i.coordmap, a, (100,90))
    assert_array_almost_equal(ir.get_data()[42:47,32:47], 3.)


def test_resample3d():
    g = AffineTransform.from_params('ijk', 'xyz', np.diag([0.5,0.5,0.5,1]))
    shape = (100,90,80)
    i = Image(np.ones(shape), g)
    i.get_data()[50:55,40:55,30:33] = 3.
    # This mapping describes a mapping from the "target" physical
    # coordinates to the "image" physical coordinates.  The 4x4 matrix
    # below indicates that the "target" physical coordinates are related
    # to the "image" physical coordinates by a shift of -4 in each
    # coordinate.  Or, to find the "image" physical coordinates, given
    # the "target" physical coordinates, we add 4 to each "target
    # coordinate".  The resulting resampled image should show the
    # overall image shifted [-6,-8,-10] voxels towards the origin
    a = np.identity(4)
    a[:3,-1] = [3,4,5]
    ir = resample(i, i.coordmap, a, (100,90,80))
    assert_array_almost_equal(ir.get_data()[44:49,32:47,20:23], 3.)


def test_resample_outvalue():
    # Test resampling with different modes, constant values, datatypes, orders

    def func(xyz):
        return xyz + np.asarray([1,0,0])

    coordmap =  vox2mni(np.eye(4))
    arr = np.arange(3 * 3 * 3).reshape(3, 3, 3)
    aff = np.eye(4)
    aff[0, 3] = 1.  # x translation
    for mapping, dt, order in product(
        [aff, func],
        [np.int8, np.intp, np.int32, np.int64, np.float32, np.float64],
        [0, 1, 3]):
        img = Image(arr.astype(dt), coordmap)
        # Test constant value of 0
        img2 = resample(img, coordmap, mapping, img.shape,
                        order=order, mode='constant', cval=0.)
        exp_arr = np.zeros(arr.shape)
        exp_arr[:-1, :, :] = arr[1:, :, :]
        assert_array_almost_equal(img2.get_data(), exp_arr)
        # Test constant value of 1
        img2 = resample(img, coordmap, mapping, img.shape,
                        order=order, mode='constant', cval=1.)
        exp_arr[-1, :, :] = 1
        assert_array_almost_equal(img2.get_data(), exp_arr)
        # Test nearest neighbor
        img2 = resample(img, coordmap, mapping, img.shape,
                        order=order, mode='nearest')
        exp_arr[-1, :, :] = arr[-1, :, :]
        assert_array_almost_equal(img2.get_data(), exp_arr)
    # Test img2img
    target_coordmap = vox2mni(aff)
    target = Image(arr, target_coordmap)
    img2 = resample_img2img(img, target, 3, 'nearest')
    assert_array_almost_equal(img2.get_data(), exp_arr)
    img2 = resample_img2img(img, target, 3, 'constant', cval=1.)
    exp_arr[-1, :, :] = 1
    assert_array_almost_equal(img2.get_data(), exp_arr)


def test_nonaffine():
    # resamples an image along a curve through the image.
    #
    # FIXME: use the reference.evaluate.Grid to perform this nicer
    # FIXME: Remove pylab references
    def curve(x): # function accept N by 1, returns N by 2 
        return (np.vstack([5*np.sin(x.T),5*np.cos(x.T)]).T + [52,47])
    for names in (('xy', 'ij', 't', 'u'),('ij', 'xy', 't', 's')):
        in_names, out_names, tin_names, tout_names = names
        g = AffineTransform.from_params(in_names, out_names, np.identity(3))
        img = Image(np.ones((100,90)), g)
        img.get_data()[50:55,40:55] = 3.
        tcoordmap = AffineTransform.from_start_step(
            tin_names,
            tout_names,
            [0],
            [np.pi*1.8/100])
        ir = resample(img, tcoordmap, curve, (100,))
    if gui_review:
        import pylab
        pylab.figure(num=3)
        pylab.imshow(img, interpolation='nearest')
        d = curve(np.linspace(0,1.8*np.pi,100))
        pylab.plot(d[0], d[1])
        pylab.gca().set_ylim([0,99])
        pylab.gca().set_xlim([0,89])
        pylab.figure(num=4)
        pylab.plot(ir.get_data())


def test_2d_from_3d():
    # Resample a 3d image on a 2d affine grid
    # This example creates a coordmap that coincides with
    # the 10th slice of an image, and checks that
    # resampling agrees with the data in the 10th slice.
    shape = (100,90,80)
    g = AffineTransform.from_params('ijk', 'xyz', np.diag([0.5,0.5,0.5,1]))
    i = Image(np.ones(shape), g)
    i.get_data()[50:55,40:55,30:33] = 3.
    a = np.identity(4)
    g2 = ArrayCoordMap.from_shape(g, shape)[10]
    ir = resample(i, g2.coordmap, a, g2.shape)
    assert_array_almost_equal(ir.get_data(), i[10].get_data())


def test_slice_from_3d():
    # Resample a 3d image, returning a zslice, yslice and xslice
    #
    # This example creates a coordmap that coincides with
    # a given z, y, or x slice of an image, and checks that
    # resampling agrees with the data in the given slice.
    shape = (100,90,80)
    g = AffineTransform.from_params('ijk',
                                    'xyz',
                                    np.diag([0.5,0.5,0.5,1]))
    img = Image(np.ones(shape), g)
    img.get_data()[50:55,40:55,30:33] = 3
    I = np.identity(4)
    zsl = slices.zslice(26,
                        ((0,49.5), 100),
                        ((0,44.5), 90),
                        img.reference)
    ir = resample(img, zsl, I, (100, 90))
    assert_array_almost_equal(ir.get_data(), img[:,:,53].get_data())
    ysl = slices.yslice(22,
                        ((0,49.5), 100),
                        ((0,39.5), 80),
                        img.reference)
    ir = resample(img, ysl, I, (100, 80))
    assert_array_almost_equal(ir.get_data(), img[:,45,:].get_data())
    xsl = slices.xslice(15.5,
                        ((0,44.5), 90),
                        ((0,39.5), 80),
                        img.reference)
    ir = resample(img, xsl, I, (90, 80))
    assert_array_almost_equal(ir.get_data(), img[32,:,:].get_data())
