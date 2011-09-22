# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nipy.io.api import load_image, save_image
from nipy.core import api
from nipy.utils.tmpdirs import InTemporaryDirectory

from nipy.testing import (assert_true, assert_false, assert_equal,
                          assert_array_almost_equal, funcfile)


TMP_FNAME = 'afile.nii'


def test_save1():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file
    img = load_image(funcfile)
    with InTemporaryDirectory():
        save_image(img, TMP_FNAME)
        img2 = load_image(TMP_FNAME)
        assert_array_almost_equal(img.affine, img2.affine)
        assert_equal(img.shape, img2.shape)
        assert_array_almost_equal(img2.get_data(), img.get_data())
        del img2


def test_save2():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file 
    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.93])
    cmap = api.AffineTransform.from_start_step('ijkt', 'xyzt', [1,3,5,0], step)
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    with InTemporaryDirectory():
        save_image(img, TMP_FNAME)
        img2 = load_image(TMP_FNAME)
        assert_array_almost_equal(img.affine, img2.affine)
        assert_equal(img.shape, img2.shape)
        assert_array_almost_equal(img2.get_data(), img.get_data())
        del img2


def test_save2b():
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file This example has
    # a non-diagonal affine matrix for the spatial part, but is
    # 'diagonal' for the space part.  this should raise a warnings
    # about 'non-diagonal' affine matrix

    # make a 5x5 transformation
    step = np.array([3.45,2.3,4.5,6.9])
    A = np.random.standard_normal((4,4))
    B = np.diag(list(step)+[1])
    B[:4,:4] = A
    shape = (13,5,7,3)
    cmap = api.AffineTransform.from_params('ijkt', 'xyzt', B)
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    with InTemporaryDirectory():
        save_image(img, TMP_FNAME)
        img2 = load_image(TMP_FNAME)
        assert_false(np.allclose(img.affine, img2.affine))
        assert_array_almost_equal(img.affine[:3,:3], img2.affine[:3,:3])
        assert_equal(img.shape, img2.shape)
        assert_array_almost_equal(img2.get_data(), img.get_data())
        del img2

# JT: nifti_ref doesn't reorder axes anymore so these tests
# are no longer expected to work
#
# def test_save3():
#     # A test to ensure that when a file is saved, the affine
#     # and the data agree. In this case, things don't agree:
#     # i) the pixdim is off
#     # ii) makes the affine off

#     step = np.array([3.45,2.3,4.5,6.9])
#     shape = (13,5,7,3)
#     cmap = api.AffineTransform.from_start_step('jkli', 'tzyx', [0,3,5,1], step)

#     data = np.random.standard_normal(shape)
#     img = api.Image(data, cmap)
#     save_image(img, TMP_FNAME)
#     img2 = load_image(TMP_FNAME)

#     yield assert_equal, tuple([img.shape[l] for l in [3,0,1,2]]), img2.shape
#     a = np.transpose(np.asarray(img), [3,0,1,2])
#     yield assert_false, np.allclose(img.affine, img2.affine)
#     yield assert_true, np.allclose(a, np.asarray(img2))


# def test_save4():
#     # Same as test_save3 except we have reordered the 'ijk' input axes.
#     shape = (13,5,7,3)
#     step = np.array([3.45,2.3,4.5,6.9])
#     # When the input coords are in the 'ljki' order, the affines get
#     # rearranged.  Note that the 'start' below, must be 0 for
#     # non-spatial dimensions, because we have no way to store them in
#     # most cases.  For example, a 'start' of [1,5,3,1] would be lost on
#     # reload
#     cmap = api.AffineTransform.from_start_step('lkji', 'tzyx', [2,5,3,1], step)
#     data = np.random.standard_normal(shape)
#     img = api.Image(data, cmap)
#     save_image(img, TMP_FNAME)
#     img2 = load_image(TMP_FNAME)
#     P = np.array([[0,0,0,1,0],
#                   [0,0,1,0,0],
#                   [0,1,0,0,0],
#                   [1,0,0,0,0],
#                   [0,0,0,0,1]])
#     res = np.dot(P, np.dot(img.affine, P.T))

#     # the step part of the affine should be set correctly
#     yield assert_array_almost_equal, res[:4,:4], img2.affine[:4,:4]

#     # start in the spatial dimensions should be set correctly
#     yield assert_array_almost_equal, res[:3,-1], img2.affine[:3,-1]

#     # start in the time dimension should not be 2 as in img, but 0
#     # because NIFTI dosen't have a time start

#     yield assert_false, (res[3,-1] == img2.affine[3,-1])
#     yield assert_true, (res[3,-1] == 2)
#     yield assert_true, (img2.affine[3,-1] == 0)

#     # shapes should be reversed because img has coordinates reversed


#     yield assert_equal, img.shape[::-1], img2.shape

#     # data should be transposed because coordinates are reversed

#     yield (assert_array_almost_equal, 
#            np.transpose(np.asarray(img2),[3,2,1,0]),
#            np.asarray(img))

#     # coordinate names should be reversed as well

#     yield assert_equal, img2.coordmap.function_domain.coord_names, \
#         img.coordmap.function_domain.coord_names[::-1]
#     yield assert_equal, img2.coordmap.function_domain.coord_names, \
#         ['i', 'j', 'k', 'l']
