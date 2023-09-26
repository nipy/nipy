# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from nibabel.affines import from_matvec
from numpy.testing import assert_array_almost_equal

from nipy.core import api
from nipy.core.reference.coordinate_map import AffineTransform as AT
from nipy.core.reference.coordinate_system import CoordinateSystem as CS
from nipy.core.reference.spaces import mni_csm
from nipy.io.api import load_image, save_image
from nipy.testing import funcfile

TMP_FNAME = 'afile.nii'


def test_save1(in_tmp_path):
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file
    img = load_image(funcfile)
    save_image(img, TMP_FNAME)
    img2 = load_image(TMP_FNAME)
    assert_array_almost_equal(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert_array_almost_equal(img2.get_fdata(), img.get_fdata())
    del img2


def test_save2(in_tmp_path):
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file
    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.93])
    cmap = api.AffineTransform.from_start_step('ijkt', 'xyzt', [1,3,5,0], step)
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, TMP_FNAME)
    img2 = load_image(TMP_FNAME)
    assert_array_almost_equal(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert_array_almost_equal(img2.get_fdata(), img.get_fdata())
    del img2


def test_save2b(in_tmp_path):
    # A test to ensure that when a file is saved, the affine and the
    # data agree. This image comes from a NIFTI file.  This example has a
    # non-diagonal affine matrix for the spatial part, but is 'diagonal' for the
    # space part.
    #
    # make a 5x5 transformation (for 4d image)
    step = np.array([3.45, 2.3, 4.5, 6.9])
    A = np.random.standard_normal((3,3))
    B = np.diag(list(step)+[1])
    B[:3, :3] = A
    shape = (13,5,7,3)
    cmap = api.vox2mni(B)
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, TMP_FNAME)
    img2 = load_image(TMP_FNAME)
    assert_array_almost_equal(img.affine, img2.affine)
    assert img.shape == img2.shape
    assert_array_almost_equal(img2.get_fdata(), img.get_fdata())
    del img2


def test_save3(in_tmp_path):
    # A test to ensure that when a file is saved, the affine
    # and the data agree. In this case, things don't agree:
    # i) the pixdim is off
    # ii) makes the affine off
    step = np.array([3.45,2.3,4.5,6.9])
    shape = (13,5,7,3)
    mni_xyz = mni_csm(3).coord_names
    cmap = AT(CS('jkli'),
              CS(('t',) + mni_xyz[::-1]),
              from_matvec(np.diag([0,3,5,1]), step))
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, TMP_FNAME)
    tmp = load_image(TMP_FNAME)
    # Detach image from file so we can delete it
    data = tmp.get_fdata().copy()
    img2 = api.Image(data, tmp.coordmap, tmp.metadata)
    del tmp
    assert tuple([img.shape[l] for l in [3,2,1,0]]) == img2.shape
    a = np.transpose(img.get_fdata(), [3,2,1,0])
    assert not np.allclose(img.affine, img2.affine)
    assert np.allclose(a, img2.get_fdata())


def test_save4(in_tmp_path):
    # Same as test_save3 except we have reordered the 'ijk' input axes.
    shape = (13,5,7,3)
    step = np.array([3.45,2.3,4.5,6.9])
    # When the input coords are in the 'ljki' order, the affines get
    # rearranged.  Note that the 'start' below, must be 0 for
    # non-spatial dimensions, because we have no way to store them in
    # most cases.  For example, a 'start' of [1,5,3,1] would be lost on
    # reload
    mni_xyz = mni_csm(3).coord_names
    cmap = AT(CS('tkji'),
              CS(('t',) + mni_xyz[::-1]),
              from_matvec(np.diag([2., 3, 5, 1]), step))
    data = np.random.standard_normal(shape)
    img = api.Image(data, cmap)
    save_image(img, TMP_FNAME)
    tmp = load_image(TMP_FNAME)
    data = tmp.get_fdata().copy()
    # Detach image from file so we can delete it
    img2 = api.Image(data, tmp.coordmap, tmp.metadata)
    del tmp
    P = np.array([[0,0,0,1,0],
                  [0,0,1,0,0],
                  [0,1,0,0,0],
                  [1,0,0,0,0],
                  [0,0,0,0,1]])
    res = np.dot(P, np.dot(img.affine, P.T))
    # the step part of the affine should be set correctly
    assert_array_almost_equal(res[:4,:4], img2.affine[:4,:4])
    # start in the spatial dimensions should be set correctly
    assert_array_almost_equal(res[:3,-1], img2.affine[:3,-1])
    # start in the time dimension should be 3.45 as in img, because NIFTI stores
    # the time offset in hdr[``toffset``]
    assert res[3,-1] != img2.affine[3,-1]
    assert res[3,-1] == 3.45
    # shapes should be reversed because img has coordinates reversed
    assert img.shape[::-1] == img2.shape
    # data should be transposed because coordinates are reversed
    assert_array_almost_equal(
        np.transpose(img2.get_fdata(),[3,2,1,0]),
        img.get_fdata())
    # coordinate names should be reversed as well
    assert (img2.coordmap.function_domain.coord_names ==
                 img.coordmap.function_domain.coord_names[::-1])
    assert (img2.coordmap.function_domain.coord_names ==
                 ('i', 'j', 'k', 't'))
