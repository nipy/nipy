from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nibabel.affines import from_matvec

from ..pca import pca_image, pca as pca_array
from ....core.api import Image, AffineTransform, CoordinateSystem as CS
from ....core.reference.coordinate_map import (product as cm_product,
                                               drop_io_dim, AxisError)
from ....core.image.image import rollimg
from ....io.api import  load_image

from nose.tools import assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_equal)
from ....testing import funcfile
from .test_pca import res2pos1

data_dict = {}

def setup():
    img = load_image(funcfile)
    # Here, I'm just doing this so I know that img.shape[0] is the number of
    # volumes
    t0_img = rollimg(img, 't')
    data_dict['nimages'] = t0_img.shape[0]
    # Below, I am just making a mask because I already have img, I know I can do
    # this. In principle, though, the pca function will just take another Image
    # as a mask
    img_data = t0_img.get_data()
    mask_cmap = drop_io_dim(img.coordmap, 't')
    first_frame = img_data[0]
    mask = Image(np.greater(first_frame, 500).astype(np.float64),
                 mask_cmap)
    data_dict['fmridata'] = img
    data_dict['mask'] = mask

    # print data_dict['mask'].shape, np.sum(data_dict['mask'].get_data())
    assert_equal(data_dict['mask'].shape, (17, 21, 3))
    assert_almost_equal(np.sum(data_dict['mask'].get_data()), 1071.0)

def _rank(p):
    return p['basis_vectors'].shape[1]


def test_PCAMask():
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    p = pca_image(data_dict['fmridata'], 't',
                  data_dict['mask'], ncomp=ncomp)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['axis'], 3)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape, data_dict['mask'].shape +
                 (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_mask_match():
    # we can't do PCA over spatial axes if we use a spatial mask
    ncomp = 5
    out_coords = data_dict['mask'].reference.coord_names
    for i, o, n in zip('ijk', out_coords, [0,1,2]):
        assert_raises(ValueError,
                      pca_image,
                      data_dict['fmridata'],
                      i,
                      data_dict['mask'],
                      ncomp)
        assert_raises(ValueError,
                      pca_image,
                      data_dict['fmridata'],
                      o,
                      data_dict['mask'],
                      ncomp)
        assert_raises(ValueError,
                      pca_image,
                      data_dict['fmridata'],
                      n,
                      data_dict['mask'],
                      ncomp)


def test_PCAMask_nostandardize():
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    p = pca_image(data_dict['fmridata'], 't',
                  data_dict['mask'],
                  ncomp=ncomp, standardize=False)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape, data_dict['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names, ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_PCANoMask():
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    p = pca_image(data_dict['fmridata'], ncomp=ncomp)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape, data_dict['mask'].shape +
                 (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_PCANoMask_nostandardize():
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    p = pca_image(data_dict['fmridata'], ncomp=ncomp, standardize=False)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape,
                 data_dict['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_keep():
    # Data is projected onto k=10 dimensional subspace then has its mean
    # removed. Should still have rank 10.
    k = 10
    ncomp = 5
    nimages = data_dict['nimages']
    ntotal = k
    X = np.random.standard_normal((nimages, k))
    p = pca_image(data_dict['fmridata'], ncomp=ncomp, design_keep=X)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape,
                 data_dict['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_resid():
    # Data is projected onto k=10 dimensional subspace then has its mean
    # removed.  Should still have rank 10.
    k = 10
    ncomp = 5
    nimages = data_dict['nimages']
    ntotal = k
    X = np.random.standard_normal((nimages, k))
    p = pca_image(data_dict['fmridata'], ncomp=ncomp, design_resid=X)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape,
                 data_dict['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    nimages = data_dict['nimages']
    ntotal = k1
    X1 = np.random.standard_normal((nimages, k1))
    X2 = np.random.standard_normal((nimages, k2))
    p = pca_image(data_dict['fmridata'], ncomp=ncomp, design_resid=X2, design_keep=X1)

    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape,
                 data_dict['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)

    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 data_dict['fmridata'].coordmap.affine)


def test_5d():
    # What happened to a 5d image? We should get 4d images back
    img = data_dict['fmridata']
    data = img.get_data()
    # Make a last input and output axis called 'v'
    vcs = CS('v')
    xtra_cmap = AffineTransform(vcs, vcs, np.eye(2))
    cmap_5d = cm_product(img.coordmap, xtra_cmap)
    data_5d = data.reshape(data.shape + (1,))
    fived = Image(data_5d, cmap_5d)
    mask = data_dict['mask']
    mask_data = mask.get_data()
    mask_data = mask_data.reshape(mask_data.shape + (1,))
    cmap_4d = cm_product(mask.coordmap, xtra_cmap)
    mask4d = Image(mask_data, cmap_4d)
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    p = pca_image(fived, 't', mask4d, ncomp=ncomp)
    assert_equal(_rank(p), ntotal)
    assert_equal(p['basis_vectors over t'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].shape, data.shape[:3] + (ncomp, 1))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)

    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','PCA components','v'])
    assert_equal(p['basis_projections'].coordmap.affine,
                 fived.coordmap.affine)
    # flip the PCA dimension to end
    data_5d = data.reshape(data.shape[:3] + (1, data.shape[3]))
    # Make the last axis name be 'group'.  't' is not a length 1 dimension we
    # are going to leave as is
    gcs = CS(['group'])
    xtra_cmap = AffineTransform(gcs, gcs, np.eye(2))
    cmap_5d = cm_product(img.coordmap, xtra_cmap)
    fived = Image(data_5d, cmap_5d)
    # Give the mask a 't' dimension, but no group dimension
    mask = data_dict['mask']
    mask_data = mask.get_data()
    mask_data = mask_data.reshape(mask_data.shape + (1,))
    # We need to replicate the time scaling of the image cmap, hence the 2. in
    # the affine
    xtra_cmap = AffineTransform(CS('t'), CS('t'), np.diag([2., 1]))
    cmap_4d = cm_product(mask.coordmap, xtra_cmap)
    mask4d = Image(mask_data, cmap_4d)
    nimages = data_dict['nimages']
    ntotal = nimages - 1
    ncomp = 5
    # We can now show the axis does not have to be time
    p = pca_image(fived, mask=mask4d, ncomp=ncomp, axis='group')
    assert_equal(p['basis_vectors over group'].shape, (nimages, ntotal))
    assert_equal(p['basis_projections'].axes.coord_names,
                 ['i','j','k','t','PCA components'])
    assert_equal(p['basis_projections'].shape,
                 data.shape[:3] + (1, ncomp))


def img_res2pos1(res, bv_key):
    # Orient basis vectors in standard direction
    axis = res['axis']
    bvs = res[bv_key]
    bps_img = res['basis_projections']
    bps = bps_img.get_data()
    signs = np.sign(bvs[0])
    res[bv_key] = bvs * signs
    new_axes = [None] * bps.ndim
    n_comps = bps.shape[axis]
    new_axes[axis] = slice(0, n_comps)
    res['basis_projections'] = Image(bps * signs[new_axes],
                                     bps_img.coordmap)
    return res


def test_other_axes():
    # With a diagonal affine, we can do PCA on any axis
    ncomp = 5
    img = data_dict['fmridata']
    in_coords = list(img.axes.coord_names)
    img_data = img.get_data()
    for axis_no, axis_name in enumerate('ijkt'):
        p = pca_image(img, axis_name, ncomp=ncomp)
        n = img.shape[axis_no]
        bv_key = 'basis_vectors over ' + axis_name
        assert_equal(_rank(p), n - 1)
        assert_equal(p[bv_key].shape, (n, n - 1))
        # We get the expected data back
        dp = pca_array(img_data, axis_no, ncomp=ncomp)
        # We have to make sure the signs are the same; on Windows it seems the
        # signs can flip even between two runs on the same data
        pos_p = img_res2pos1(p, bv_key)
        pos_dp = res2pos1(dp)
        img_bps = pos_p['basis_projections']
        assert_almost_equal(pos_dp['basis_vectors'], pos_p[bv_key])
        assert_almost_equal(pos_dp['basis_projections'], img_bps.get_data())
        # And we've replaced the expected axis
        exp_coords = in_coords[:]
        exp_coords[exp_coords.index(axis_name)] = 'PCA components'
        assert_equal(img_bps.axes.coord_names, exp_coords)
    # If the affine is not diagonal, we'll get an error
    aff = from_matvec(np.arange(16).reshape(4,4))
    nd_cmap = AffineTransform(img.axes, img.reference, aff)
    nd_img = Image(img_data, nd_cmap)
    for axis_name in 'ijkt':
        assert_raises(AxisError, pca_image, nd_img, axis_name)
    # Only for the non-diagonal parts
    aff = np.array([[1, 2, 0, 0, 10],
                    [2, 1, 0, 0, 11],
                    [0, 0, 3, 0, 12],
                    [0, 0, 0, 4, 13],
                    [0, 0, 0, 0, 1]])
    nd_cmap = AffineTransform(img.axes, img.reference, aff)
    nd_img = Image(img_data, nd_cmap)
    for axis_name in 'ij':
        assert_raises(AxisError, pca_image, nd_img, axis_name)
    for axis_name in 'kt':
        p = pca_image(img, axis_name, ncomp=ncomp)
        exp_coords = in_coords[:]
        exp_coords[exp_coords.index(axis_name)] = 'PCA components'
        assert_equal(p['basis_projections'].axes.coord_names, exp_coords)
