from __future__ import absolute_import
from __future__ import print_function

# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import os
from nose.tools import assert_equal
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal,
                           assert_raises)
from ....core.image.image_spaces import (make_xyz_image,
                                         xyz_affine)
from ..parcel_analysis import (ParcelAnalysis, parcel_analysis,
                               _smooth_image_pair)


NSUBJ = 10
NLABELS = 10
SIZE = (50, 50, 50)
AFFINE = np.diag(np.concatenate((np.random.rand(3), np.ones((1,)))))


def test_smooth_image_pair():
    con_img = make_xyz_image(np.random.normal(0, 1, size=SIZE),
                             AFFINE, 'talairach')
    vcon_img = make_xyz_image(np.random.normal(0, 1, size=SIZE),
                             AFFINE, 'talairach')
    for sigma in (1, (1, 1.2, 0.8)):
        for method in ('default', 'spm'):
            scon_img, svcon_img = _smooth_image_pair(con_img, vcon_img,
                                                     sigma, method=method)
    assert_raises(ValueError, _smooth_image_pair, con_img, vcon_img, 1,
                  method='fsl')


def make_fake_data():
    con_imgs = [make_xyz_image(np.random.normal(0, 1, size=SIZE),
                               AFFINE, 'talairach') for i in range(NSUBJ)]
    parcel_img = make_xyz_image(np.random.randint(NLABELS, size=SIZE),
                                AFFINE, 'talairach')
    return con_imgs, parcel_img


def _test_parcel_analysis(smooth_method, parcel_info, vcon=False,
                          full_res=True):
    con_imgs, parcel_img = make_fake_data()
    if vcon:
        vcon_imgs = con_imgs
    else:
        vcon_imgs = None
    g = ParcelAnalysis(con_imgs, parcel_img,
                       vcon_imgs=vcon_imgs,
                       smooth_method=smooth_method,
                       parcel_info=parcel_info)
    t_map_img = g.t_map()
    assert_array_equal(t_map_img.shape, SIZE)
    assert_array_equal(xyz_affine(t_map_img), AFFINE)
    parcel_mu_img, parcel_prob_img = g.parcel_maps(full_res=full_res)
    assert_array_equal(parcel_mu_img.shape, SIZE)
    assert_array_equal(xyz_affine(parcel_mu_img), AFFINE)
    assert_array_equal(parcel_prob_img.shape, SIZE)
    assert_array_equal(xyz_affine(parcel_prob_img), AFFINE)
    assert parcel_prob_img.get_data().max() <= 1
    assert parcel_prob_img.get_data().min() >= 0
    outside = parcel_img.get_data() == 0
    assert_array_equal(t_map_img.get_data()[outside], 0)
    assert_array_equal(parcel_mu_img.get_data()[outside], 0)
    assert_array_equal(parcel_prob_img.get_data()[outside], 0)


def test_parcel_analysis():
    parcel_info = (list(range(NLABELS)), list(range(NLABELS)))
    _test_parcel_analysis('default', parcel_info)


def test_parcel_analysis_nonstandard():
    _test_parcel_analysis('default', None, vcon=True, full_res=False)


def test_parcel_analysis_spm():
    _test_parcel_analysis('spm', None)


def test_parcel_analysis_nosmooth():
    con_imgs, parcel_img = make_fake_data()
    msk_img = make_xyz_image(np.ones(SIZE, dtype='uint'),
                             AFFINE, 'talairach')
    X = np.random.normal(0, 1, size=(NSUBJ, 5))
    c = np.random.normal(0, 1, size=(5,))
    g = ParcelAnalysis(con_imgs, parcel_img,
                       msk_img=msk_img,
                       design_matrix=X,
                       cvect=c,
                       fwhm=0)
    t_map = g.t_map().get_data()
    m_error = np.abs(np.mean(t_map))
    v_error = np.abs(np.var(t_map) - (NSUBJ - 5) / float(NSUBJ - 7))
    print('Errors: %f (mean), %f (var)' % (m_error, v_error))
    assert m_error < .1
    assert v_error < .1


def _test_parcel_analysis_error(**kw):
    con_imgs, parcel_img = make_fake_data()
    return ParcelAnalysis(con_imgs, parcel_img, **kw)


def test_parcel_analysis_error():
    assert_raises(ValueError, _test_parcel_analysis_error,
                  vcon_imgs=list(range(NSUBJ + 1)))
    assert_raises(ValueError, _test_parcel_analysis_error,
                  cvect=np.ones(1))
    assert_raises(ValueError, _test_parcel_analysis_error,
                  design_matrix=np.random.rand(NSUBJ, 2))
    assert_raises(ValueError, _test_parcel_analysis_error,
                  design_matrix=np.random.rand(NSUBJ + 1, 2),
                  cvect=np.ones(2))
    assert_raises(ValueError, _test_parcel_analysis_error,
                  design_matrix=np.random.rand(NSUBJ, 2),
                  cvect=np.ones(3))


def test_parcel_analysis_write_mode():
    # find a subdirectory name that doesn't exist to check that
    # attempts to write in a non-existing directory do not raise
    # errors
    con_imgs, parcel_img = make_fake_data()
    subdirs = [o for o in os.listdir('.') if os.path.isdir(o)]
    res_path = 'a'
    while res_path in subdirs:
        res_path += 'a'
    p = ParcelAnalysis(con_imgs, parcel_img, res_path=res_path,
                       write_smoothed_images=True)
    assert_raises(IOError, p.dump_results)
    _ = p.t_map()
    _ = p.parcel_maps()


def test_parcel_analysis_function():
    con_imgs, parcel_img = make_fake_data()
    parcel_mu_img, parcel_prob_img = parcel_analysis(con_imgs, parcel_img)
    assert_array_equal(parcel_mu_img.shape, SIZE)
    assert_array_equal(xyz_affine(parcel_mu_img), AFFINE)
    assert_array_equal(parcel_prob_img.shape, SIZE)
    assert_array_equal(xyz_affine(parcel_prob_img), AFFINE)
    assert parcel_prob_img.get_data().max() <= 1
    assert parcel_prob_img.get_data().min() >= 0
    outside = parcel_img.get_data() == 0
    assert_array_equal(parcel_mu_img.get_data()[outside], 0)
    assert_array_equal(parcel_prob_img.get_data()[outside], 0)
