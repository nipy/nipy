# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings

from nose.tools import assert_equal

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal,
                           assert_raises)
import numpy as np
from ....core.image.image_spaces import (make_xyz_image,
                                         xyz_affine)
from ..parcel_analysis import ParcelAnalysis

NSUBJ = 10
SIZE = (50, 50, 50)
AFFINE = np.diag(np.concatenate((np.random.rand(3), np.ones((1,)))))


def make_fake_data():
    con_imgs = [make_xyz_image(np.random.normal(0, 1, size=SIZE),
                               AFFINE, 'talairach') for i in range(NSUBJ)]
    parcel_img = make_xyz_image(np.random.randint(10, size=SIZE),
                                AFFINE, 'talairach')
    return con_imgs, parcel_img


def test_parcel_analysis():
    con_imgs, parcel_img = make_fake_data()
    g = ParcelAnalysis(con_imgs, parcel_img)
    t_map_img = g.t_map()
    assert_array_equal(t_map_img.shape, SIZE)
    assert_array_equal(xyz_affine(t_map_img), AFFINE)
    parcel_mu_img, parcel_prob_img = g.parcel_maps()
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
