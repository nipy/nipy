# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from nipy.io.api import load_image
from nipy.core.image.image import rollimg

from  .. import model
from ..model import ModelOutputImage, estimateAR
from ...api import FmriImageList

from nipy.algorithms.statistics.models.regression import (
    OLSModel, ar_bias_corrector, ar_bias_correct)
from nipy.algorithms.statistics.formula.formulae import(
    Formula, Term, make_recarray)


from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import assert_raises, assert_true, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nipy.testing import funcfile, anatfile

def test_model_out_img():
    # Model output image
    cmap = load_image(anatfile).coordmap
    shape = (2,3,4)
    fname = 'myfile.nii'
    with InTemporaryDirectory():
        moi = ModelOutputImage(fname, cmap, shape)
        for i in range(shape[0]):
            moi[i] = i
        for i in range(shape[0]):
            assert_array_equal(moi[i], i)
        moi.save()
        assert_raises(ValueError, moi.__setitem__, 0, 1)
        assert_raises(ValueError, moi.__getitem__, 0)
        new_img = load_image(fname)
        for i in range(shape[0]):
            assert_array_equal(new_img[i].get_data(), i)
        del new_img


def test_run():
    ar1_fname = 'ar1_out.nii'
    funcim = load_image(funcfile)
    fmriims = FmriImageList.from_image(funcim, volume_start_times=2.)
    one_vol = fmriims[0]
    # Formula - with an intercept
    t = Term('t')
    f = Formula([t, t**2, t**3, 1])
    # Design matrix and contrasts
    time_vector = make_recarray(fmriims.volume_start_times, 't')
    con_defs = dict(c=t, c2=t+t**2)
    desmtx, cmatrices = f.design(time_vector, contrasts=con_defs)

    # Run with Image and ImageList
    for inp_img in (rollimg(funcim, 't'), fmriims):
        with InTemporaryDirectory():
            # Run OLS model
            outputs = []
            outputs.append(model.output_AR1(ar1_fname, fmriims))
            outputs.append(model.output_resid('resid_OLS_out.nii', fmriims))
            ols = model.OLS(fmriims, f, outputs)
            ols.execute()
            # Run AR1 model
            outputs = []
            outputs.append(
                model.output_T('T_out.nii', cmatrices['c'], fmriims))
            outputs.append(
                model.output_F('F_out.nii', cmatrices['c2'], fmriims))
            outputs.append(
                model.output_resid('resid_AR_out.nii', fmriims))
            rho = load_image(ar1_fname)
            ar = model.AR1(fmriims, f, rho, outputs)
            ar.execute()
            f_img = load_image('F_out.nii')
            assert_equal(f_img.shape, one_vol.shape)
            f_data = f_img.get_data()
            assert_true(np.all((f_data>=0) & (f_data<30)))
            resid_img = load_image('resid_AR_out.nii')
            assert_equal(resid_img.shape, funcim.shape)
            assert_array_almost_equal(np.mean(resid_img.get_data()), 0, 3)
            e_img = load_image('T_out_effect.nii')
            sd_img = load_image('T_out_sd.nii')
            t_img = load_image('T_out_t.nii')
            t_data = t_img.get_data()
            assert_array_almost_equal(t_data,
                                      e_img.get_data() / sd_img.get_data())
            assert_true(np.all(np.abs(t_data) < 6))
            # Need to delete to help windows delete temporary files
            del rho, resid_img, f_img, e_img, sd_img, t_img, f_data, t_data


def test_ar_modeling():
    # Compare against standard routines
    rng = np.random.RandomState(20110903)
    N = 10
    Y = rng.normal(size=(N,1)) * 10 + 100
    X = np.c_[np.linspace(-1,1,N), np.ones((N,))]
    my_model = OLSModel(X)
    results = my_model.fit(Y)
    # fmristat wrapper
    rhos = estimateAR(results.resid, my_model.design, order=2)
    assert_equal(rhos.shape, (2,))
    assert_true(np.all(np.abs(rhos <= 1)))
    # standard routine
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
    # Make 2D and 3D Y
    Y = rng.normal(size=(N,4)) * 10 + 100
    results = my_model.fit(Y)
    rhos = estimateAR(results.resid, my_model.design, order=2)
    assert_equal(rhos.shape, (2,4))
    assert_true(np.all(np.abs(rhos <= 1)))
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
    # 3D
    results.resid = np.reshape(results.resid, (N,2,2))
    rhos = estimateAR(results.resid, my_model.design, order=2)
    assert_equal(rhos.shape, (2,2,2))
    assert_true(np.all(np.abs(rhos <= 1)))
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
