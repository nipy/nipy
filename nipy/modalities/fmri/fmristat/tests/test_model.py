# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nipy.algorithms.statistics.formula.formulae import Formula, Term, make_recarray
from nipy.algorithms.statistics.models.regression import (
    OLSModel,
    ar_bias_correct,
)
from nipy.core.image.image import rollimg
from nipy.io.api import load_image
from nipy.testing import anatfile, funcfile

from ...api import FmriImageList
from .. import model
from ..model import ModelOutputImage, estimateAR

FUNC_IMG = load_image(funcfile)
FUNC_LIST = FmriImageList.from_image(FUNC_IMG, volume_start_times=2.)


def test_model_out_img(in_tmp_path):
    # Model output image
    cmap = load_image(anatfile).coordmap
    shape = (2,3,4)
    fname = 'myfile.nii'
    moi = ModelOutputImage(fname, cmap, shape)
    for i in range(shape[0]):
        moi[i] = i
    for i in range(shape[0]):
        assert_array_equal(moi[i], i)
    moi.save()
    pytest.raises(ValueError, moi.__setitem__, 0, 1)
    pytest.raises(ValueError, moi.__getitem__, 0)
    new_img = load_image(fname)
    for i in range(shape[0]):
        assert_array_equal(new_img[i].get_fdata(), i)
    del new_img


def example_formula():
    time_vector = make_recarray(FUNC_LIST.volume_start_times, 't')
    t = Term('t')
    con_defs = {'c': t, 'c2': t+t**2}
    # Formula - with an intercept
    f = Formula([t, t**2, t**3, 1])
    # Design matrix and contrasts
    desmtx, cmatrices = f.design(time_vector, contrasts=con_defs)
    return f, desmtx, cmatrices


def test_model_inputs():
    f, _, cmatrices = example_formula()
    start_times = FUNC_LIST.volume_start_times
    for MC, kwargs in ((model.OLS, {}), (model.AR1, {'rho': FUNC_LIST[0]})):
        # This works correctly
        m = MC(FUNC_LIST, f, **kwargs)
        assert np.all(m.volume_start_times == start_times)
        # Need volume_start_times for image.
        pytest.raises(ValueError, MC, FUNC_IMG, f, **kwargs)
        # With timevector.
        m = MC(FUNC_IMG, f, outputs=[], volume_start_times=start_times,
               **kwargs)
        assert np.all(m.volume_start_times == start_times)


@pytest.mark.parametrize("imp_img, kwargs", ((FUNC_LIST, {}),
                                             (rollimg(FUNC_IMG, 't'),
                                              {'volume_start_times':
                                               FUNC_LIST.volume_start_times})))
def test_run(in_tmp_path, imp_img, kwargs):
    ar1_fname = 'ar1_out.nii'
    f, _, cmatrices = example_formula()

    # Run OLS model
    outputs = []
    outputs.append(model.output_AR1(ar1_fname, imp_img))
    outputs.append(model.output_resid('resid_OLS_out.nii', imp_img))
    ols = model.OLS(imp_img, f, outputs, **kwargs)
    ols.execute()
    # Run AR1 model
    outputs = []
    outputs.append(
        model.output_T('T_out.nii', cmatrices['c'], imp_img))
    outputs.append(
        model.output_F('F_out.nii', cmatrices['c2'], imp_img))
    outputs.append(
        model.output_resid('resid_AR_out.nii', imp_img))
    rho = load_image(ar1_fname)
    ar = model.AR1(imp_img, f, rho, outputs, **kwargs)
    ar.execute()
    f_img = load_image('F_out.nii')
    assert f_img.shape == FUNC_IMG.shape[:-1]
    f_data = f_img.get_fdata()
    assert np.all((f_data>=0) & (f_data<30))
    resid_img = load_image('resid_AR_out.nii')
    assert resid_img.shape == FUNC_IMG.shape
    assert_array_almost_equal(np.mean(resid_img.get_fdata()), 0, 3)
    e_img = load_image('T_out_effect.nii')
    sd_img = load_image('T_out_sd.nii')
    t_img = load_image('T_out_t.nii')
    t_data = t_img.get_fdata()
    assert_array_almost_equal(t_data,
                              e_img.get_fdata() / sd_img.get_fdata())
    assert np.all(np.abs(t_data) < 6)
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
    assert rhos.shape == (2,)
    assert np.all(np.abs(rhos <= 1))
    # standard routine
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
    # Make 2D and 3D Y
    Y = rng.normal(size=(N,4)) * 10 + 100
    results = my_model.fit(Y)
    rhos = estimateAR(results.resid, my_model.design, order=2)
    assert rhos.shape == (2,4)
    assert np.all(np.abs(rhos <= 1))
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
    # 3D
    results.resid = np.reshape(results.resid, (N,2,2))
    rhos = estimateAR(results.resid, my_model.design, order=2)
    assert rhos.shape == (2,2,2)
    assert np.all(np.abs(rhos <= 1))
    rhos2 = ar_bias_correct(results, 2)
    assert_array_almost_equal(rhos, rhos2, 8)
