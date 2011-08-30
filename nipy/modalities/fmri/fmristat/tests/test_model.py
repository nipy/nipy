# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings

from nipy.io.api import load_image

from  .. import model
from ...api import FmriImageList
from ...formula import Formula, Term, make_recarray
from nibabel.tmpdirs import InTemporaryDirectory

from nipy.testing import funcfile


# FIXME: This does many things, but it does not test any values
# with asserts.
def test_run():
    ar1_fname = 'ar1_out.nii'
    funcim = load_image(funcfile)
    fmriims = FmriImageList.from_image(funcim, volume_start_times=2.)
    # Formula - with no intercept
    t = Term('t')
    f = Formula([t, t**2, t**3])
    # Design matrix and contrasts
    time_vector = make_recarray(fmriims.volume_start_times, 't')
    con_defs = dict(c=t, c2=t+t**2)
    desmtx, cmatrices = f.design(time_vector, contrasts=con_defs)

    with InTemporaryDirectory():
        # Run OLS model
        outputs = []
        outputs.append(model.output_AR1(ar1_fname, fmriims))
        outputs.append(model.output_resid('resid_OLS_out.nii', fmriims))
        ols = model.OLS(fmriims, f, outputs)
        ols.execute()

        outputs = []
        outputs.append(model.output_T('T_out.nii', cmatrices['c'], fmriims))
        outputs.append(model.output_F('F_out.nii', cmatrices['c2'], fmriims))
        outputs.append(model.output_resid('resid_AR_out.nii', fmriims))
        rho = load_image(ar1_fname)
        ar = model.AR1(fmriims, f, rho, outputs)
        ar.execute()
