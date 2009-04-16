import os, gc, shutil

import numpy as np

from nipy.testing import *

from nipy.utils.tests.data import repository

from  nipy.core.api import Image
from nipy.fixes.scipy.stats.models.contrast import Contrast

from nipy.modalities.fmri.api import FmriImageList
from nipy.modalities.fmri.protocol import ExperimentalFactor,\
  ExperimentalQuantitative
from nipy.modalities.fmri.functions import SplineConfound
from nipy.modalities.fmri.hrf import glover, glover_deriv

# FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
#from nipy.modalities.fmri.fmristat.utils import FmriStatAR, FmriStatOLS

class test_FmriStat(TestCase):

    def setup_formula(self):

        on = [9.0, 27.0, 45.0, 63.0, 81.0, 99.0, 117.0,
              135.0, 153.0, 171.0, 189.0, 207.0, 225.0,
              243.0, 261.0, 279.0, 297.0, 315.0, 333.0, 351.0]
        off = [18.0, 36.0, 54.0, 72.0, 90.0, 108.0, 126.0,
               144.0, 162.0, 180.0, 198.0, 216.0, 234.0,
               252.0, 270.0, 288.0, 306.0, 324.0, 342.0, 360.0]
        p = ['hot', 'warm'] * 10
        all = []
        for i in range(20):
            all.append([p[i], on[i], off[i]])
        pain = ExperimentalFactor('pain', all, delta=False)
        drift_fn = SplineConfound(window=(0,360), df=7)
        drift = ExperimentalQuantitative('drift', drift_fn)
        self.pain = pain
        self.drift = drift

        self.IRF = glover

        self.pain.convolve(self.IRF)
        self.formula = self.pain + self.drift

    def setUp(self):
        self.setup_formula()

    # FIXME: data_setUp is never called!  As a result, self.img is not
    #     defined.  Need to verify the data file exists (try
    #     test_fmri.nii.gz) and the calling convention for FmriImageList
    def data_setUp(self):
        volume_start_times = np.arange(120)*3.
        slicetimes = np.array([0.14, 0.98, 0.26, 1.10, 0.38, 1.22, 0.50, 1.34, 0.62, 1.46, 0.74, 1.58, 0.86])

        self.img = FmriImageList("test_fmri.hdr", datasource=repository, volume_start_times=volume_start_times,
                                  slicetimes=slicetimes, usematfile=False)

    def tearDown(self):
        # FIXME: Use NamedTemporaryFile (import from tempfile) instead
        # of specific temporary files that need to be removed via
        # shutl.rmtree.
        shutil.rmtree('fmristat_run', ignore_errors=True)
        for rhofile in ['rho.hdr', 'rho.img']:
            shutil.rmtree(rhofile, ignore_errors=True)

class test_SliceTimes(test_FmriStat):
    # FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
    @dec.knownfailure
    @dec.slow
    @dec.data
    def test_model_slicetimes(self):
        OLS = FmriStatOLS(self.img, self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.nmax = 75
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = FmriStatAR(OLS)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_Resid1(test_FmriStat):
    # FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
    @dec.knownfailure    
    @dec.slow
    @dec.data
    def test_model_resid1(self):
        self.img.slicetimes = None
        OLS = FmriStatOLS(self.img, self.formula, path=".", clobber=True,
                                   slicetimes=self.img.slicetimes, resid=True)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = FmriStatAR(OLS, clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_Resid2(test_FmriStat):
    # FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
    @dec.knownfailure
    @dec.slow
    @dec.data
    def test_model_resid2(self):
        self.img.slicetimes = None
        OLS = FmriStatOLS(self.img, self.formula, path=".", clobber=True,
                                   slicetimes=self.img.slicetimes)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = FmriStatAR(OLS, resid=True, clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_HRFDeriv(test_FmriStat):
    # FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
    @dec.knownfailure
    @dec.slow
    @dec.data
    def test_hrf_deriv(self):
        self.IRF = glover_deriv

        self.pain.convolve(self.IRF)
        self.pain.convolved = True
        
        self.formula = self.pain + self.drift

       
        pain = Contrast(self.pain, self.formula, name='hot-warm')
        self.img.slicetimes = None
        OLS = FmriStatOLS(self.img, self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = FmriStatAR(OLS, contrasts=[pain], clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()
        
class test_Contrast(test_FmriStat):
    # FIXME: FmriStatOLS and FmriStatAR _not_ undefined!
    @dec.knownfailure
    @dec.slow
    @dec.data
    def test_contrast(self):
        pain = Contrast(self.pain, self.formula, name='pain')

        self.img.slicetimes = None
        OLS = FmriStatOLS(self.img, self.formula, 
                                   slicetimes=self.img.slicetimes,
                                   clobber=True)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)
        
        from nipy.ui.visualization import viewer
        v=viewer.BoxViewer(rho)
        v.draw()

        AR = FmriStatAR(OLS, contrasts=[pain], clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

        t = Image('fmristat_run/contrasts/pain/F.hdr')
        v=viewer.BoxViewer(t)
        v.draw()






