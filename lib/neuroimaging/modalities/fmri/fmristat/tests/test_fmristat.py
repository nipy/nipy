import os, gc, shutil

import numpy as N
from scipy.sandbox.models.contrast import Contrast

from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.utils.tests.data import repository
from neuroimaging.modalities.fmri.fmri import fMRIImage
from neuroimaging.modalities.fmri.protocol import ExperimentalFactor,\
  ExperimentalQuantitative
from neuroimaging.modalities.fmri.functions import SplineConfound
from neuroimaging.modalities.fmri.fmristat.utils import fMRIStatAR, fMRIStatOLS
from  neuroimaging.core.api import Image
from neuroimaging.modalities.fmri.hrf import glover, glover_deriv

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()

class test_fMRIStat(NumpyTestCase):

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

    def data_setUp(self):
        frametimes = N.arange(120)*3.
        slicetimes = N.array([0.14, 0.98, 0.26, 1.10, 0.38, 1.22, 0.50, 1.34, 0.62, 1.46, 0.74, 1.58, 0.86])

        self.img = fMRIImage("test_fmri.hdr", datasource=repository, frametimes=frametimes,
                                  slicetimes=slicetimes, usematfile=False)

    def tearDown(self):
        shutil.rmtree('fmristat_run', ignore_errors=True)
        for rhofile in ['rho.hdr', 'rho.img']:
            shutil.rmtree(rhofile, ignore_errors=True)

class test_SliceTimes(test_fMRIStat):

    @slow
    @data
    def test_model_slicetimes(self):
        OLS = fMRIStatOLS(self.img, self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.nmax = 75
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = fMRIStatAR(OLS)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_Resid1(test_fMRIStat):

    @slow
    @data
    def test_model_resid1(self):
        self.img.slicetimes = None
        OLS = fMRIStatOLS(self.img, self.formula, path=".", clobber=True,
                                   slicetimes=self.img.slicetimes, resid=True)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = fMRIStatAR(OLS, clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_Resid2(test_fMRIStat):

    @slow
    @data
    def test_model_resid2(self):
        self.img.slicetimes = None
        OLS = fMRIStatOLS(self.img, self.formula, path=".", clobber=True,
                                   slicetimes=self.img.slicetimes)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = fMRIStatAR(OLS, resid=True, clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

class test_HRFDeriv(test_fMRIStat):

    @slow
    @data
    def test_hrf_deriv(self):
        self.IRF = glover_deriv

        self.pain.convolve(self.IRF)
        self.pain.convolved = True

        self.formula = self.pain + self.drift

       
        pain = Contrast(self.pain, self.formula, name='hot-warm')
        self.img.slicetimes = None
        OLS = fMRIStatOLS(self.img, self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)

        AR = fMRIStatAR(OLS, contrasts=[pain], clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()
        
class test_Contrast(test_fMRIStat):

    @slow
    @data
    def test_contrast(self):
        pain = Contrast(self.pain, self.formula, name='pain')

        self.img.slicetimes = None
        OLS = fMRIStatOLS(self.img, self.formula, 
                                   slicetimes=self.img.slicetimes,
                                   clobber=True)
        OLS.fit()
        rho = OLS.rho_estimator.img
        rho.tofile('rho.hdr', clobber=True)
        
        if PYLAB_DEF:
            from neuroimaging.ui.visualization import viewer
            v=viewer.BoxViewer(rho)
            v.draw()


        AR = fMRIStatAR(OLS, contrasts=[pain], clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

        t = Image('fmristat_run/contrasts/pain/F.hdr')
        if PYLAB_DEF:
            v=viewer.BoxViewer(t)
            v.draw()

if __name__ == '__main__':
    NumpyTest.run()
