import unittest, scipy, sets, os, gc, shutil
import neuroimaging.fmri as fmri
import neuroimaging.fmri.protocol as protocol
import neuroimaging.fmri.fmristat as fmristat
from neuroimaging.statistics import contrast, utils
import neuroimaging.image as image
import numpy as N
import neuroimaging.fmri.hrf as hrf
import pylab

class fMRIStatTest(unittest.TestCase):

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
        pain = protocol.ExperimentalFactor('pain', all)
        drift_fn = protocol.SplineConfound(window=[0,360], df=7)
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        self.pain = pain
        self.drift = drift

        self.IRF = hrf.HRF()

        self.pain.convolve(self.IRF)
        self.formula = self.pain + self.drift

    def setUp(self):
        self.url = 'http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img'
        
        frametimes = N.arange(120)*3.
        slicetimes = N.array([0.14, 0.98, 0.26, 1.10, 0.38, 1.22, 0.50, 1.34, 0.62, 1.46, 0.74, 1.58, 0.86])
        self.img = fmri.fMRIImage(self.url, frametimes=frametimes,
                                  slicetimes=slicetimes, usematfile=False)

        self.setup_formula()

    def tearDown(self):
        shutil.rmtree('fmristat_run', ignore_errors=True)
                      
    def test_model_slicetimes(self):
        OLS = fmristat.fMRIStatOLS(self.img, formula=self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.nmax = 75
        OLS.fit(resid=True)
        rho = OLS.rho_estimator.img
        rho.tofile('rho.img')
        os.remove('rho.img')
        os.remove('rho.hdr')

        AR = fmristat.fMRIStatAR(OLS)
        AR.fit()
        del(OLS); del(AR); gc.collect()

    def test_model_resid1(self):
        self.img.slicetimes = None
        OLS = fmristat.fMRIStatOLS(self.img, formula=self.formula,
                                   slicetimes=self.img.slicetimes, resid=True)
        OLS.fit(resid=True)
        rho = OLS.rho_estimator.img
        rho.tofile('rho.img')
        os.remove('rho.img')
        os.remove('rho.hdr')

        AR = fmristat.fMRIStatAR(OLS)
        AR.fit()
        del(OLS); del(AR); gc.collect()

    def test_model_resid2(self):
        self.img.slicetimes = None
        OLS = fmristat.fMRIStatOLS(self.img, formula=self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.fit(resid=True)
        rho = OLS.rho_estimator.img
        rho.tofile('rho.img')
        os.remove('rho.img')
        os.remove('rho.hdr')

        AR = fmristat.fMRIStatAR(OLS, resid=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

    def test_hrf_deriv(self):
        self.IRF = hrf.HRF(deriv=True)

        self.pain.convolve(self.IRF)
        self.pain.convolved = True

        self.formula = self.pain + self.drift

       
        pain = contrast.Contrast(self.pain, self.formula, name='hot-warm')
        self.img.slicetimes = None
        OLS = fmristat.fMRIStatOLS(self.img, formula=self.formula,
                                   slicetimes=self.img.slicetimes)
        OLS.fit(resid=True)
        rho = OLS.rho_estimator.img
        rho.tofile('rho.img')
        
        os.remove('rho.img')
        os.remove('rho.hdr')

        AR = fmristat.fMRIStatAR(OLS, contrasts=[pain])
        AR.fit()
        del(OLS); del(AR); gc.collect()
        

    def test_contrast(self):
        pain = contrast.Contrast(self.pain, self.formula, name='pain')

        self.img.slicetimes = None
        OLS = fmristat.fMRIStatOLS(self.img, formula=self.formula,
                                   slicetimes=self.img.slicetimes,
                                   clobber=True)
        OLS.fit(resid=True)
        rho = OLS.rho_estimator.img
        rho.tofile('rho.img', clobber=True)
        
        from neuroimaging.visualization import viewer
        v=viewer.BoxViewer(rho)
        v.draw()
        #pylab.show()

        os.remove('rho.img')
        os.remove('rho.hdr')

        AR = fmristat.fMRIStatAR(OLS, contrasts=[pain], clobber=True)
        AR.fit()
        del(OLS); del(AR); gc.collect()

        t = image.Image('fmristat_run/contrasts/pain/F.img')
        v=viewer.BoxViewer(t)
        v.draw()
        #pylab.show()

def suite():
    suite = unittest.makeSuite(fMRIStatTest)
    return suite


if __name__ == '__main__':
    unittest.main()
