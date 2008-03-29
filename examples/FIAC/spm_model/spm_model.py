__docformat__ = 'restructuredtext'

import gc, os, fpformat, time
                
import numpy as N
import numpy.linalg as L
from scipy.stats import f as FDbn

from neuroimaging.fixes.scipy.stats.models.regression import OLSModel

import neuroimaging.modalities.fmri.fmristat.utils as fmristat
from neuroimaging.modalities.fmri.regression import FContrastOutput

import model # FIAC.model

import correlation, reml

class GLSModel(OLSModel):

    """
    Generalized least squares model with a general covariance structure

    This should probably go into neuroimaging.fixes.scipy.stats.models.regression

    """

    def __init__(self, design, sigma):
        self.cholsigmainv = L.cholesky(L.pinv(sigma)).T
        super(GLSModel, self).__init__(design)

    def whiten(self, Y):
        return N.dot(self.cholsigmainv, Y)

class SPMFirstStage(fmristat.FmriStatOLS):
    """
    OLS pass of SPM: essentially the same as
    FmriStatOLS.

    Only difference is the setup of the iterator in the second stage,
    this is done in the 'getparcelmap' method.

    FmriStat assumes this will be an image of AR(1) coefficients, discretized
    at some level (usually rounded at resolution +- 0.01).

    The second pass of our SPM model fits a matrix constant across voxels.

    The easy thing to do is therefore to just make the parcelmap by slices.

    """
    
    def __init__(self, *args, **keywords):
        keywords['resid'] = True # we need the output for estimating
                                 # pooled covariance 
        fmristat.FmriStatOLS.__init__(self, *args, **keywords)

    def getparcelmap(self):
        """
        :Returns: TODO
        """
        shape = self.resid_output.img[:].shape[1:]
        parcelseq = range(shape[0])
        parcelmap = N.zeros(shape)
        for i in range(shape[0]):
            parcelmap[i] = i
        return parcelmap, parcelseq
            

class SPMSecondStage(fmristat.FmriStatAR):

    """
    The only difference between this an FmriStat is the model of covariance
    in the second stage. The pooled covariance matrix is retrieved from the
    OLS output.
    """
    

    def __init__(self, OLS, sigma, *args, **keywords):
        """
        
        Building on OLS results, fit the AR(1) model.
                
        """

        fmristat.FmriStatAR.__init__(self, OLS, *args, **keywords)
        self.firststage_sigma = sigma
        if not isinstance(OLS, SPMFirstStage):
            raise ValueError, 'expecting an SPMFirstStage object in SPMSecondStage'

        

    def model(self):
        """
        :Returns: `GLSModel`
        """

        design = self.dmatrix
        return GLSModel(design, self.firststage_sigma)

#-----------------------------------------------------------------------------#
#
# Run level model 
#
#-----------------------------------------------------------------------------#

class Run(model.Run):

    """
    Run specific model. Model formula is returned by formula method.
    """

    def __init__(self, *args, **keywords):
        keywords['resultdir'] = "spm"
        model.Run.__init__(self, *args, **keywords)

    def estimate_pooled_covariance(self, ARtarget=[0.3], pvalue=1.0e-04):
        """
        Use SPM's REML implementation to estimate a pooled covariance matrix.

        Thresholds an F statistic at a marginal pvalue to estimate
        covariance matrix.

        """
        resid = self.firststage_model.resid_output.img
        n = resid[:].shape[0]
        components = correlation.ARcomponents(ARtarget, n)

        F = self.Fcutoff.img
        dfF, dftot = self.Fcutoff.contrast.matrix.shape

        ## TODO check neuroimaging.fixes.scipy.stats.models.contrast to see if rank is
        ## correctly set -- I don't think it is right now.

        dfresid = resid.shape[0] - dftot
        thresh = FDbn.ppf(pvalue, dfF, dfresid)
        
        self.raw_sigma = 0
        nvox = 0
        for i in range(F.shape[0]):
            d = resid[:,i]
            d.shape = (d.shape[0], N.product(d.shape[1:]))
            keep = N.greater(F[i], thresh)
            keep.shape = N.product(keep.shape)
            d = d.compress(keep, axis=1)
            self.raw_sigma += N.dot(d, d.T)
            nvox += d.shape[1]
        self.raw_sigma /= nvox
        C, h, _ = reml.reml(self.raw_sigma,
                            components,
                            n=nvox)
        self.firststage_sigma = C
        self.reml_components = h
        C.tofile(file(os.path.join(self.resultdir, "covariance.bin"), 'w'), format="<f8")

    def OLS(self, **OLSopts):
        """
        OLS pass through data.
        """
        
        self.load()
        if self.normalize:
            OLSopts['normalize'] = self.brainavg

        overall = model.Contrast(self.experiment, self.formula,
                                 name='overallF-stage1')
        self.Fcutoff = FContrastOutput(self.fmri.grid, overall,
                                       path=self.resultdir, 
                                       frametimes=self.fmri.frametimes+
                                       self.shift,
                                       clobber=True)

        self.firststage_model = SPMFirstStage(self.fmri,
                                      formula=self.formula,
                                      mask=self.mask,
                                      tshift=self.shift, 
                                      path=self.resultdir,
                                      outputs=[self.Fcutoff],
                                      **OLSopts)

        self._setup_contrasts()
        self.firststage_model.reference = self.average
        
        toc = time.time()
        self.firststage_model.fit()
        tic = time.time()
        
        print 'OLS time', `tic-toc`
        
        self.estimate_pooled_covariance()

        self.clear()

    def AR(self, **ARopts):

        self.load()

        toc = time.time()
        self.secondstage_model = SPMSecondStage(self.firststage_model,
                                                self.firststage_sigma,
                                                contrasts=[self.overallF,
                                                           self.average,
                                                           self.speaker,
                                                           self.sentence,
                                                           self.interaction,
                                                           self.delays],
                                                tshift=self.shift,
                                                **ARopts)

        # delete OLS residuals

        os.remove(os.path.join(self.resultdir, "OLSresid.nii"))

        self.secondstage_model.fit()
        tic = time.time()
        
        self.clear()
        print 'AR time', `tic-toc`


def run(subj=3, run=3):
    """
    Run through a fit of FIAC data.
    """
    study = model.Study(root=model.io.data_path)
    subject = model.Subject(subj, study=study)
    runmodel = Run(subject, run)
    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)

if __name__ == "__main__":
    run()
