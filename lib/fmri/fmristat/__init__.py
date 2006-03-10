import gc
import enthought.traits as traits
from neuroimaging.statistics import iterators 
from neuroimaging.statistics.regression import OLSModel, ARModel
import neuroimaging.fmri as fmri
from neuroimaging.fmri.regression import AR1Output

class fMRIStatOLS(iterators.LinearModelIterator):

    """
    OLS pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()

    def __init__(self, fmri_image, outputs=[], **keywords):
        traits.HasTraits.__init__(self, outputs=outputs, **keywords)
        self.fmri_image = fmri.fMRIImage(fmri_image)
        self.iterator = iter(self.fmri_image)

        self.rho_estimator = AR1Output(self.fmri_image)
        self.outputs.append(self.rho_estimator)

    def model(self, **keywords):
        time = self.fmri_image.frametimes
        if self.slicetimes is not None:
            _slice = self.iterator.grid.itervalue.slice
            model = OLSModel(design=self.formula.design(time + self.slicetimes[_slice[1]]))
        else:
            model = OLSModel(design=self.dmatrix)
        return model

## class fMRIStatSession(fMRIStatIterator):

##     def firstpass(self, fwhm=8.0):

##         if self.slicetimes is None:
##             self.dmatrix = self.formula(self.img.frametimes)
##         rhoout = AROutput()
##         fwhmout = FWHMOutput(self.fmri)

##         glm = SessionGLM(self.fmri, self.design, outputs=[rhoout])#, fwhmout])
##         glm.fit(resid=True, norm_resid=True)

## #        kernel = LinearFilter3d(rhoout.image, fwhm=fwhm)
##         self.rho = rhoout.image
##         self.rho.tofile('rho.img')
##         self.fwhm = fwhmout.fwhmest.fwhm

##     def secondpass(self):

##         self.labels = floor(self.rho * 100.) / 100.
##         self.fmri = LabelledfMRIImage(self.fmri, self.labels)
        
##         glm = SessionGLM(self.fmri, self.design)
##         glm.fit(resid=True, norm_resid=True)


