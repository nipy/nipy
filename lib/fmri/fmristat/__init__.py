import gc
import enthought.traits as traits
from neuroimaging.statistics import iterators 
from neuroimaging.statistics.regression import OLSModel, ARModel
import neuroimaging.fmri as fmri
import neuroimaging.image.kernel_smooth as kernel_smooth
from neuroimaging.fmri.regression import AR1Output, TContrastOutput, FContrastOutput
import numpy as N

class fMRIStatOLS(iterators.LinearModelIterator):

    """
    OLS pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()
    fwhm = traits.Float(6.0)
    nmax = traits.Int(200) # maximum number of rho values

    def __init__(self, fmri_image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fmri.fMRIImage(fmri_image)
        self.iterator = iter(self.fmri_image)

        self.rho_estimator = AR1Output(self.fmri_image)
        self.outputs.append(self.rho_estimator)
        self.dmatrix = self.formula.design(time=self.fmri_image.frametimes)

    def model(self, **keywords):
        time = self.fmri_image.frametimes
        if self.slicetimes is not None:
            _slice = self.iterator.grid.itervalue.slice
            model = OLSModel(design=self.formula.design(time=time + self.slicetimes[_slice[1]]))
        else:
            model = OLSModel(design=self.dmatrix)
        return model

    def fit(self, **keywords):
        iterators.LinearModelIterator.fit(self, **keywords)

        smoother = kernel_smooth.LinearFilter(self.rho_estimator.img.grid, fwhm=self.fwhm)
        self.rho = smoother.smooth(self.rho_estimator.img)
        self.getlabels()

    def getlabels(self):
        if self.slicetimes == None:
            tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
            tmp.shape = N.product(tmp.shape)
            self.labels = tmp
            self.labelset = list(N.unique(tmp))
            del(tmp); gc.collect()
        else:
            self.labels = []
            self.labelset = []
            for i in range(self.rho.grid.shape[0]):
                tmp = self.rho.getslice(slice(i,i+1))
                tmp.shape = N.product(tmp.shape)
                tmp = N.around(tmp * (self.nmax / 2.)) / (self.nmax / 2.)
                newlabels = list(N.unique(tmp))
                self.labelset += [newlabels]
                self.labels += [tmp]

class fMRIStatAR(iterators.LinearModelIterator):

    """
    AR(1) pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()
    fwhm = traits.Float(6.0)
    path = traits.Str('.')

    def __init__(self, OLS, contrasts=None, **keywords):
        """
        Building on OLS results, fit the AR(1) model.

        Contrasts is a sequence of terms to be tested in the model.

        """
        
        traits.HasTraits.__init__(self, **keywords)
        if not isinstance(OLS, fMRIStatOLS):
            raise ValueError, 'expecting an fMRIStatOLS object in fMRIStatAR'
        self.fmri_image = OLS.fmri_image
        
        # copy the formula
        
        self.slicetimes = OLS.slicetimes
        time = self.fmri_image.frametimes
        self.formula = OLS.formula
        if self.slicetimes is None:
            self.dmatrix = OLS.dmatrix
            self.fmri_image.grid.itertype = 'parcel'
        else:
            self.fmri_image.grid.itertype = 'slice/parcel'
            self.designs = []
            for i in range(len(self.slicetimes)):
                self.designs.append(self.formula.design(time=time + self.slicetimes[i]))

        self.contrasts = {}
        if contrasts is not None:
            if type(contrasts) not in [type([]), type(())]:
                contrasts = [contrasts]
            for i in range(len(contrasts)):
                contrasts[i].getmatrix(time=self.fmri_image.frametimes)
                if contrasts[i].rank == 1:
                    cur = TContrastOutput(self.fmri_image, contrasts[i], path=self.path)
                else:
                    cur = FContrastOutput(self.fmri_image, contrasts[i], path=self.path)
                self.contrasts[contrasts[i].name] = cur
                
        # setup the iterator

        self.fmri_image.grid.labels = OLS.labels
        self.fmri_image.grid.labelset = OLS.labelset

        self.iterator = iter(self.fmri_image)
        self.j = 0

        self.outputs += self.contrasts.values()

    def model(self, **keywords):
        self.j += 1
        time = self.fmri_image.frametimes
        if self.slicetimes is not None:
            rho = self.iterator.grid.itervalue.label
            i = self.iterator.grid.itervalue.slice[1]
            model = ARModel(rho=rho, design=self.designs[i])
        else:
            rho = self.iterator.grid.itervalue.label
            model = ARModel(rho=rho, design=self.dmatrix)
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


