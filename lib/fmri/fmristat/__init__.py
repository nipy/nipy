import gc, os, string, fpformat
import enthought.traits as traits
from neuroimaging.statistics import iterators, utils 
from neuroimaging.statistics.regression import OLSModel, ARModel
import neuroimaging.fmri as fmri
import neuroimaging.image.kernel_smooth as kernel_smooth
from neuroimaging.fmri.regression import AR1Output, TContrastOutput, FContrastOutput, ResidOutput
import numpy as N
import numpy.linalg as L
import numpy.random as R

from delay import DelayContrast, DelayContrastOutput

try:
    import pylab
    from neuroimaging.fmri.plotting import MultiPlot
    canplot = True
except:
    canplot = False
    pass

class fMRIStatOLS(iterators.LinearModelIterator):

    """
    OLS pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()
    fwhm = traits.Float(6.0)
    nmax = traits.Int(200) # maximum number of rho values
    mask = traits.Any()
    path = traits.String('fmristat_run')
    resid = traits.false

    def __init__(self, fmri_image, outputs=[], **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fmri.fMRIImage(fmri_image)
        self.iterator = iter(self.fmri_image)

        self.rho_estimator = AR1Output(self.fmri_image)
        self.outputs += outputs
        self.outputs.append(self.rho_estimator)
        self.dmatrix = self.formula.design(time=self.fmri_image.frametimes)

        if self.resid:
            self.resid_output = ResidOutput(self.fmri_image, path=self.path, basename='OLSresid')
            self.outputs.append(self.resid_output)
            
        self.setup_output()
        
    def model(self, **keywords):
        time = self.fmri_image.frametimes
        if self.slicetimes is not None:
            _slice = self.iterator.grid.itervalue.slice
            model = OLSModel(design=self.formula.design(time=time + self.slicetimes[_slice[1]]))
        else:
            model = OLSModel(design=self.dmatrix)
        return model

    def fit(self, reference=None, **keywords):

        iterators.LinearModelIterator.fit(self, **keywords)

        smoother = kernel_smooth.LinearFilter(self.rho_estimator.img.grid, fwhm=self.fwhm)
        self.rho = smoother.smooth(self.rho_estimator.img)
        self.getlabels()

    def getlabels(self):

        val = R.standard_normal() # I take almost surely seriously....

        if self.mask is not None:
            _mask = self.mask.readall()
            self.rho.image.data = N.where(_mask, self.rho.image.data, val)

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

        rval = N.around(val * (self.nmax / 2.)) / (self.nmax / 2.)
        try:
            self.labelset.pop(self.labelset.index(rval))
        except:
            pass

    def setup_output(self):

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ftime = self.fmri_image.frametimes
        dmatrix = self.dmatrix.astype('<f8')
        ftime.shape = (ftime.shape[0],1)
        dmatrix = N.hstack([ftime, dmatrix])
        ftime.shape = (ftime.shape[0],)

        outname = os.path.join(self.path, 'matrix.csv')
        outfile = file(outname, 'w')
        outfile.write(string.join([fpformat.fix(x,4) for x in dmatrix], ',') + '\n')
        outfile.close()

        outname = os.path.join(self.path, 'matrix.bin')
        outfile = file(outname, 'w')
        dmatrix = self.dmatrix.astype('<f8')
        dmatrix.tofile(outfile)
        outfile.close()

        if canplot:
            ftime = self.fmri_image.frametimes

            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.formula, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for design matrix')
            pl.draw()
            pylab.savefig(os.path.join(self.path, 'matrix.png'))
            f.clf()


    def estimateFWHM_AR(self, reference,
                        fwhm_data=10., ARorder=1, df_target=100.):
        """
        Estimate smoothing of AR coefficient to get
        a targeted df.

        Worsley, K.J. (2005). \'Spatial smoothing of autocorrelations to control the degrees of freedom in fMRI analysis.\' NeuroImage, 26:635-641.

        """

        reference.getmatrix(time=self.fmri_image.frametimes)

        x = N.dot(N.transpose(L.generalized_inverse(self.dmatrix)),
                  reference.matrix)

        def aclag(x, j):
            return N.add.reduce(x[j:] * x[0:-j]) / N.add.reduce(x**2)

        tau = 0.
        for j in range(ARorder):
            tau = tau + aclag(x, j+1)**2

        ndim = len(self.fmri_image.shape) - 1

        def df_eff(fwhm_filter):
            f = N.pow(1 + 2. * (fwhm_filter / fwhm_data)**2, -ndim/2.)
            return utils.rank(self.dmatrix) / (1 + 2. * f * tau)
            
        df_eff_inv = utils.monotone_fn_inverter(df_eff, arange(0, 50, 0.1))
        self.fwhm = df_eff_inv(self.df_target)



class fMRIStatAR(iterators.LinearModelIterator):

    """
    AR(1) pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()
    fwhm = traits.Float(6.0)
    path = traits.Str('.')
    resid = traits.false
    
    def __init__(self, OLS, contrasts=None, outputs=[], **keywords):
        """
        Building on OLS results, fit the AR(1) model.

        Contrasts is a sequence of terms to be tested in the model.

        """
        
        traits.HasTraits.__init__(self, **keywords)
        self.outputs += outputs
        if not isinstance(OLS, fMRIStatOLS):
            raise ValueError, 'expecting an fMRIStatOLS object in fMRIStatAR'
        self.fmri_image = OLS.fmri_image

        # output path
        
        self.path = OLS.path

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

        self.contrasts = []
        if contrasts is not None:
            if type(contrasts) not in [type([]), type(())]:
                contrasts = [contrasts]
            for i in range(len(contrasts)):
                contrasts[i].getmatrix(time=self.fmri_image.frametimes)
                if isinstance(contrasts[i], DelayContrast):
                    cur = DelayContrastOutput(self.fmri_image, contrasts[i], path=self.path)
                elif contrasts[i].rank == 1:
                    cur = TContrastOutput(self.fmri_image, contrasts[i], path=self.path)
                else:
                    cur = FContrastOutput(self.fmri_image, contrasts[i], path=self.path)
                self.contrasts.append(cur)
                
        # setup the iterator

        self.fmri_image.grid.labels = OLS.labels
        self.fmri_image.grid.labelset = OLS.labelset

        self.iterator = iter(self.fmri_image)
        self.j = 0

        self.outputs += self.contrasts

        if self.resid:
            self.resid_output = ResidOutput(self.fmri_image, path=self.path, basename='ARresid')
            self.outputs.append(self.resid_output)

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


