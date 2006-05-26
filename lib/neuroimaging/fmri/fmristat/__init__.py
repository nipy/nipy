import gc, os, fpformat
from enthought import traits

import numpy as N
import numpy.linalg as L
import numpy.random as R
import scipy.ndimage
import pylab

from neuroimaging.fmri import fMRIImage
from neuroimaging.fmri.fmristat.delay import DelayContrast, DelayContrastOutput
from neuroimaging.fmri.plotting import MultiPlot
from neuroimaging.fmri.regression import AROutput, TContrastOutput, \
  FContrastOutput, ResidOutput
from neuroimaging.image import kernel_smooth
from neuroimaging.image.fwhm import fastFWHM
from neuroimaging.image.utils import fwhm2sigma
from neuroimaging.statistics import utils 
from neuroimaging.statistics.regression import LinearModelIterator, \
  OLSModel, ARModel

canplot = True

class WholeBrainNormalize(traits.HasTraits):

    mask = traits.Any()

    def __init__(self, fmri_image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if self.mask is not None:
            self._mask = self.mask.readall()
            self._mask.shape = N.product(self._mask.shape)
            
        self.n = fmri_image.grid.shape[0]
        self.avg = N.zeros((self.n,), N.Float)

        for i in range(self.n):
            d = fmri_image.getslice(slice(i,i+1))
            if hasattr(self, '_mask'):
                d.shape = N.product(d.shape)
                d = N.compress(self._mask, d)
            self.avg[i] = d.mean()

    def __call__(self, fmri_data):
        out = N.zeros(fmri_data.shape, N.Float)
        for i in range(self.n):
            out[i] = fmri_data[i] * 100. / self.avg[i]
        return out

class fMRIStatOLS(LinearModelIterator):

    """
    OLS pass of fMRIstat.
    """
    
    normalize = traits.Any()
    
    formula = traits.Any()
    slicetimes = traits.Any()
    tshift = traits.Float(0.)

    fwhm_rho = traits.Float(6.)
    fwhm_data = traits.Float(6.)
    target_df = traits.Float(100.)
    nmax = traits.Int(200) # maximum number of rho values
    mask = traits.Any()
    path = traits.String('fmristat_run')
    resid = traits.false
    clobber = traits.false
    output_fwhm = traits.false

    def __init__(self, fmri_image, outputs=[], **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fMRIImage(fmri_image)

        if self.normalize is not None:
            self.fmri_image.postread = self.normalize

        self.iterator = iter(self.fmri_image)

        self.outputs += outputs

        ftime = self.fmri_image.frametimes + self.tshift
        self.dmatrix = self.formula.design(time=ftime)

        if self.resid or self.output_fwhm:
            self.resid_output = ResidOutput(self.fmri_image.grid, path=self.path, basename='OLSresid', clobber=self.clobber)
            self.outputs.append(self.resid_output)

        self.rho_estimator = AROutput(self.fmri_image.grid, clobber=self.clobber)
        self.rho_estimator.setup_bias_correct(OLSModel(design=self.dmatrix))
        self.outputs.append(self.rho_estimator)

        self.setup_output()
        
    def model(self, **keywords):
        ftime = self.fmri_image.frametimes + self.tshift
        if self.slicetimes is not None:
            _slice = self.iterator.grid.itervalue.slice
            model = OLSModel(design=self.formula.design(time=ftime + self.slicetimes[_slice[1]]))
        else:
            model = OLSModel(design=self.dmatrix)
        return model

    def fit(self, reference=None, **keywords):

        LinearModelIterator.fit(self, **keywords)

        sgrid = self.fmri_image.grid.subgrid(0)

        if self.output_fwhm:
            resid = fMRIImage(self.resid_output.img)
            fwhmest = fastFWHM(resid, fwhm=os.path.join(self.path, 'fwhmOLS.img'), clobber=self.clobber)
            fwhmest()
            self.fwhm_data = fwhmest.integrate(mask=self.mask)[1]
            print 'FWHM for data estimated as: %02f' % self.fwhm_data

        if self.reference is not None:
            self.estimateFWHM_AR(self.reference)
            
##      this will fail for non-affine grids, or grids
##      whose axes are not aligned in the standard way            

        sigma = fwhm2sigma(self.fwhm_rho / N.array(self.fmri_image.image.pixdim[1:4][::-1]))

##         smoother = kernel_smooth.LinearFilter(sgrid, fwhm=self.fwhm_rho)
##         self.rho_estimator.img.grid = sgrid
##         self.rho = smoother.smooth(self.rho_estimator.img)

        srho = scipy.ndimage.gaussian_filter(self.rho_estimator.img.readall(), sigma)
        self.rho_estimator.img.writeslice(slice(0, self.rho_estimator.img.grid.shape[0], 1), srho)
        self.rho = self.rho_estimator.img
        self.getlabels()

    def getlabels(self):

        if self.mask is not None:
            _mask = self.mask.readall()
            self.rho.image.data = N.where(_mask, self.rho.image.data, N.nan)

        if self.slicetimes == None:
            tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
            tmp.shape = N.product(tmp.shape)
            self.labels = tmp
            tmp = N.compress(1 - N.isnan(tmp), tmp)
            self.parcelseq = list(N.unique(tmp))
            del(tmp); gc.collect()
        else:
            self.labels = []
            self.parcelseq = []
            for i in range(self.rho.grid.shape[0]):
                tmp = self.rho.getslice(slice(i,i+1))
                tmp.shape = N.product(tmp.shape)
                tmp = N.around(tmp * (self.nmax / 2.)) / (self.nmax / 2.)
                newlabels = list(N.unique(tmp))
                self.parcelseq += [newlabels]
                self.labels += [tmp]

    def setup_output(self):

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ftime = self.fmri_image.frametimes + self.tshift
        dmatrix = self.dmatrix.astype('<f8')
        ftime.shape = (ftime.shape[0],1)
        dmatrix = N.hstack([ftime, dmatrix])
        ftime.shape = (ftime.shape[0],)

        outname = os.path.join(self.path, 'matrix.csv')
        outfile = file(outname, 'w')
        tmatrix = [[fpformat.fix(dmatrix[i][j], 4) for j in range(dmatrix.shape[1])] for i in range(dmatrix.shape[0])]
        outfile.write('\n'.join(','.join(tmatrix[i]) for i in range(dmatrix.shape[0])))
        outfile.close()

        outname = os.path.join(self.path, 'matrix.bin')
        outfile = file(outname, 'w')
        dmatrix = self.dmatrix.astype('<f8')
        dmatrix.tofile(outfile)
        outfile.close()

        if canplot:

            f = pylab.gcf()
            f.clf()

            pl = MultiPlot(self.formula, tmin=ftime.min(), tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for design matrix')
            pl.draw()
            pylab.savefig(os.path.join(self.path, 'matrix.png'))
            f.clf()


    def estimateFWHM_AR(self, reference,
                        ARorder=1, df_target=100.):
        """
        Estimate smoothing of AR coefficient to get
        a targeted df.

        Worsley, K.J. (2005). \'Spatial smoothing of autocorrelations to control the degrees of freedom in fMRI analysis.\' NeuroImage, 26:635-641.

        """

        reference.getmatrix(time=self.fmri_image.frametimes + self.tshift)

        x = N.dot(N.transpose(L.pinv(self.dmatrix)),
                  reference.matrix)

        def aclag(x, j):
            return N.add.reduce(x[j:] * x[0:-j]) / N.add.reduce(x**2)

        tau = 0.
        for j in range(ARorder):
            tau = tau + aclag(x, j+1)**2

        ndim = len(self.fmri_image.shape) - 1

        dfresid = self.fmri_image.shape[0] - utils.rank(self.dmatrix)

        def df_eff(fwhm_filter):
            f = N.power(1 + 2. * (fwhm_filter / self.fwhm_data)**2, -ndim/2.)
            return dfresid / (1 + 2. * f * tau)
            
        df_eff_inv = utils.monotone_fn_inverter(df_eff, N.linspace(0, 50, 500))
        if df_eff(0) > df_target:
            self.fwhm_rho = 0.
        else:
            try:
                self.fwhm_rho = df_eff_inv(df_target)[0]
            except:
                self.fwhm_rho = 0.
        print 'FWHM for AR estimated as: %02f' % self.fwhm_rho


class fMRIStatAR(LinearModelIterator):

    """
    AR(1) pass of fMRIstat.
    """
    
    formula = traits.Any()
    slicetimes = traits.Any()
    tshift = traits.Float(0.)

    fwhm = traits.Float(6.0)
    path = traits.Str('.')
    resid = traits.false
    clobber = traits.false

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
        ftime = self.fmri_image.frametimes + self.tshift

        self.formula = OLS.formula
        if self.slicetimes is None:
            self.dmatrix = OLS.dmatrix
            self.fmri_image.grid.itertype = 'parcel'
        else:
            self.fmri_image.grid.itertype = 'slice/parcel'
            self.designs = []
            for i in range(len(self.slicetimes)):
                self.designs.append(self.formula.design(time=ftime + self.slicetimes[i]))

        self.contrasts = []
        if contrasts is not None:
            if type(contrasts) not in [type([]), type(())]:
                contrasts = [contrasts]
            for i in range(len(contrasts)):
                contrasts[i].getmatrix(time=ftime)
                if isinstance(contrasts[i], DelayContrast):
                    cur = DelayContrastOutput(self.fmri_image.grid,
                                              contrasts[i], path=self.path,
                                              clobber=self.clobber,
                                              frametimes=ftime)
                elif contrasts[i].rank == 1:
                    cur = TContrastOutput(self.fmri_image.grid, contrasts[i],
                                          path=self.path,
                                          clobber=self.clobber,
                                          frametimes=ftime)
                else:
                    cur = FContrastOutput(self.fmri_image.grid, contrasts[i],
                                          path=self.path,
                                          clobber=self.clobber,
                                          frametimes=ftime)
                self.contrasts.append(cur)
                
        # setup the iterator

        self.fmri_image.grid.parcelmap = OLS.parcelmap
        self.fmri_image.grid.parcelseq = OLS.parcelseq

        self.iterator = iter(self.fmri_image)
        self.j = 0

        self.outputs += self.contrasts

        if self.resid:
            self.resid_output = ResidOutput(self.fmri_image.grid,
                                            path=self.path,
                                            basename='ARresid',
                                            clobber=self.clobber)
            self.outputs.append(self.resid_output)

    def model(self, **keywords):
        self.j += 1
        if self.slicetimes is not None:
            rho = self.iterator.grid.itervalue.label
            i = self.iterator.grid.itervalue.slice[1]
            model = ARModel(rho=rho, design=self.designs[i])
        else:
            rho = self.iterator.grid.itervalue.label
            model = ARModel(rho=rho, design=self.dmatrix)
        return model
