import gc, os, fpformat

import numpy as N
import numpy.linalg as L
from scipy.sandbox.models.regression import ols_model, ar_model
from scipy.sandbox.models.utils import monotone_fn_inverter, rank 

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.modalities.fmri.fmristat.delay import DelayContrast, \
     DelayContrastOutput
from neuroimaging.algorithms.statistics.regression import LinearModelIterator
from neuroimaging.modalities.fmri.regression import AROutput, \
     TContrastOutput, FContrastOutput, ResidOutput
from neuroimaging.core.reference.iterators import fMRIParcelIterator, \
     fMRISliceParcelIterator, ParcelIterator, SliceParcelIterator
from neuroimaging.algorithms.fwhm import fastFWHM



from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.multiplot import MultiPlot

class WholeBrainNormalize(object):

    def __init__(self, fmri_image, mask=None):
        
        if mask is not None:
            mask = mask[:]
            nvox = mask.astype(N.int32).sum()
        else:
            nvox = N.product(fmri_image.grid.shape[1:])

        self.n = fmri_image.grid.shape[0]
        self.avg = N.zeros((self.n,))

        for i in range(self.n):
            d = fmri_image[i:i+1]
            if mask is not None:
                d = d * mask # can't do in place as the slice points into a 
                             # memmap which may not be writable.
            self.avg[i] = d.sum() / nvox

    def __call__(self, fmri_data):
        out = N.zeros(fmri_data.shape)
        for i in range(self.n):
            out[i] = fmri_data[i] * 100. / self.avg[i]
        return out

class fMRIStatOLS(LinearModelIterator):

    """
    OLS pass of fMRIstat.
    """
    

    def __init__(self, fmri_image, formula, outputs=None, normalize=None,
                 output_fwhm=False, clobber=False, mask=None, slicetimes=None,
                 tshift=0.0, fwhm_rho=6.0, fwhm_data=6.0, resid=False,
                 nmax=200, path='fmristat_run'):

        self.formula = formula
        self.output_fwhm = output_fwhm
        self.clobber = clobber
        self.mask = mask
        self.slicetimes = slicetimes
        self.tshift = tshift
        self.fwhm_rho = fwhm_rho
        self.fwhm_data = fwhm_data
        self.nmax = nmax
        self.path = path
        
        self.fmri_image = fMRIImage(fmri_image)
        if normalize is not None:
            self.normalize = normalize
        else:
            self.normalize = None

        ftime = self.fmri_image.frametimes + self.tshift
        self.dmatrix = self.formula.design(*(ftime,))

        if outputs is None:
            outputs = []

        if resid or self.output_fwhm:
            self.resid_output = ResidOutput(self.fmri_image.grid,
                                            path=self.path,
                                            basename='OLSresid',
                                            clobber=self.clobber)
            self.resid_output.it = \
                          self.fmri_image.slice_iterator(mode='w').copy(self.resid_output.img)
            outputs.append(self.resid_output)

        model = ols_model(self.dmatrix)
        self.rho_estimator = AROutput(self.fmri_image.grid, model)
        outputs.append(self.rho_estimator)

        self.setup_output()

        LinearModelIterator.__init__(self, fmri_image.slice_iterator(),
                                     outputs)
        
    def model(self):
        ftime = self.fmri_image.frametimes + self.tshift
        if self.slicetimes is not None:
            _slice = self.iterator.item.slice
            model = ols_model(self.formula.design(*(ftime + self.slicetimes[_slice[1]],)))
        else:
            model = ols_model(self.dmatrix)
        return model

    def fit(self, reference=None):

        if self.normalize:

            class fMRINormalize(self.iterator.__class__):

                def __init__(self, iterator, normalizer):
                    self.iterator = iterator
                    self.normalizer = normalizer

                def next(self):
                    v = self.iterator.next()
                    self.item = self.iterator.item
                    return self.normalizer(v)

                def __iter__(self):
                    return self

            self._iterator = self.iterator
            self.iterator = fMRINormalize(self._iterator, self.normalize)

        LinearModelIterator.fit(self)

        sgrid = self.fmri_image.grid.subgrid(0)

        if self.output_fwhm:
            resid = fMRIImage(self.resid_output.img)
            fwhmest = fastFWHM(resid, fwhm=os.path.join(self.path, 'fwhmOLS.img'), clobber=self.clobber)
            fwhmest()
            self.fwhm_data = fwhmest.integrate(mask=self.mask)[1]
            print 'FWHM for data estimated as: %02f' % self.fwhm_data

        if reference is not None: self.estimateFWHM_AR(reference)
            
##      this will fail for non-affine grids, or grids
##      whose axes are not aligned in the standard way            

        #sigma = fwhm2sigma(self.fwhm_rho / N.array(self.fmri_image.image.pixdim[1:4][::-1]))

##         smoother = kernel_smooth.LinearFilter(sgrid, fwhm=self.fwhm_rho)
##         self.rho_estimator.img.grid = sgrid
##         self.rho = smoother.smooth(self.rho_estimator.img)

##        srho = scipy.ndimage.gaussian_filter(self.rho_estimator.img.readall(), sigma)
 ##       self.rho_estimator.img.writeslice(slice(0, self.rho_estimator.img.grid.shape[0], 1), srho)
        self.rho = self.rho_estimator.img
##        self.getparcelmap()

    def getparcelmap(self):

        if self.mask is not None:
            _mask = self.mask.readall()
            self.rho[:] = N.where(_mask, self.rho[:], N.nan)

        if self.slicetimes == None:
            tmp = N.around(self.rho.readall() * (self.nmax / 2.)) / (self.nmax / 2.)
            tmp.shape = tmp.size
            parcelmap = tmp
            tmp = N.compress(1 - N.isnan(tmp), tmp)
            parcelseq = list(N.unique(tmp))
            del(tmp); gc.collect()
        else:
            parcelmap = []
            parcelseq = []
            for i in range(self.rho.grid.shape[0]):
                tmp = self.rho[i:i+1]
                tmp.shape = tmp.size
                tmp = N.around(tmp * (self.nmax / 2.)) / (self.nmax / 2.)
                newlabels = list(N.unique(tmp))
                parcelseq += [newlabels]
                parcelmap += [tmp]
        return parcelmap, parcelseq
            
    def setup_output(self):

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ftime = self.fmri_image.frametimes + self.tshift
        dmatrix = self.dmatrix.astype('<f8')
        ftime.shape = (ftime.shape[0], 1)
        dmatrix = N.hstack([ftime, dmatrix])
        ftime.shape = (ftime.shape[0],)

        outname = os.path.join(self.path, 'matrix.csv')
        outfile = file(outname, 'w')
        tmatrix = [[fpformat.fix(dmatrix[i][j], 4)
                    for j in range(dmatrix.shape[1])]
                   for i in range(dmatrix.shape[0])]
        outfile.write('\n'.join(','.join(tmatrix[i]) for i in range(dmatrix.shape[0])))
        outfile.close()

        outname = os.path.join(self.path, 'matrix.bin')
        outfile = file(outname, 'w')
        dmatrix = self.dmatrix.astype('<f8')
        dmatrix.tofile(outfile)
        outfile.close()

        if PYLAB_DEF:
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

        x = N.dot(L.pinv(self.dmatrix).T, reference.matrix)

        def aclag(x, j):
            return N.add.reduce(x[j:] * x[0:-j]) / N.add.reduce(x**2)

        tau = 0.
        for j in range(ARorder):
            tau += aclag(x, j+1)**2

        ndim = len(self.fmri_image.shape) - 1

        dfresid = self.fmri_image.shape[0] - rank(self.dmatrix)

        def df_eff(fwhm_filter):
            f = N.power(1 + 2. * (fwhm_filter / self.fwhm_data)**2, -ndim/2.)
            return dfresid / (1 + 2. * f * tau)
            
        df_eff_inv = monotone_fn_inverter(df_eff, N.linspace(0, 50, 500))
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
    

    def __init__(self, OLS, contrasts=None, outputs=None, clobber=False,
                 resid=False, tshift=0.0, parcel=None):
        """
        Building on OLS results, fit the AR(1) model.

        Contrasts is a sequence of terms to be tested in the model.
        """

        if outputs is None:
            outputs = []
        else:
            outputs = outputs[:]
        if not isinstance(OLS, fMRIStatOLS):
            raise ValueError, 'expecting an fMRIStatOLS object in fMRIStatAR'

        self.fmri_image = OLS.fmri_image
        self.normalize = OLS.normalize
        
        # output path
        path = OLS.path

        # copy the formula
        
        self.slicetimes = OLS.slicetimes
        ftime = self.fmri_image.frametimes + tshift

        self.formula = OLS.formula
        if self.slicetimes is None:
            self.dmatrix = OLS.dmatrix
            iterator = fMRIParcelIterator
        else:
            self.designs = []
            for s in self.slicetimes:
                self.designs.append(self.formula.design(*(ftime+s,))) 
            iterator = fMRISliceParcelIterator

        if parcel is None:
            parcelmap, parcelseq = OLS.getparcelmap()
        else:
            parcelmap, parcelseq = parcel

        if iterator == fMRIParcelIterator:
            iterator_ = ParcelIterator
        elif iterator == fMRISliceParcelIterator:
            iterator_ = SliceParcelIterator

        self.contrasts = []
        if contrasts is not None:
            if not isinstance(contrasts, (tuple, list)):
                contrasts = [contrasts]
            for contrast in contrasts:
                contrast.getmatrix(time=ftime)
                if isinstance(contrast, DelayContrast):
                    cur = DelayContrastOutput(self.fmri_image.grid,
                                              contrast, path=path,
                                              clobber=clobber,
                                              frametimes=ftime,
                                              it=iterator_(self.fmri_image.grid,
                                                           parcelmap,
                                                           parcelseq, mode='w'))
                elif contrast.rank == 1:
                    cur = TContrastOutput(self.fmri_image.grid, contrast,
                                          path=path,
                                          clobber=clobber,
                                          frametimes=ftime,
                                          it=iterator_(self.fmri_image.grid,
                                                       parcelmap,
                                                       parcelseq, mode='w'))
                else:
                    cur = FContrastOutput(self.fmri_image.grid, contrast,
                                          path=path,
                                          clobber=clobber,
                                          frametimes=ftime,
                                          it=iterator_(self.fmri_image.grid,
                                                       parcelmap,
                                                       parcelseq, mode='w'))
                self.contrasts.append(cur)

        outputs += self.contrasts

        if resid:
            self.resid_output = ResidOutput(self.fmri_image.grid,
                                            path=path,
                                            basename='ARresid',
                                            clobber=clobber)
            self.resid_output.it = iterator(self.resid_output.img, parcelmap,
                                            parcelseq, mode='w')
            outputs.append(self.resid_output)

        it = iterator(OLS.fmri_image, parcelmap, parcelseq)
        LinearModelIterator.__init__(self, it, outputs)

    def fit(self):
        if self.normalize:
            class fMRINormalize(self.iterator.__class__):

                def __init__(self, iterator, normalizer):
                    self.iterator = iterator
                    self.normalizer = normalizer

                def next(self):
                    v = self.iterator.next()
                    self.item = self.iterator.item
                    return self.normalizer(v)

                def __iter__(self):
                    return self

            self._iterator = self.iterator
            self.iterator = fMRINormalize(self._iterator, self.normalize)

        LinearModelIterator.fit(self)

    def model(self):
        itervalue = self.iterator.item
        if self.slicetimes is not None:
            design = self.designs[itervalue.i]
        else:
            design = self.dmatrix
        # is using the first parcel label correct here?
        # rho needs to be a single float...

        return ar_model(design, itervalue.label[0]) 




