import copy, os

import numpy as N
import numpy.linalg as L
from scipy.linalg import toeplitz
from scipy.sandbox.models.utils import recipr
from neuroimaging import traits

from neuroimaging.core.image.image import Image
import neuroimaging.algorithms.regression as imreg

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.multiplot import MultiPlot

class fMRIRegressionOutput(imreg.ImageRegressionOutput):
    """
    A class to output things in GLM passes through fMRI data. It
    uses the fmri_image's iterator values to output to an image.

    The difference between this class and ImageRegressionOutput is the
    iterator that drives everything: here it the iterator of an fMRIImage,
    in the former it is of an Image.
    """

    nout = traits.Int(1)
    imgarray = traits.false
    clobber = traits.false

    def __init__(self, grid, **keywords):
        imreg.ImageRegressionOutput.__init__(self, grid, outgrid=grid.subgrid(0), **keywords)

    def __iter__(self):
        return self

    def next(self, data=None):
        if self.grid.get_iter_param("itertype") == 'slice':
            value = copy.copy(self.grid.itervalue())
            value.slice = value.slice[1]
        else:
            value = self.grid.itervalue()
        self.img.next(data=data, value=value)

    def extract(self, results):
        return 0.

class ResidOutput(fMRIRegressionOutput):

    outdir = traits.Str()
    ext = traits.Str('.hdr')
    basename = traits.Str('resid')

    def __init__(self, grid, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, grid, **keywords)                
        self.outdir = os.path.join(path)
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        outname = os.path.join(self.outdir, '%s%s' % (self.basename, self.ext))
        self.img = Image(outname, mode='w', grid=self.grid,
                               clobber=self.clobber)
        self.nout = self.grid.shape[0]
        self.sync_grid()

    def extract(self, results):
        return results.resid
    
    def next(self, data=None):
        value = self.grid.itervalue()
        self.img.next(data=data, value=value)


class TContrastOutput(fMRIRegressionOutput, imreg.TContrastOutput):

    contrast = traits.Any() # should really start specifying classes with traits, too
    effect = traits.true
    sd = traits.true
    t = traits.true
    outdir = traits.Str()
    ext = traits.Str('.hdr')
    subpath = traits.Str('contrasts')
    frametimes = traits.Any()

    def __init__(self, grid, contrast, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, grid, **keywords)                
        self.contrast = contrast

        self.outdir = os.path.join(path, self.subpath, self.contrast.name)
        self.path = path
        self.setup_contrast(time=self.frametimes)
        self.setup_output()

    def setup_output(self):

        imreg.TContrastOutput.setup_output(self)

        if PYLAB_DEF:
            ftime = self.frametimes
            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.contrast.term, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for contrast: \'%s\'' % self.contrast.name)
            pl.draw()
            pylab.savefig(os.path.join(self.outdir, 'matrix.png'))
            f.clf()

    def next(self, data=None):
        if self.grid._itertype == 'slice':
            value = copy.copy(self.grid.itervalue())
            value.slice = value.slice[1]
        else:
            value = self.grid.itervalue()

        self.timg.next(data=data.t, value=value)
        if self.effect:
            self.effectimg.next(data=data.effect, value=value)
        if self.sd:
            self.sdimg.next(data=data.sd, value=value)

    def extract(self, results):
        return imreg.TContrastOutput.extract(self, results)

class FContrastOutput(fMRIRegressionOutput, imreg.FContrastOutput):

    contrast = traits.Any() 
    outdir = traits.Str()
    ext = traits.Str('.hdr')
    subpath = traits.Str('contrasts')
    frametimes = traits.Any()

    def __init__(self, grid, contrast, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, grid, **keywords)                
        self.contrast = contrast
        self.path = path
        self.outdir = os.path.join(self.path, self.subpath, self.contrast.name)
        self.setup_contrast(time=self.frametimes)
        self.setup_output()

    def setup_output(self):

        imreg.FContrastOutput.setup_output(self)

        if PYLAB_DEF:
            ftime = self.frametimes

            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.contrast.term, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for contrast: \'%s\'' % self.contrast.name)
            pl.draw()
            pylab.savefig(os.path.join(self.outdir, 'matrix.png'))
            f.clf()

    def extract(self, results):
        return imreg.FContrastOutput.extract(self, results)

class AR1Output(fMRIRegressionOutput):

    def __init__(self, grid, **keywords):
        arraygrid = grid.subgrid(0)
        fMRIRegressionOutput.__init__(self, grid, arraygrid=arraygrid, **keywords)

    def extract(self, results):
        resid = results.resid
        rho = N.add.reduce(resid[0:-1]*resid[1:] / N.add.reduce(resid[1:-1]**2))
        return rho
                

class AROutput(fMRIRegressionOutput):

    order = traits.Int(1)
    def __init__(self, grid, **keywords):
        arraygrid = grid.subgrid(0)
        fMRIRegressionOutput.__init__(self, grid, arraygrid=arraygrid, **keywords)
        self.invM = N.identity(self.order + 1)

    def setup_bias_correct(self, model):

        R = N.identity(model.design.shape[0]) - N.dot(model.design, model.calc_beta)
        M = N.zeros((self.order+1,)*2, N.float64)
        I = N.identity(R.shape[0])

        for i in range(self.order+1):
            Di = N.dot(R, toeplitz(I[i]))
            for j in range(self.order+1):
                Dj = N.dot(R, toeplitz(I[j]))
                M[i,j] = N.diagonal((N.dot(Di, Dj))/(1.+(i>0))).sum()
                    
        self.invM = L.inv(M)

        return
    
    def extract(self, results):
        resid = results.resid.reshape((results.resid.shape[0],
                                       N.product(results.resid.shape[1:])))
        ntime = resid.shape[0]

        sum_sq = results.scale.reshape(resid.shape[1:]) * results.df_resid

        Cov = N.zeros((self.order + 1,) + sum_sq.shape, N.float64)
        Cov[0] = sum_sq
        for i in range(1, self.order+1):
            Cov[i] = N.add.reduce(resid[i:] * resid[0:-i], 0)
        Cov = N.dot(self.invM, Cov)
        output = Cov[1:] * recipr(Cov[0])
        return N.squeeze(output)


## this makes sense only if the labels are [slice, other] labels...
## no rush for this yet
    

## class FWHMOutput(Output):

##     def __init__(self, fmri, labels=None):
##         Output.__init__(self, labels=labels)
##         self.ndim = fmri.shape[0]
##         self.fwhmest = iter(iterFWHM(fmri.frame(0)))

##     def next(self, data=None, iterator=None):
##         if hasattr(iterator, 'newslice'): # for LabelledfMRIImage class
##             if iterator.newslice:
##                 self.fwhmest.next(data=iterator.buffer)
##         else:
##             self.fwhmest.next(data=data)

##         del(data) ; gc.collect()

##     def extract(self, results):
##         return results.norm_resid
