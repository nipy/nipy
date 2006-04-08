import copy, os, csv, string, fpformat
import numpy as N
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
import neuroimaging.image.regression as imreg
from neuroimaging.statistics import utils

import pylab
from plotting import MultiPlot
canplot = True

class fMRIRegressionOutput(imreg.ImageRegressionOutput):
    """
    A class to output things in GLM passes through fMRI data. It
    uses the fmri_image\'s iterator values to output to an image.

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
        if self.grid.itervalue.type is 'slice':
            value = copy.copy(self.grid.itervalue)
            value.slice = value.slice[1]
        else:
            value = self.grid.itervalue
        self.img.next(data=data, value=value)

    def extract(self, results):
        return 0.

class ResidOutput(fMRIRegressionOutput):

    outdir = traits.Str()
    ext = traits.Str('.img')
    basename = traits.Str('resid')

    def __init__(self, grid, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, grid, **keywords)                
        self.outdir = os.path.join(path)
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        outname = os.path.join(self.outdir, '%s%s' % (self.basename, self.ext))
        self.img = image.Image(outname, mode='w', grid=self.grid,
                               clobber=self.clobber)
        self.nout = self.grid.shape[0]
        self.sync_grid()

    def extract(self, results):
        return results.resid
    
    def next(self, data=None):
        value = self.grid.itervalue
        self.img.next(data=data, value=value)


class TContrastOutput(fMRIRegressionOutput, imreg.TContrastOutput):

    contrast = traits.Any() # should really start specifying classes with traits, too
    effect = traits.true
    sd = traits.true
    t = traits.true
    outdir = traits.Str()
    ext = traits.Str('.img')
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

        if canplot:
            ftime = self.frametimes
            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.contrast.term, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for contrast: \'%s\'' % self.contrast.name)
            pl.draw()
            pylab.savefig(os.path.join(self.outdir, 'matrix.png'))
            f.clf()

    def next(self, data=None):
        if self.grid.itervalue.type is 'slice':
            value = copy.copy(self.grid.itervalue)
            value.slice = value.slice[1]
        else:
            value = self.grid.itervalue

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
    ext = traits.Str('.img')
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

        if canplot:
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
                

## class AROutput(RegressionOutput):

##     order = traits.Int(1)
    
##     def extract(self, results):
##         resid = results.resid
##         ntime = resid.shape[0]

##         # A simple function def to look more like Keith's code
##         # should probably be called mktoeplitz!

##         def mkdiag(values, offset):
##             matrix = N.zeros((values.shape[0] + offset,)*2, N.Float)
##             for i in range(values.shape[0]):
##                 matrix[i+offset,i] = values[i]
##                 matrix[i,i+offset] = values[i]
##             return matrix
        
##         D1 = mkdiag(N.ones(ntime-1, Float), 1)
##         sigma_sq = N.add.reduce(resid**2, 0)

##         R = identity(model.design.shape[0]) - dot(model.design, model.calc_beta)
##         if ARorder == 1:
##             M11 = trace(R)
##             M12 = trace(dot(R, D1))
##             M21 = M12 / 2.
##             tmp = dot(R, D1)
##             M22 = trace(dot(tmp, tmp)) / 2.
##             M = array([[M11, M12], [M21, M22]])
##         else: 
##             M = zeros((ARorder+1,), Float)
            
##             for i in range(ARorder+1):
##                 for j in range(ARorder+1):
##                     Di = dot(R, mkdiag(ones((ntime-i+1,), Float), i-1))/(1.+(i==1))
##                     Dj = dot(R, mkdiag(ones((ntime-j+1,), Float), j-1))/(1.+(j==1))
##                     M[i,j] = trace(dot(Di, Dj))/(1.+(i>1))
                    
##         invM = inverse(M)

##         if ARorder == 1:
##             Cov = array([sigma_sq, sigma_sq])
##             Cov[1] = add.reduce(resid[1:,] * resid[0:-1], 0)
##             Cov = dot(invM, Cov)
##             test = less_equal(Cov[0], 0).astype(Int)
##             output = Cov[1] * (1. - test) / (Cov[0] + test)
##             output.shape = slice_shape
##         else:
##             Cov = zeros((ARorder + 1, sigma_sq.shape), Float)
##             for i in range(ARorder+1):
##                 Cov[i] = add.reduce(resid[i:,] * resid[0:-i], 0)
##             Cov = dot(invM, Cov)
##             test = less_equal(Cov[0], 0).astype(Int)
##             output = Cov[1:] * (1 - test) / (Cov[0] + test)
##         return output


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
