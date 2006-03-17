import copy, os, csv, string, fpformat
import numpy as N
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
from neuroimaging.statistics.regression import RegressionOutput
from neuroimaging.statistics import utils

import pylab
from plotting import MultiPlot
canplot = True


class fMRIRegressionOutput(RegressionOutput):
    """
    A class to output things in GLM passes through fMRI data. It
    uses the fmri_image\'s iterator values to output to an image.

    It can output 
    """

    nout = traits.Int(1)
    imgarray = traits.false
    clobber = traits.false

    def __init__(self, fmri_image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fmri_image
        self.grid = self.fmri_image.grid.subgrid(0)
        if self.nout > 1:
            self.grid = grid.DuplicatedGrids([self.grid]*self.nout)
        if self.imgarray:
            self.img = iter(image.Image(N.zeros(self.grid.shape, N.Float), grid=self.grid))

    def sync_grid(self, img=None):
        """
        Synchronize an image's grid iterator to self.grid's iterator.
        """
        if img is None:
            img = self.img
        img.grid.itertype = self.grid.itertype
        img.grid.labels = self.grid.labels
        img.grid.labelset = self.grid.labelset
        iter(img)
        
    def __iter__(self):
        return self

    def next(self, data=None):
        if self.fmri_image.itervalue.type is 'slice':
            value = copy.copy(self.fmri_image.itervalue)
            value.slice = value.slice[1]
        else:
            value = self.fmri_image.itervalue
        self.img.next(data=data, value=value)

    def extract(self, results):
        return 0.

class TContrastOutput(fMRIRegressionOutput):

    contrast = traits.Any() # should really start specifying classes with traits, too
    effect = traits.true
    sd = traits.true
    t = traits.true
    outdir = traits.Str()
    ext = traits.Str('.img')
    subpath = traits.Str('contrasts')

    def __init__(self, fmri_image, contrast, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, fmri_image, **keywords)                
        self.grid = self.fmri_image.grid.subgrid(0)
        self.contrast = contrast

        self.outdir = os.path.join(path, self.subpath, self.contrast.name)
        self.path = path
        self.setup_contrast()
        self.setup_output()

    def setup_contrast(self):
        self.contrast.getmatrix(time=self.fmri_image.frametimes)

    def setup_output(self):

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        outname = os.path.join(self.outdir, 't%s' % self.ext)
        self.timg = image.Image(outname, mode='w', grid=self.grid,
                                clobber=self.clobber)
        self.sync_grid(img=self.timg)

        if self.effect:
            outname = os.path.join(self.outdir, 'effect%s' % self.ext)
            self.effectimg = image.Image(outname, mode='w', grid=self.grid,
                                         clobber=self.clobber)
            self.sync_grid(img=self.effectimg)
        if self.sd:
            outname = os.path.join(self.outdir, 'sd%s' % self.ext)
            self.sdimg = iter(image.Image(outname, mode='w', grid=self.grid,
                                          clobber=self.clobber))
            self.sync_grid(img=self.sdimg)

        if not hasattr(self.contrast, 'matrix'):
            self.contrast.getmatrix(time=self.fmri_image.frametimes)

        outname = os.path.join(self.outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        outfile.write(string.join([fpformat.fix(x,4) for x in self.contrast.matrix], ',') + '\n')
        outfile.close()

        outname = os.path.join(self.outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

        if canplot:
            ftime = self.fmri_image.frametimes
            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.contrast.term, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for contrast: \'%s\'' % self.contrast.name)
            pl.draw()
            pylab.savefig(os.path.join(self.outdir, 'matrix.png'))
            f.clf()

    def extract(self, results):
        return results.Tcontrast(self.contrast.matrix, sd=self.sd, t=self.t)

    def next(self, data=None):
        if self.fmri_image.itervalue.type is 'slice':
            value = copy.copy(self.fmri_image.itervalue)
            value.slice = value.slice[1]
        else:
            value = self.fmri_image.itervalue

        self.timg.next(data=data.t, value=value)
        if self.effect:
            self.effectimg.next(data=data.effect, value=value)
        if self.sd:
            self.sdimg.next(data=data.effect, value=value)

class FContrastOutput(fMRIRegressionOutput):

    contrast = traits.Any() # should really start specifying classes with traits, too
    outdir = traits.Str()
    ext = traits.Str('.img')
    subpath = traits.Str('contrasts')

    def __init__(self, fmri_image, contrast, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, fmri_image, **keywords)                
        self.grid = self.fmri_image.grid.subgrid(0)
        self.contrast = contrast
        self.path = path
        self.outdir = os.path.join(self.path, self.subpath, self.contrast.name)
        self.setup_contrast()
        self.setup_output()

    def setup_contrast(self):
        self.contrast.getmatrix(time=self.fmri_image.frametimes)

    def setup_output(self):

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        outname = os.path.join(self.outdir, 'F%s' % self.ext)
        self.img = iter(image.Image(outname, mode='w', grid=self.grid,
                                    clobber=self.clobber))
        self.sync_grid()

        outname = os.path.join(self.outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        writer = csv.writer(outfile)
        for row in self.contrast.matrix:
            writer.writerow([fpformat.fix(x, 4) for x in row])
        outfile.close()

        outname = os.path.join(self.outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

        if canplot:
            ftime = self.fmri_image.frametimes

            f = pylab.gcf()
            f.clf()
            pl = MultiPlot(self.contrast.term, tmin=0, tmax=ftime.max(),
                           dt = ftime.max() / 2000., title='Column space for contrast: \'%s\'' % self.contrast.name)
            pl.draw()
            pylab.savefig(os.path.join(self.outdir, 'matrix.png'))
            f.clf()

    def extract(self, results):
        F = results.Fcontrast(self.contrast.matrix).F
        return results.Fcontrast(self.contrast.matrix).F


class AR1Output(fMRIRegressionOutput):

    imgarray = traits.true 

    def extract(self, results):
        resid = results.resid
        rho = N.add.reduce(resid[0:-1]*resid[1:] / N.add.reduce(resid[1:-1]**2))
        return rho


class ResidOutput(fMRIRegressionOutput):

    outdir = traits.Str()
    ext = traits.Str('.img')
    basename = traits.Str('resid')

    def __init__(self, fmri_image, path='.', **keywords):
        fMRIRegressionOutput.__init__(self, fmri_image, **keywords)                
        self.grid = self.fmri_image.grid
        self.outdir = os.path.join(path)
        self.path = path
    
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        outname = os.path.join(self.outdir, '%s%s' % (self.basename, self.ext))
        self.img = image.Image(outname, mode='w', grid=self.grid)
        self.nout = self.grid.shape[0]
        self.sync_grid()

    def extract(self, results):
        return results.resid
    
    def next(self, data=None):
        value = copy.copy(self.fmri_image.itervalue)
        self.img.next(data=data, value=value)


                 

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
