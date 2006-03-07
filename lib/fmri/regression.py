import copy
import numpy as N
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
from neuroimaging.statistics.regression import RegressionOutput

class fMRIRegressionOutput(RegressionOutput):
    """
    A class to output things in GLM passes through fMRI data. It
    uses the fmri_image\'s iterator values to output to an image.

    It can output 
    """

    nout = traits.Int(1)

    def __init__(self, fmri_image, labels=None, labelset=None, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fmri_image
        self.grid = self.fmri_image.grid.subgrid(0)
        if self.nout > 1:
            self.grid = grid.DuplicatedGrids([self.grid]*self.nout)
        self.img = iter(image.Image(N.zeros(self.grid.shape, N.Float), labels, grid=self.grid))

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

class AR1Output(fMRIRegressionOutput):

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
