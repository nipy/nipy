"""
TODO
"""
__docformat__ = 'restructuredtext'

import os

import numpy as N
import numpy.linalg as L
from scipy.linalg import toeplitz
from neuroimaging.fixes.scipy.stats.models.utils import recipr

import neuroimaging.algorithms.statistics.regression as imreg

class FmriRegressionOutput(imreg.ImageRegressionOutput):
    """
    A class to output things in GLM passes through fMRI data. It
    uses the fmri_image's iterator values to output to an image.

    The difference between this class and ImageRegressionOutput is the
    iterator that drives everything: here it the iterator of an FmriImage,
    in the former it is of an Image.
    """

    def __init__(self, grid, nout=1):
        """ 
        :Parameters:
            `grid` : TODO
                TODO
            `nout` : int
                TODO
        """ 
        imreg.ImageRegressionOutput.__init__(self, grid, nout=nout,
                                             outgrid=grid.subgrid(0))


class ResidOutput(FmriRegressionOutput):
    """
    TODO
    """

    def __init__(self, grid, nout=1, clobber=False,
                 path='.', ext='.nii', basename='resid', it=None):
        """
        :Parameters:
            `grid` : TODO
                TODO
            `nout` : int
                TODO
            `clobber` : bool
                TODO
            `path` : string
                TODO
            `ext` : string
                TODO
            `basename` : string
                TODO
            `it` : TODO
                TODO
        """
        FmriRegressionOutput.__init__(self, grid, nout)
        outdir = os.path.join(path)
        self.outgrid = grid
        if it is None:
            self.it.axis = [1]
        else:
            self.it = it
        self.img, self.it = self._setup_img(clobber, outdir, ext, basename)
        self.nout = self.grid.shape[0]

    def extract(self, results):
        """
        :Parameters:
            `resid` : TODO
                TODO
                
        :Returns: TODO
        """
        return results.resid
    

class TContrastOutput(FmriRegressionOutput, imreg.TContrastOutput):
    """
    TODO
    """

    def __init__(self, grid, contrast, nout=1, clobber=False,
                 path='.', ext='.nii', subpath='contrasts', frametimes=[],
                 effect=True, sd=True, t=True, it=None):                 
        """
        :Parameters:
            `grid` : TODO
                TODO
            `contrast` : TODO
                TODO
            `nout` : int
                TODO
            `clobber` : bool
                TODO
            `path` : string
                TODO
            `ext` : string
                TODO
            `subpath` : string
                TODO
            `frametimes` : TODO
                TODO
            `effect` : bool
                TODO
            `sd` : bool
                TODO
            `t` : bool
                TODO
            `it` : TODO
                TODO
        """
        FmriRegressionOutput.__init__(self, grid, nout)
        if it is not None:
            self.it = it
        self.contrast = contrast
        self.effect = effect
        self.sd = sd
        self.t = t
        self._setup_contrast(time=frametimes)
        self._setup_output(clobber, path, subpath, ext, frametimes)
        
    def _setup_output(self, clobber, path, subpath, ext, frametimes):
        outdir = os.path.join(path, subpath, self.contrast.name)
        imreg.TContrastOutput._setup_output(self, clobber, path, subpath, ext)


    def extract(self, results):
        """
        :Parameters:
            `results` : TODO
                TODO
                
        :Returns: TODO
        """
        return imreg.TContrastOutput.extract(self, results)

class FContrastOutput(FmriRegressionOutput, imreg.FContrastOutput):
    """
    TODO
    """

    def __init__(self, grid, contrast, path='.', ext='.nii', clobber=False,
                 subpath='contrasts', frametimes=[], nout=1, it=None):
        """
        :Parameters:
            `grid` : TODO
                TODO
            `contrast` : TODO
                TODO
            `path` : string            
                TODO
            `ext` : string
                TODO
            `clobber` : bool
                TODO
            `subpath` : strings
                TODO
            `frametimes` : TODO
                TODO
            `nout` : int
                TODO
            `it` : TODO
                TODO                        
        """
        FmriRegressionOutput.__init__(self, grid, nout)
        if it is not None:
            self.it = it
        self.contrast = contrast
        self._setup_contrast(time=frametimes)
        self._setup_output(clobber, path, subpath, ext, frametimes)

    def _setup_output(self, clobber, path, subpath, ext, frametimes):
        outdir = os.path.join(path, subpath, self.contrast.name)
        imreg.FContrastOutput._setup_output(self, clobber, path, subpath, ext)

    def extract(self, results):
        """
        :Parameters:
            `results` : TODO
                TODO
                
        :Returns: TODO
        """
        return imreg.FContrastOutput.extract(self, results)

class AR1Output(FmriRegressionOutput):
    """
    TODO
    """

    def extract(self, results):
        """
        :Parameters:
            `results` : TODO
                TODO
                
        :Returns: TODO
        """
        resid = results.resid
        rho = N.add.reduce(resid[0:-1]*resid[1:] / N.add.reduce(resid[1:-1]**2))
        return rho
                

class AROutput(FmriRegressionOutput):
    """
    TODO
    """

    def __init__(self, grid, model, order=1, nout=1):
        """
        :Parameters:
            `grid` : TODO
                TODO
            `model` : TODO
                TODO
            `order` : int
                TODO
            `nout` : int
                TODO
        """
        self.order = order
        FmriRegressionOutput.__init__(self, grid, nout)
        self._setup_bias_correct(model)

    def _setup_bias_correct(self, model):

        R = N.identity(model.design.shape[0]) - N.dot(model.design, model.calc_beta)
        M = N.zeros((self.order+1,)*2)
        I = N.identity(R.shape[0])

        for i in range(self.order+1):
            Di = N.dot(R, toeplitz(I[i]))
            for j in range(self.order+1):
                Dj = N.dot(R, toeplitz(I[j]))
                M[i,j] = N.diagonal((N.dot(Di, Dj))/(1.+(i>0))).sum()
                    
        self.invM = L.inv(M)

        return
    
    def extract(self, results):
        """
        :Parameters:
            `results` : TODO
                TODO
                
        :Returns: ``numpy.ndarray``
        """
        resid = results.resid.reshape((results.resid.shape[0],
                                       N.product(results.resid.shape[1:])))

        sum_sq = results.scale.reshape(resid.shape[1:]) * results.df_resid

        cov = N.zeros((self.order + 1,) + sum_sq.shape)
        cov[0] = sum_sq
        for i in range(1, self.order+1):
            cov[i] = N.add.reduce(resid[i:] * resid[0:-i], 0)
        cov = N.dot(self.invM, cov)
        output = cov[1:] * recipr(cov[0])
        return N.squeeze(output)



## this makes sense only if the labels are [slice, other] labels...
## no rush for this yet
    

## class FWHMOutput(Output):

##     def __init__(self, fmri, labels=None):
##         Output.__init__(self, labels=labels)
##         self.ndim = fmri.shape[0]
##         self.fwhmest = iter(iterFWHM(fmri.frame(0)))

##     def next(self, data=None, iterator=None):
##         if hasattr(iterator, 'newslice'): # for LabelledFmriImage class
##             if iterator.newslice:
##                 self.fwhmest.next(data=iterator.buffer)
##         else:
##             self.fwhmest.next(data=data)

##         del(data) ; gc.collect()

##     def extract(self, results):
##         return results.norm_resid
