"""
TODO
"""
__docformat__ = 'restructuredtext'

import gc

import numpy as np
import numpy.fft as fft
import numpy.linalg as L

from neuroimaging.core.api import Image, Affine
from neuroimaging.core.reference.coordinate_map import replicate

class LinearFilter(object):
    '''
    A class to implement some FFT smoothers for Image objects.
    By default, this does a Gaussian kernel smooth. More choices
    would be better!
    '''

    normalization = 'l1sum'
    
    def __init__(self, coordmap, shape, fwhm=6.0, scale=1.0, location=0.0,
                 cov=None):
        """
        :Parameters:
            coordmap : TODO
                TODO
            fwhm : ``float``
                TODO
            scale : ``float``
                TODO
            location : ``float``
                TODO
        """
        
        self.coordmap = coordmap
        self.bshape = shape
        self.fwhm = fwhm
        self.scale = scale
        self.location = location
        self.cov = cov
        self._setup_kernel()

    def _setup_kernel(self):
        if not isinstance(self.coordmap, Affine):
            raise ValueError, 'for FFT smoothing, need a regular (affine) coordmap'

        voxels = np.indices(self.bshape).astype(np.float64)

        center = np.asarray(self.bshape)/2
        center = self.coordmap([center[i] for i in range(len(self.bshape))])

        voxels.shape = (voxels.shape[0], np.product(voxels.shape[1:]))
        X = (self.coordmap(voxels.T) - center).T
        X.shape = (self.coordmap.ndim[0],) + tuple(self.bshape)
        kernel = self(X)
        
        kernel = _crop(kernel)
        self.norms = {'l2':np.sqrt((kernel**2).sum()),
                      'l1':np.fabs(kernel).sum(),
                      'l1sum':kernel.sum()}

        self._kernel = kernel

        self.shape = (np.ceil((np.asarray(self.bshape) +
                              np.asarray(kernel.shape))/2)*2+2)
        self.fkernel = np.zeros(self.shape)
        slices = [slice(0, kernel.shape[i]) for i in range(len(kernel.shape))]
        self.fkernel[slices] = kernel
        self.fkernel = fft.rfftn(self.fkernel)

        return kernel

    def _normsq(self, X):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear. 
        """

        _X = np.copy(X)
        if self.fwhm is not 1.0:
            f = fwhm2sigma(self.fwhm)
            if f.shape == ():
                f = np.ones(len(self.bshape)) * f
            for i in range(len(self.bshape)):
                _X[i] /= f[i]
        if self.cov != None:
            _chol = L.cholesky(self.cov)
            _X = np.dot(L.inv(_chol), _X)
        D2 = np.add.reduce(_X**2, 0)
        D2.shape = X.shape[1:]
        return D2


    def __call__(self, X):
        _normsq = self._normsq(X) / 2.
        t = np.less_equal(_normsq, 15)
        return np.exp(-np.minimum(_normsq, 15)) * t

    def smooth(self, inimage, clean=False, is_fft=False):
        """
        :Parameters:
            inimage : `core.api.Image`
                The image to be smoothed
            clean : ``bool``
                Should we call ``nan_to_num`` on the data before smoothing?
            is_fft : ``bool``
                Has the data already been fft'd?

        :Returns: `Image`
        """
        if inimage.ndim == 4:
            _out = np.zeros(inimage.shape)
            nslice = inimage.shape[0]
        elif inimage.ndim == 3:
            nslice = 1
        else:
            raise NotImplementedError, 'expecting either 3 or 4-d image.'

        for _slice in range(nslice):
            if inimage.ndim == 4:
                data = inimage[_slice]
            elif inimage.ndim == 3:
                data = inimage[:]

            if clean:
                data = np.nan_to_num(data)
            if not is_fft:
                data = self._presmooth(data)
                data *= self.fkernel 
            else:
                data *= self.fkernel

            data = fft.irfftn(data) / self.norms[self.normalization]

            gc.collect()
            _dslice = [slice(0, self.bshape[i], 1) for i in range(3)]
            if self.scale != 1:
                data = self.scale * data[_dslice]

            if self.location != 0.0:
                data += self.location

            gc.collect()

            # Write out data 

            if inimage.ndim == 4:
                _out[_slice] = data
            else:
                _out = data
            _slice += 1

        gc.collect()
        _out = _out[[slice(self._kernel.shape[i]/2, self.bshape[i] +
                           self._kernel.shape[i]/2) for i in range(len(self.bshape))]]
        if inimage.ndim == 3:
            return Image(_out, coordmap=self.coordmap)
        else:
            return Image(_out, coordmap=replicate(self.coordmap, inimage.shape[0]))


    def _presmooth(self, indata):
        slices = [slice(0, self.bshape[i], 1) for i in range(len(self.shape))]
        _buffer = np.zeros(self.shape)
        _buffer[slices] = indata
        return fft.rfftn(_buffer)


def fwhm2sigma(fwhm):
    """
    Convert a FWHM value to sigma in a Gaussian kernel.

    :Parameters:
        fwhm : ``float``
            TODO

    :Returns: ``float``
    """
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value.

    :Parameters:
        sigma : ``float``

    :Returns: ``float``
    """
    return sigma * np.sqrt(8 * np.log(2))

def _crop(X, tol=1.0e-10):
    """
    Find a bounding box for support of fabs(X) > tol and returned
    crop region.
    """
    
    aX = np.fabs(X)
    n = len(X.shape)
    I = np.indices(X.shape)[:, np.greater(aX, tol)]
    if I.shape[1] > 0:
        m = [I[i].min() for i in range(n)]
        M = [I[i].max() for i in range(n)]
        slices = [slice(m[i], M[i]+1, 1) for i in range(n)]
        return X[slices]
    else:
        return np.zeros((1,)*n)

