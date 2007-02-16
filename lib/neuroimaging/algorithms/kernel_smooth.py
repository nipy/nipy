"""
TODO
"""
__docformat__ = 'restructuredtext'

import gc

import numpy as N
import numpy.fft as fft
import numpy.linalg as L

from neuroimaging.core.api import Image, Affine

class LinearFilter(object):
    '''
    A class to implement some FFT smoothers for Image objects.
    By default, this does a Gaussian kernel smooth. More choices
    would be better!
    '''

    normalization = 'l1sum'
    
    def __init__(self, grid, fwhm=6.0, scale=1.0, location=0.0,
                 cov=None):
        """
        :Parameters:
            grid : TODO
                TODO
            fwhm : ``float``
                TODO
            scale : ``float``
                TODO
            location : ``float``
                TODO
        """
        
        self.grid = grid
        self.fwhm = fwhm
        self.scale = scale
        self.location = location
        self.cov = cov
        self._setup_kernel()

    def _setup_kernel(self):
        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'for FFT smoothing, need a regular (affine) grid'

        voxels = N.indices(self.grid.shape).astype(N.float64)

        center = N.asarray(self.grid.shape)/2
        center = self.grid.mapping([[center[i]] for i in range(len(self.grid.shape))])

        voxels.shape = (voxels.shape[0], N.product(voxels.shape[1:]))
        X = self.grid.mapping(voxels) - center
        X.shape = (3,) + self.grid.shape
        kernel = self(X)
        
        kernel = _crop(kernel)
        self.norms = {'l2':N.sqrt((kernel**2).sum()),
                      'l1':N.fabs(kernel).sum(),
                      'l1sum':kernel.sum()}

        self._kernel = kernel

        self.shape = (N.ceil((N.asarray(self.grid.shape) +
                              N.asarray(kernel.shape))/2)*2+2)
        self.fkernel = N.zeros(self.shape)
        slices = [slice(0, kernel.shape[i]) for i in range(len(kernel.shape))]
        self.fkernel[slices] = kernel
        self.fkernel = fft.rfftn(self.fkernel)

        return kernel

    def _normsq(self, X):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear. 
        """

        _X = N.copy(X)
        if self.fwhm is not 1.0:
            f = fwhm2sigma(self.fwhm)
            if f.shape == ():
                f = N.ones(len(self.grid.shape)) * f
            for i in range(len(self.grid.shape)):
                _X[i] /= f[i]
        if self.cov != None:
            _chol = L.cholesky(self.cov)
            _X = N.dot(L.inv(_chol), _X)
        D2 = N.add.reduce(_X**2, 0)
        D2.shape = X.shape[1:]
        return D2


    def __call__(self, X):
        _normsq = self._normsq(X) / 2.
        t = N.less_equal(_normsq, 15)
        return N.exp(-N.minimum(_normsq, 15)) * t

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
            _out = N.zeros(inimage.shape)
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
                data = N.nan_to_num(data)
            if not is_fft:
                data = self._presmooth(data)
                data *= self.fkernel 
            else:
                data *= self.fkernel

            data = fft.irfftn(data) / self.norms[self.normalization]

            gc.collect()
            _dslice = [slice(0, self.grid.shape[i], 1) for i in range(3)]
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
        _out = _out[[slice(self._kernel.shape[i]/2, self.grid.shape[i] +
                           self._kernel.shape[i]/2) for i in range(len(self.grid.shape))]]
        if inimage.ndim == 3:
            return Image(_out, grid=self.grid)
        else:
            return Image(_out, grid=self.grid.replicate(inimage.grid.shape[0]))


    def _presmooth(self, indata):
        slices = [slice(0, self.grid.shape[i], 1) for i in range(len(self.shape))]
        _buffer = N.zeros(self.shape)
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
    return fwhm / N.sqrt(8 * N.log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value.

    :Parameters:
        sigma : ``float``

    :Returns: ``float``
    """
    return sigma * N.sqrt(8 * N.log(2))



def _crop(X, tol=1.0e-10):
    """
    Find a bounding box for support of fabs(X) > tol and returned
    crop region.
    """
    
    aX = N.fabs(X)
    n = len(X.shape)
    I = N.indices(X.shape)[:, N.greater(aX, tol)]
    if I.shape[1] > 0:
        m = [I[i].min() for i in range(n)]
        M = [I[i].max() for i in range(n)]
        slices = [slice(m[i], M[i]+1, 1) for i in range(n)]
        return X[slices]
    else:
        return N.zeros((1,)*n)

