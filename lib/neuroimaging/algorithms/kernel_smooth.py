__docformat__ = 'restructuredtext'

import gc

import numpy as N
import numpy.fft as fft
import numpy.linalg as NL

from neuroimaging.core.api import Image, Affine

class LinearFilter(object):
    '''
    A class to implement some FFT smoothers for Image objects.
    By default, this does a Gaussian kernel smooth. More choices
    would be better!
    '''

    def __init__(self, grid, fwhm=6.0, padding=5, scale=1.0, location=0.0,
                 cov=None):

        self.grid = grid
        self.fwhm = fwhm
        self.padding = padding
        self.scale = scale
        self.location = location
        self.cov = cov
        self.shape = N.array(self.grid.shape) + self.padding
        self._setup_kernel()

    def _setup_kernel(self):
        _normsq = self._normsq() / 2.
        self.kernel = N.exp(-N.minimum(_normsq, 15))
        norm = N.sqrt((self.kernel**2).sum())
        self.kernel = self.kernel / norm
        self.kernel = fft.rfftn(self.kernel)


    def _normsq(self):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear. 
        """

        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'for FFT smoothing, need a regular (affine) grid'

        voxels = N.indices(self.shape).astype(N.float64)

        for i in range(voxels.shape[0]):
            test = N.less(voxels[i], self.shape[i] / 2.)
            voxels[i] = test * voxels[i] + \
                        (1. - test) * (voxels[i] - self.shape[i])
    
        voxels.shape = (voxels.shape[0], N.product(voxels.shape[1:]))
        X = self.grid.mapping(voxels)

        if self.fwhm is not 1.0:
            X = X / fwhm2sigma(self.fwhm)
        if self.cov != None:
            _chol = NL.cholesky(self.cov)
            X = N.dot(NL.inv(_chol), X)
        D2 = N.add.reduce(X**2, 0)
        D2.shape = self.shape
        return D2

    def smooth(self, inimage, clean=False, is_fft=False):

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
                data *= self.kernel
            else:
                data *= self.kernel

            data = fft.irfftn(data)

            gc.collect()
            if self.scale != 1:
                data = self.scale * data[:inimage.shape[0],
                                         :inimage.shape[1],
                                         :inimage.shape[2]]

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
        _out = _out[[slice(0, n) for n in self.grid.shape]]
        if inimage.ndim == 3:
            return Image(_out, grid=self.grid)
        else:
            return Image(_out, grid=self.grid.replicate(inimage.grid.shape[0]))

    def _presmooth(self, indata):
        _buffer = N.zeros(self.shape)
        _buffer[:indata.shape[0],
                :indata.shape[1],
                :indata.shape[2]] = indata
        return fft.rfftn(_buffer)


def fwhm2sigma(fwhm):
    """
    Convert a FWHM value to sigma in a Gaussian kernel.
    """
    return fwhm / N.sqrt(8 * N.log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value.
    """
    return sigma * N.sqrt(8 * N.log(2))



if __name__ == '__main__':
    from pylab import show
    a = 100*N.random.random((100, 100, 100))
    img = Image(a)
    filt = LinearFilter(img.grid)
    smoothed = filt.smooth(img)

    from neuroimaging.ui.visualization.viewer import BoxViewer

    view = BoxViewer(img)
    view.draw()
    sview = BoxViewer(smoothed, colormap='jet')
    sview.draw()
    show()

