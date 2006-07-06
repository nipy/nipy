import gc

import numpy as N
import numpy.dft as FFT
import numpy.linalg as NL
from neuroimaging import traits

from neuroimaging.image import Image
from neuroimaging.image.utils import fwhm2sigma
from neuroimaging.reference.mapping import Affine

class LinearFilter(traits.HasTraits):
    '''
    A class to implement some FFT smoothers for VImage objects.
    '''

    padding = traits.Int(5)
    fwhm = traits.Float(6.)
    norm = traits.Float(2.)
    periodic = traits.true
    cov = traits.Any()

    def setup_kernel(self):
        _normsq = self.normsq() / 2.
        self.kernel = N.exp(-N.minimum(_normsq, 15))
        norm = self.kernel.sum()
        self.kernel = self.kernel / norm
        self.kernel = FFT.rfftn(self.kernel)

    def __init__(self, grid, **keywords):

        traits.HasTraits.__init__(self, **keywords)
        self.grid = grid
        self.shape = N.array(self.grid.shape) + self.padding
        self.setup_kernel()

    def normsq(self):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear. 
        """

        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'for FFT smoothing, need a regular (affine) grid'

        voxels = N.indices(self.shape).astype(N.float64)

        for i in range(voxels.shape[0]):
            test = N.less(voxels[i], self.shape[i] / 2.)
            voxels[i] = test * voxels[i] + (1. - test) * (voxels[i] - self.shape[i])
    
        voxels.shape = (voxels.shape[0], N.product(voxels.shape[1:]))
        X = self.grid.mapping.map(voxels)

        if self.fwhm is not 1.0:
            X = X / fwhm2sigma(self.fwhm)
        if self.cov is not None:
            _chol = NL.cholesky(self.cov)
            X = N.dot(NL.inv(_chol), X)
        D2 = N.add.reduce(X**2, 0)
        D2.shape = self.shape
        return D2

    def smooth(self, inimage, is_fft=False, scale=1.0, location=0.0, clean=True, **keywords):

        # check grids here? we should...
        
        if inimage.grid != self.grid:
            raise ValueError, 'grids do not agree in kernel_smooth'

        ndim = len(inimage.grid.shape)
        if ndim == 4:
            _out = N.zeros(inimage.shape)
            nslice = inimage.shape[0]
        elif ndim == 3:
            nslice = 1
        else:
            raise NotImplementedError, 'expecting either 3 or 4-d image.'

        for _slice in range(nslice):
            if ndim == 4:
                indata = inimage.getslice(slice(_slice,_slice+1))
            elif ndim == 3:
                indata = inimage.readall()

            if clean:
                indata = N.nan_to_num(indata)
            if not is_fft:
                Y = self.presmooth(indata)
                tmp = Y * self.kernel
            else:
                tmp = indata * self.kernel

            tmp2 = FFT.irfftn(tmp)
            del(tmp)
            gc.collect()
            outdata = scale * tmp2[0:inimage.shape[0],0:inimage.shape[1],0:inimage.shape[2]]

            if location != 0.0:
                outdata = outdata + location
            del(tmp2)
            gc.collect()

            # Write out data 

            if ndim == 4:
                _out[_slice] = outdata
            else:
                _out = outdata
            _slice = _slice + 1

        gc.collect()

        if ndim == 3:
            return Image(_out, grid=self.grid)
        else:
            return Image(_out, grid=self.grid.duplicate(inimage.grid.shape[0]))


    def presmooth(self, indata):
        _buffer = N.zeros(self.shape, N.float64)
        _buffer[0:indata.shape[0],0:indata.shape[1],0:indata.shape[2]] = indata
        return FFT.rfftn(_buffer)
