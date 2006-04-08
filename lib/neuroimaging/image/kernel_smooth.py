import gc
import numpy as N
import numpy.dft as FFT
import numpy.linalg as NL
import neuroimaging.reference.grid as grid
import neuroimaging.reference.mapping as mapping
import neuroimaging.image as image
import enthought.traits as traits

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
        self.kernel = FFT.real_fftnd(self.kernel)

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

        if not isinstance(self.grid.mapping, mapping.Affine):
            raise ValueError, 'for FFT smoothing, need a regular (affine) grid'

        ndim = len(self.grid.shape)
        center = N.zeros((ndim,), N.Float)
        voxels = N.indices(self.shape).astype(N.Float)

        for i in range(voxels.shape[0]):
            test = N.less(voxels[i], self.shape[i] / 2.)
            voxels[i] = test * voxels[i] + (1. - test) * (voxels[i] - self.shape[i])
    
        voxels.shape = (voxels.shape[0], N.product(voxels.shape[1:]))
        X = self.grid.mapping.map(voxels)

        if self.fwhm is not 1.0:
            X = X / image.utils.fwhm2sigma(self.fwhm)
        if self.cov is not None:
            _chol = NL.cholesky_decomposition(cov)
            X = N.dot(NL.inverse(_chol), X)
        D2 = N.add.reduce(X**2, 0)
        D2.shape = self.shape
        return D2

    def smooth(self, inimage, indata=None, outdata=None, is_fft=False, scale=1.0, location=0.0, clean=True, **keywords):

        # check grids here? we should...
        
        if inimage.grid != self.grid:
            raise ValueError, 'grids do not agree in kernel_smooth'

        ndim = len(inimage.grid.shape)
        if ndim == 4:
            _out = N.zeros(inimage.shape)

        if indata is None:
            if ndim == 4:
                loop = iter(range(inimage.shape[0]))
                indata = inimage.getslice(slice(0,1))
            elif ndim == 3:
                indata = inimage.readall()
                loop = iter(range(1))
            else:
                raise NotImplementedError, 'expecting either 3 or 4-d image.'

        _slice = 0
        if ndim == 3:
            nslice = 1
        else:
            nslice = inimage.shape[0]

        for i in range(nslice):
            if clean:
                indata = N.nan_to_num(indata)
            if not is_fft:
                Y = self.presmooth(indata)
                tmp = Y * self.kernel
            else:
                tmp = indata * self.kernel

            tmp2 = FFT.inverse_real_fftnd(tmp)
            del(tmp)
            gc.collect()
            if scale is not 1.0:
                outdata = scale * tmp2[0:inimage.shape[0],0:inimage.shape[1],0:inimage.shape[2]]
            else:
                outdata = tmp2[0:inimage.shape[0],0:inimage.shape[1],0:inimage.shape[2]]

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

            # Read in next image
            if ndim == 4:
                indata = inimage.getslice(slice(_slice,_slice+1))

        gc.collect()

        if ndim == 3:
            return image.Image(_out, grid=self.grid)
        else:
            return image.Image(_out, grid=grid.DuplicatedGrids([self.grid]*inimage.grid.shape[0]))


    def presmooth(self, indata):
        _buffer = N.zeros(self.shape, N.Float)
        _buffer[0:indata.shape[0],0:indata.shape[1],0:indata.shape[2]] = indata
        return FFT.real_fftnd(_buffer)
