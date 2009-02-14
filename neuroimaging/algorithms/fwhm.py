"""
This module provides classes and definitions for using full width at half
maximum (FWHM) to be used in conjunction with Gaussian Random Field Theory
to determine resolution elements (resels).

A resolution element (resel) is defined as a block of pixels of the same
size as the FWHM of the smoothed image.

There are two methods implemented to estimate (3d, or volumewise) FWHM
based on a 4d Image:

    fastFHWM: used if the entire 4d Image is available
    iterFWHM: used when 4d Image is being filled in by slices of residuals
    
"""

__docformat__ = 'restructuredtext'

import gc

import numpy as np
from numpy.linalg import det
from neuroimaging.fixes.scipy.stats.models.utils import recipr

from neuroimaging.core.api import Image, Affine, CoordinateSystem

class Resels(object):
    """The Resels class.
    """
    def __init__(self, coordmap, normalized=False, fwhm=None, resels=None,
                 mask=None, clobber=False, D=3):
        """
        :Parameters:
            coordmap : ``CoordinateMap``
                 CoordinateMap over which fwhm and resels are to be estimated.                
                 Used in fwhm/resel conversion.
            fwhm : ``Image``
                Optional Image of FWHM. Used to convert
                FWHM Image to resels if FWHM is not being estimated.
            resels : ``Image``
                Optional Image of resels. Used to 
                compute resels within a mask, for instance, if
                FWHM has already been estimated.
            mask : ``Image``
                Mask over which to integrate resels.
            clobber : ``bool``
                Clobber output FWHM and resel images?
            D : ``int``
                Can be 2 or 3, the dimension of the final volume.
        """
        self.fwhm = fwhm
        self.resels = resels
        self.mask = mask
        self.clobber = clobber
        self.coordmap = coordmap
        self.D = D
        self.normalized = normalized
        _transform = self.coordmap.affine
        self.wedge = np.power(np.fabs(det(_transform)), 1./self.D)

    def integrate(self, mask=None):
        """
        :Parameters:
            mask : ``Image``
                  Optional mask over which to integrate (add) resels.
        
        :Returns: (total_resels, FWHM, nvoxel)

                  total_resels: the resels contained in the mask
                  FWHM: an estimate of FWHM based on the average resel per voxel
                  nvoxel: the number of voxels in the mask 
        """
        _resels = self.resels[:]
        if mask is not None:
            _mask = mask
        else:
            _mask = self.mask
        if _mask is not None:
            _mask = _mask[:].astype(np.int32)
            nvoxel = _mask.sum()
        else:
            _mask = 1.
            nvoxel = _resels.size
        _resels = (_resels * _mask).sum()
        _fwhm = self.resel2fwhm(_resels / nvoxel)
        return _resels, _fwhm, nvoxel

    def resel2fwhm(self, resels):
        """
        :Parameters:
            resels : ``float``
                Convert a resel value to an equivalent isotropic FWHM based on
                step sizes in self.coordmap.
        
        :Returns: FWHM
        """
        return np.sqrt(4*np.log(2.)) * self.wedge * recipr(np.power(resels, 1./self.D))

    def fwhm2resel(self, fwhm):
        """
        :Parameters:
            fwhm : ``float``
                Convert an FWHM value to an equivalent resels per voxel based on
                step sizes in self.coordmap.


        :Returns: resels
        """
        return recipr(np.power(fwhm / np.sqrt(4*np.log(2)) * self.wedge, self.D))

    def __iter__(self):
        """
        :Returns: ``self``
        """
        if not self.fwhm:
            im = Image(np.zeros(self.resid.shape), coordmap=self.coordmap)
        else:
            im = \
              Image(self.fwhm, clobber=self.clobber, mode='w', coordmap=self.coordmap)
        self.fwhm = im

        if not self.resels:
            im = Image(np.zeros(self.resid.shape), coordmap=self.coordmap)
        else:
            im = \
              Image(self.resels, clobber=self.clobber, mode='w', coordmap=self.coordmap)
        self.resels = im

        return self

class ReselImage(Resels):

    def __init__(self, resels=None, fwhm=None, **keywords):
        """
        :Parameters:
            resels : `core.api.Image`
                Image of resel per voxel values. 
            fwhm : `core.api.Image`
                Image of FWHM values.
            keywords : ``dict``
                Passed as keywords arguments to `core.api.Image`
        """
        if not resels and not fwhm:
            raise ValueError, 'need either a resels image or an FWHM image'

        if fwhm is not None:
            fwhm = Image(fwhm, **keywords)
            Resels.__init__(self, fwhm, resels=resels, fwhm=fwhm)

        if resels is not None:
            resels = Image(resels, **keywords)
            Resels.__init__(self, resels, resels=resels, fwhm=fwhm)

        if not self.fwhm:
            self.fwhm = Image(self.resel2fwhm(self.resels[:]),
                              coordmap=self.coordmap, **keywords)

        if not self.resels:
            self.resels = Image(self.fwhm2resel(self.fwhm[:]),
                                coordmap=self.coordmap, **keywords)

    def __iter__(self): 
        """
        :Returns: ``self``
        """
        return self

class iterFWHM(Resels):
    """
    Estimate FWHM on an image of residuals sequentially. This is handy when,
    say, residuals from a linear model are written out slice-by-slice.

    Resulting FWHM is clipped at self.FWHMmax, which defaults to 50.
    """

    FWHMmax=50.

    def __init__(self, coordmap, normalized=False, df_resid=5.0, mask=None, **keywords):
        """Setup a FWHM estimator.
        
        :Parameters:
            coordmap : ``CoordinateMap``
                 CoordinateMap over which fwhm and resels are to be estimated.                
                 Used in fwhm/resel conversion.
            normalized : ``bool``
                Are residuals normalized to have length 1? If False, residuals
                are normalized before estimating FWHM.
            df_resid : ``float``
                How many degrees of freedom are there in the residuals?
                Must be greater than self.D + 1.
            mask : ``Image``
                  Optional mask over which to integrate (add) resels.
            keywords : ``dict``
                Passed as keyword parameters to `Resels.__init__`
        """

        Resels.__init__(self, coordmap, mask=mask, **keywords)
        self.normalized = normalized
        self.Y = coordmap.shape[1]
        self.X = coordmap.shape[2]
        self._setup_nneigh()
        self._fwhm = np.zeros((self.Y, self.X))
        self._resels = np.zeros((self.Y, self.X))
        self.Yindex = np.arange(self.Y)
        self.Xindex = np.arange(self.X)
        self.YX = self.Y*self.X
        self.Yshift = np.array([0,1,0,1])
        self.Xshift = np.array([0,0,1,1])
        self.Yweight = -np.array([1,-1,1,-1.])
        self.Xweight = -np.array([1,1,-1,-1.])

        self.slices = []
        self.nslices = coordmap.shape[0]
        iter(self)
        if df_resid <= self.D + 1:
            raise ValueError, 'insufficient residual degrees of freedom to estimate FWHM'
        
    def __call__(self, resid):
        """
        Estimate FWHM and resels per voxel.

        :Parameters:
            resid : ``Image``
                Image of residuals used for estimating FWHM and resels per voxel.
        
        :Returns: ``None``
        """
        resid = resid.slice_iterator()
        iter(self)
        while True:
            try:
                data = resid.next()
                self.set_next(data=data)
            except StopIteration:
                break

    def normalize(self, resid):
        """
        Normalize residuals subtracting mean, and fixing length to 1.

        :Parameters:
            resid : Array of residuals.
                
        :Returns: Normalized residuals.
        """
        _mu = 0.
        _sumsq = 0.

        n = resid.shape[-1]

        for i in range(n):
            if self.D == 3:
                _frame = resid[:,:,i]
            elif self.D == 2:
                _frame = resid[:,i]
            _mu += _frame
            _sumsq += _frame**2

        _mu /= n
        _invnorm = recipr(np.sqrt((_sumsq - n * _mu**2)))

        value = np.zeros(resid.shape)

        for i in range(n):
            if self.D == 3:
                _frame = resid[:,:,i]
                _frame -= _mu 
                _frame *= _invnorm
                value[:,:,i] = _frame
            elif self.D == 2:
                _frame = resid[:,i]
                _frame -= _mu
                _frame *= _invnorm
                value[:,i] = _frame

        return value

    def __iter__(self):
        """
        :Returns: ``self``
        """
        Resels.__iter__(self)

        self.slice = 0
        self.slices = []
        return self

    def _setup_nneigh(self):
        """
        Setup the number of neighbours within a slice of size self.Y, self.X.

        :Returns: ``None``
        """
        self.nneigh = 4. * np.ones((self.Y, self.X))
        self.nneigh[0] = 2
        self.nneigh[-1] = 2
        self.nneigh[:, 0] = 2
        self.nneigh[:, -1] = 2
        self.nneigh[0, 0] = 1
        self.nneigh[0, -1] = 1
        self.nneigh[-1, -1] = 1
        self.nneigh[-1, 0] = 1

    def set_next(self, resid):
        """
        Pass a slice of residuals into slicewise estimate of FWHM.

        :Parameters:
            resid : ``array``
                 slice of residuals
        
        :Returns: ``None``
        """

        wresid = 1.0 * np.transpose(data, (1,2,0))
        if not self.normalized:
            wresid = self.normalize(wresid)

        if self.slice == 0:
            self.u = 1. * wresid
            self.ux = wresid[:, 1:] - wresid[:, :-1]
            self.uy = wresid[1:, :] - wresid[:-1, :]
            self.Axx = np.add.reduce(self.ux**2, 2)
            self.Ayy = np.add.reduce(self.uy**2, 2)

            if self.D == 2:
                for index in range(4):
                    y = self.Yindex + self.Yshift[index]
                    x = self.Xindex + self.Xshift[index]
                    yslice = slice(y[0], y[-1])
                    xslice = slice(x[0], x[-1])
                    axx = self.Axx[yslice, :]
                    ayy = self.Ayy[:, xslice]
                    axy = np.add.reduce(self.uy[:, xslice] *
                                       self.ux[yslice, :], 2)
                    axy *= self.Yweight[index] * self.Xweight[index]
                    detlam = _calc_detlam(axx, ayy, 1, ayx, 0, 0)
                    test = np.greater(detlam, 0).astype(np.float64)
                    _resels = np.sqrt(test * detlam)
                    self._resels[yslice, xslice] += _resels
                    self._fwhm[yslice, xslice] += self.resel2fwhm(_resels)

        else:
            self.uz = wresid - self.u
            self.Azz = np.add.reduce(self.uz**2, 2)

            # The 4 upper cube corners:
            for index in range(4):
                y = self.Yindex + self.Yshift[index] 
                x = self.Xindex + self.Xshift[index]
                yslice = slice(y[0], y[-1])
                xslice = slice(x[0], x[-1])
                axx = self.Axx[yslice, :]
                ayy = self.Ayy[:, xslice]
                azz = self.Azz[yslice, xslice]

                xx = self.ux[yslice: ]
                yy = self.uy[:, xslice]
                zz = self.uz[yslice, xslice]
                ayx = np.add.reduce(yy * xx, 2)
                azx = np.add.reduce(xx * zz, 2)
                azy = np.add.reduce(yy * zz, 2)
                ayx *= self.Yweight[index] * self.Xweight[index]
                azx *= self.Yweight[index]
                azy *= self.Xweight[index]
                detlam = _calc_detlam(axx, ayy, azz, ayx, azx, azy)

                test = np.greater(detlam, 0).astype(np.float64)
                _resels = np.sqrt(test * detlam)
                self._resels[yslice, xslice] += _resels
                self._fwhm[yslice, xslice] += self.resel2fwhm(_resels)
                
            # Get slice ready for output
            self._fwhm /= ((self.slice > 1) + 1.) * self.nneigh
            self._resels /= ((self.slice > 1) + 1.) * self.nneigh

            # Clear buffers

            self._fwhm = np.zeros(self._fwhm.shape)
            self._resels = np.zeros(self._resels.shape)
            self.u = 1. * wresid
            self.ux = wresid[:, 1:] - wresid[:, :-1]
            self.uy = wresid[1:, :] - wresid[:-1, :]
            self.Axx = np.add.reduce(self.ux**2, 2)
            self.Ayy = np.add.reduce(self.uy**2, 2)

            # The 4 lower cube corners:

            for index in range(4):
                x = self.Yindex + self.Yshift[index]
                y = self.Xindex + self.Xshift[index]
                yslice = slice(y[0], y[-1])
                xslice = slice(x[0], x[-1])
                axx = self.Axx[xslice, :]
                ayy = self.Ayy[:, yslice]
                azz = self.Azz[xslice, yslice]

                xx = self.ux[xslice, :]
                yy = self.uy[:, yslice]
                zz = self.uz[xslice, yslice]
                ayx = np.add.reduce(yy * xx, 2)
                azx = np.add.reduce(xx * zz, 2)
                azy = np.add.reduce(yy * zz, 2)
                
                ayx *= self.Yweight[index] * self.Xweight[index]
                azx *= self.Yweight[index]
                azy *= self.Xweight[index]
                detlam = _calc_detlam(axx, ayy, azz, ayx, azx, azy)

                test = np.greater(np.fabs(detlam), 0).astype(np.float64)
                _resels = np.sqrt(test * detlam)
                self._resels[xslice, yslice] += _resels
                self._fwhm[xslice, yslice] += self.resel2fwhm(_resels)
                
            if self.slice == self.nslices - 1:
                self._fwhm /= ((self.slice > 1) + 1.) * self.nneigh
                self._resels /= ((self.slice > 1) + 1.) * self.nneigh

                self.output()
        
        self.slice += 1

    def output(self):
        """
        :Returns: ``None``
        """
        value = self.coordmap.next()

        self.fwhm[value.slice] =  np.clip(self._fwhm, 0, self.FWHMmax)
        self.resels[value.slice] = np.clip(self._resels, 0, self.FWHMmax)

class fastFWHM(Resels):

    def __init__(self, resid, **keywords):
        """
        Given a 4d image of residuals, i.e. not one filled in step by step
        by an iterator, estimate FWHM and resels.

        :Parameters:
            resid : ``array``
                 Image of residuals used to estimate FWHM and resels per voxel.
        
        :Returns: ``None``
        """

        cm = Affine(resid.coordmap.affine[1:,1:], 
                    CoordinateSystem(resid.coordmap.input_coords.coordinates[1:]), 
                    CoordinateSystem(resid.coordmap.output_coords.coordinates[1:]))

        Resels.__init__(self, cm, **keywords)
        self.n = resid.shape[0]
        self.rarray = np.asarray(resid)
        self.resid = Image(self.rarray, resid.coordmap)

    def __call__(self):
        """
        Estimate FWHM and resels per voxel.
        
        :Returns: ``None``
        """

        Resels.__iter__(self)

        _mu = 0.
        _sumsq = 0.

        if not self.normalized:
            for i in range(self.n):
                _frame = self.rarray[slice(i, i+1)]
                _mu += _frame
                _sumsq += _frame**2

            _mu /= self.n
            _invnorm = recipr(np.sqrt((_sumsq - self.n * _mu**2)))
        else:
            _mu = 0.
            _invnorm = 1.


        Lzz = Lzy = Lzx = Lyy = Lyx = Lxx = 0.        
        for i in range(self.n):
            verbose = True
            if verbose:
                print 'Slice: [%d]' % i
            _frame = self.rarray[slice(i, i+1)]
            _frame = (_frame - _mu) * _invnorm
            _frame.shape = _frame.shape[1:]


            g = np.gradient(_frame)
            dz, dy, dx = g[0], g[1], g[2]

            del(_frame) 

            Lzz += dz * dz
            Lzy += dz * dy
            Lzx += dz * dx
            Lyy += dy * dy
            Lyx += dy * dx
            Lxx += dx * dx

            del(dz); del(dy); del(dx); del(g) ; gc.collect()
            
        detlam = _calc_detlam(Lxx, Lyy, Lzz, Lyx, Lzx, Lzy)
        print detlam.shape


        test = np.greater(detlam, 0)
        resels = np.sqrt(detlam * test)
        print resels.shape, self.resels.shape
        fwhm = self.resel2fwhm(resels)

        fullslice = [slice(0, x) for x in self.resid.shape]
        self.resels = Image(resels, self.coordmap)
        self.fwhm = Image(fwhm, self.coordmap)


def _calc_detlam(xx, yy, zz, yx, zx, zy):
    """
    Calculate determinant of symmetric 3x3 matrix

    [[xx,yx,xz],
     [yx,yy,zy],
     [zx,zy,zz]]
     """
    
    return zz * (yy*xx - yx**2) - \
           zy * (zy*xx - zx*yx) + \
           zx * (zy*yx - zx*yy)
