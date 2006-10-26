"""
This module provides classes and definitions for using full width at half
maximum (FWHM) to be used in conjunction with Gaussian Random Field Theory
to determine resolution elements (resels).

A resolution element (resel) is defined as a block of pixels of the same
size as the FWHM of the smoothed image.
"""

import gc

import numpy as N
from numpy.linalg import det
from scipy.sandbox.models.utils import recipr

from neuroimaging.core.image.image import Image

class Resels(object):
    
    def __init__(self, grid, normalized=False, fwhm=None, resels=None,
                 mask=None, clobber=False, D=3):
        
        self.fwhm = fwhm
        self.resels = resels
        self.mask = mask
        self.clobber = clobber
        self.grid = grid
        self.D = D

        _transform = self.grid.mapping.transform
        self.wedge = N.power(N.fabs(det(_transform)), 1./self.D)
        self.normalized = normalized

    def integrate(self, mask=None):
        _resels = self.resels.readall()
        if mask is not None:
            _mask = mask
        else:
            _mask = self.mask
        if _mask is not None:
            _mask = _mask.readall().astype(N.int32)
            nvoxel = _mask.sum()
        else:
            _mask = 1.
            nvoxel = _resels.size
        _resels = (_resels * _mask).sum()
        _fwhm = self.resel2fwhm(_resels / nvoxel)
        return _resels, _fwhm, nvoxel

    def resel2fwhm(self, x):
        return N.sqrt(4*N.log(2.)) * self.wedge * recipr(N.power(x, 1./self.D))

    def fwhm2resel(self, x):
        return recipr(N.power(x / N.sqrt(4*N.log(2)) * self.wedge, self.D))

    def __iter__(self):
        if not self.fwhm:
            im = Image(N.zeros(self.grid.shape, N.float64), grid=self.grid)
        else:
            im = \
              Image(self.fwhm, clobber=self.clobber, mode='w', grid=self.grid)
        self.fwhm = iter(im)

        if not self.resels:
            im = Image(N.zeros(self.grid.shape, N.float64), grid=self.grid)
        else:
            im = \
              Image(self.resels, clobber=self.clobber, mode='w', grid=self.grid)
        self.resels = iter(im)

        return self

class ReselImage(Resels):

    def __init__(self, resels=None, fwhm=None, **keywords):

        if not resels and not fwhm:
            raise ValueError, 'need either a resels image or an FWHM image'

        if fwhm:
            fwhm = Image(fwhm, **keywords)
            Resels.__init__(self, fwhm, resels=resels, fwhm=fwhm)

        if resels:
            resels = Image(resels, **keywords)
            Resels.__init__(self, resels, resels=resels, fwhm=fwhm)

        if not self.fwhm:
            self.fwhm = Image(self.resel2fwhm(self.resels.readall()),
                              mapping=self.resels.grid.mapping, **keywords)

        if not self.resels:
            self.resels = Image(self.fwhm2resel(self.fwhm.readall()),
                                mapping=self.fwhm.grid.mapping, **keywords)

    def __iter__(self):
        return

class iterFWHM(Resels):
    """
    Estimate FWHM on an image of residuals sequentially. This is handy when,
    say, residuals from a linear model are written out slice-by-slice.
    """

    def __init__(self, grid, df_resid=5.0, df_limit=4.0, mask=None, **keywords):
        '''Setup a FWHM estimator.'''

        Resels.__init__(self, grid, mask=mask, **keywords)
        self.df_resid = df_resid
        self.df_limit = df_limit
        self.Y = grid.shape[1]
        self.X = grid.shape[2]
        self.setup_nneigh()
        self._fwhm = N.zeros((self.Y,self.X), N.float64)
        self._resels = N.zeros((self.Y,self.X), N.float64)
        self.Yindex = N.arange(self.Y)
        self.Xindex = N.arange(self.X)
        self.YX = self.Y*self.X
        self.Yshift = N.array([0,1,0,1])
        self.Xshift = N.array([0,0,1,1])
        self.Yweight = -N.array([1,-1,1,-1.])
        self.Xweight = -N.array([1,1,-1,-1.])

        self.slices = []
        self.nslices = grid.shape[0]
        iter(self)

    def __call__(self, resid):
        resid = iter(resid)
        iter(self)
        while True:
            try:
                data = resid.next()
                self.set_next(data=data)
            except StopIteration:
                break

    def normalize(self, resid):

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

        _mu = _mu / n
        _invnorm = recipr(N.sqrt((_sumsq - n * _mu**2)))

        value = N.zeros(resid.shape, N.float64)

        for i in range(n):
            if self.D == 3:
                _frame = resid[:,:,i]
                _frame = (_frame - _mu) * _invnorm
                value[:,:,i] = _frame
            elif self.D == 2:
                _frame = resid[:,i]
                _frame = (_frame - _mu) * _invnorm
                value[:,i] = _frame

        return value

    def __iter__(self):
        Resels.__iter__(self)

        self.slice = 0
        self.slices = []
        return self

    def setup_nneigh(self):
        self.nneigh = 4. * N.ones((self.Y,self.X))
        self.nneigh[0] = 2
        self.nneigh[-1] = 2
        self.nneigh[:,0] = 2
        self.nneigh[:,-1] = 2
        self.nneigh[0,0] = 1
        self.nneigh[0,-1] = 1
        self.nneigh[-1,-1] = 1
        self.nneigh[-1,0] = 1

    def set_next(self, data, FWHMmax=50.):

        wresid = 1.0 * N.transpose(data, (1,2,0))
        if not self.normalized:
            wresid = self.normalize(wresid)

        if self.slice == 0:
            self.u = 1. * wresid
            self.ux = wresid[:,1:] - wresid[:,0:-1]
            self.uy = wresid[1:,:] - wresid[0:-1,:]
            self.Axx = N.add.reduce(self.ux**2, 2)
            self.Ayy = N.add.reduce(self.uy**2, 2)

            if self.D == 2:
                for index in range(4):
                    y = self.Yindex + self.Yshift[index]
                    x = self.Xindex + self.Xshift[index]
                    axx = self.Axx[y[0]:y[-1],:]
                    ayy = self.Ayy[:,x[0]:x[-1]]
                    axy = N.add.reduce(self.uy[:,x[0]:x[-1]] * self.ux[y[0]:y[-1],:], 2) * self.Yweight[index] * self.Xweight[index]
                    detlam = axx * ayy - axy**2
                    test = N.greater(detlam, 0).astype(N.float64)
                    _resels = N.sqrt(test * detlam)
                    self._resels[y[0]:y[-1], x[0]:x[-1]] = self._resels[y[0]:y[-1], x[0]:x[-1]] + _resels
                    self._fwhm[y[0]:y[-1], x[0]:x[-1]] = self._fwhm[y[0]:y[-1], x[0]:x[-1]] + self.resel2fwhm(_resels)

        else:
            self.uz = wresid - self.u
            self.Azz = N.add.reduce(self.uz**2, 2)

            # The 4 upper cube corners:
            for index in range(4):
                y = self.Yindex + self.Yshift[index] 
                x = self.Xindex + self.Xshift[index]
                axx = self.Axx[y[0]:y[-1],:]
                ayy = self.Ayy[:,x[0]:x[-1]]
                azz = self.Azz[y[0]:y[-1],x[0]:x[-1]]
                if self.df_resid > self.df_limit:
                    ayx = N.add.reduce(self.uy[:,x[0]:x[-1]] * self.ux[y[0]:y[-1],:], 2) * self.Yweight[index] * self.Xweight[index]
                    azx = N.add.reduce(self.ux[y[0]:y[-1],:] * self.uz[y[0]:y[-1],x[0]:x[-1]], 2) * self.Yweight[index]
                    azy = N.add.reduce(self.uy[:,x[0]:x[-1]] * self.uz[y[0]:y[-1],x[0]:x[-1]], 2) * self.Xweight[index]
                    detlam = azz * (ayy*axx - ayx**2) - azy * (azy*axx - azx*ayx) + azx * (azy*ayx - azx*ayy)

                test = N.greater(detlam, 0).astype(N.float64)
                _resels = N.sqrt(test * detlam)
                self._resels[y[0]:y[-1], x[0]:x[-1]] = self._resels[y[0]:y[-1], x[0]:x[-1]] + _resels
                self._fwhm[y[0]:y[-1], x[0]:x[-1]] = self._fwhm[y[0]:y[-1], x[0]:x[-1]] + self.resel2fwhm(_resels)
                
            # Get slice ready for output
            self._fwhm = self._fwhm / (((self.slice > 1) + 1.) * self.nneigh)
            self._resels = self._resels / (((self.slice > 1) + 1.) * self.nneigh)

            self.output(FWHMmax)

            # Clear buffers

            self._fwhm = N.zeros(self._fwhm.shape, N.float64)
            self._resels = N.zeros(self._resels.shape, N.float64)
            self.u = 1. * wresid
            self.ux = wresid[:,1:] - wresid[:,0:-1]
            self.uy = wresid[1:,:] - wresid[0:-1,:]
            self.Axx = N.add.reduce(self.ux**2, 2)
            self.Ayy = N.add.reduce(self.uy**2, 2)

            # The 4 lower cube corners:

            for index in range(4):
                x = self.Yindex + self.Yshift[index]
                y = self.Xindex + self.Xshift[index]
                axx = self.Axx[x[0]:x[-1],:]
                ayy = self.Ayy[:,y[0]:y[-1]]
                azz = self.Azz[x[0]:x[-1],y[0]:y[-1]]
                if self.df_resid > self.df_limit:
                    ayx = N.add.reduce(self.uy[:,y[0]:y[-1]] * self.ux[x[0]:x[-1],:], 2) * self.Yweight[index] * self.Xweight[index]
                    azx = N.add.reduce(self.ux[x[0]:x[-1],:] * self.uz[x[0]:x[-1],y[0]:y[-1]], 2) * self.Yweight[index]
                    azy = N.add.reduce(self.uy[:,y[0]:y[-1]] * self.uz[x[0]:x[-1],y[0]:y[-1]], 2) * self.Xweight[index]
                    detlam = azz * (ayy*axx - ayx**2) - azy * (azy*axx - azx*ayx) + azx * (azy*ayx - azx*ayy)
                test = N.greater(N.fabs(detlam), 0).astype(N.float64)
                _resels = N.sqrt(test * detlam)
                self._resels[x[0]:x[-1], y[0]:y[-1]] = self._resels[x[0]:x[-1], y[0]:y[-1]] + _resels
                self._fwhm[x[0]:x[-1], y[0]:y[-1]] = self._fwhm[x[0]:x[-1], y[0]:y[-1]] + self.resel2fwhm(_resels)
                
            if self.slice == self.nslices - 1:
                
                self._fwhm = self._fwhm / ((self.slice > 1) + 1.) / self.nneigh
                self._resels = self._resels / (((self.slice > 1) + 1.) * self.nneigh)

                self.output()
        
        self.slice = self.slice + 1

    def output(self, FWHMmax=50.):
        value = self.grid.next()

        self.fwhm[value.slice] =  N.clip(self._fwhm, 0, FWHMmax)
        self.resels[value.slice] = N.clip(self._resels, 0, FWHMmax)

class fastFWHM(Resels):
    """
    Given a 4d image of residuals, i.e. not one filled in step by step
    by an iterator, estimate fwhm and resels.
    """

    def __init__(self, rimage, **keywords):
        Resels.__init__(self, rimage.grid.subgrid(0), **keywords)
        self.n = rimage.grid.shape[0]
        self.rimage = Image(rimage)

    def __call__(self, verbose=False):

        Resels.__iter__(self)

        _mu = 0.
        _sumsq = 0.

        if not self.normalized:
            for i in range(self.n):
                if verbose:
                    print '(Normalizing) Frame: [%d]' % i
                _frame = self.rimage[slice(i,i+1)]
                _mu = _mu + _frame
                _sumsq += _frame**2

            _mu = _mu / self.n
            _invnorm = recipr(N.sqrt((_sumsq - self.n * _mu**2)))
        else:
            _mu = 0.
            _invnorm = 1.


        Lzz = Lzy = Lzx = Lyy = Lyx = Lxx = 0.        
        for i in range(self.n):
            if verbose:
                print 'Slice: [%d]' % i
            _frame = self.rimage[slice(i,i+1)]
            _frame = (_frame - _mu) * _invnorm
            _frame.shape = _frame.shape[1:]


            g = N.gradient(_frame)
            dz, dy, dx = g[0], g[1], g[2]

            del(_frame) 

            Lzz += dz * dz
            Lzy += dz * dy
            Lzx += dz * dx
            Lyy += dy * dy
            Lyx += dy * dx
            Lxx += dx * dx

            del(dz); del(dy); del(dx); del(g) ; gc.collect()
            

        detlam = Lzz * (Lyy*Lxx - Lyx**2) - Lzy * (Lzy*Lxx - Lzx*Lyx) + Lzx * (Lzy*Lyx - Lzx*Lyy)

        test = N.greater(detlam, 0)
        resels = N.sqrt(detlam * test)
        fwhm = self.resel2fwhm(resels)

        fullslice = [slice(0, x) for x in self.grid.shape]
        self.resels[fullslice] = resels
        self.fwhm[fullslice] = fwhm

