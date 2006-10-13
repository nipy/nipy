"""
Calculates statistics to aid in the diagnosis of problems in the time series.
"""

import numpy as N

from neuroimaging.core.image.image import Image

class TimeSeriesDiagnosticsStats(object):

    def __init__(self, fmri_image):
        self.mean = True
        self.sd = True
        self.mse = True
        self.mask = None

        self.fmri_image = fmri_image
        self.ntime = self.fmri_image.grid.shape[0]
        self.nslice = self.fmri_image.grid.shape[1]
 
        self.mse_time = N.zeros((self.ntime-1,), N.float64)
        self.mse_slice = N.zeros((self.ntime-1, self.nslice), N.float64)
        self.mean_signal = N.zeros((self.ntime,), N.float64)
        self.max_mse_slice = N.zeros((self.nslice,), N.float64)
        self.min_mse_slice = N.zeros((self.nslice,), N.float64)
        self.mean_mse_slice = N.zeros((self.nslice,), N.float64)

        self._npixel = {}

        self._compute()

    def _compute(self):

        if self.mean or self.sd or self.mse:
            grid = self.fmri_image.grid.subgrid(0)
            allslice = [slice(0,i,1) for i in self.fmri_image.grid.shape[1:]]
            if self.mean:
                self.mean_image = \
                  Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.sd:
                self.sd_image = \
                  Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.mse:
                self.mse_image = \
                  Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)

        if self.mask is not None:
            nvoxel = self.mask.readall().sum()
        else:
            nvoxel = N.product(self.fmri_image.grid.shape[1:])

        for i in range(self.ntime-1):
            tmp1 = N.squeeze(self.fmri_image[slice(i,i+1)])
            tmp = (tmp1 -
                   N.squeeze(self.fmri_image[slice(i+1,i+2)]))

            if self.mask is not None:
                tmp *= self.mask.readall()
                tmp1 *= self.mask.readall()

            tmp3 = N.power(tmp, 2)
            tmp2 = self.mse_image.readall()
            self.mse_image[allslice] = tmp2 + tmp3

            self.mse_time[i] = tmp3.sum() / nvoxel
            self.mean_signal[i] = tmp1.sum() / nvoxel

            for j in range(self.nslice):
                if self.mask is not None:
                    if not self._npixel.has_key(j):
                        self._npixel[j] = self.mask.getslice(slice(j,j+1)).sum()
                else:
                    if not self._npixel.has_key(j):
                        self._npixel[j] = N.product(self.fmri_image.grid.shape[2:])
                self.mse_slice[i,j] = N.power(tmp[j], 2).sum() / self._npixel[j]

        if self.mean:
            self.mean_image[allslice] = \
              N.sum(self.fmri_image.readall(), axis=0) / nvoxel
        if self.sd:
            if self.mask is not None:
                mask = self.mask.readall()
                self.sd_image[allslice] = \
                  N.std(mask * self.fmri_image.readall(), axis=0)
            else:
                self.sd_image[allslice] = \
                  N.std(self.fmri_image.readall(), axis=0)

        tmp = self.fmri_image[slice(i+1,i+2)]
        if self.mask is not None:
            tmp *= self.mask.readall()
        self.mean_signal[i+1] = tmp.sum() / nvoxel

        self.max_mse_slice = N.maximum.reduce(self.mse_slice, axis=0)
        self.min_mse_slice = N.minimum.reduce(self.mse_slice, axis=0)
        self.mean_mse_slice = N.mean(self.mse_slice, axis=0)

        self.mse_image[allslice] = \
          N.sqrt(self.mse_image.readall() / (self.ntime-1))
