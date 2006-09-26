import numpy as N

from neuroimaging import traits

from neuroimaging.core.image.image import Image

class TimeSeriesDiagnosticsStats(traits.HasTraits):
    mean = traits.true
    sd = traits.true
    mse = traits.true
    mask = traits.Any

    def __init__(self, fmri_image, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.fmri_image = fmri_image

    def compute(self):
        ntime = self.fmri_image.grid.shape[0]
        nslice = self.fmri_image.grid.shape[1]
                                                
        self.MSEtime = N.zeros((ntime-1,), N.float64)
        self.MSEslice = N.zeros((ntime-1, nslice), N.float64)
        self.mean_signal = N.zeros((ntime,), N.float64)
        self.maxMSEslice = N.zeros((nslice,), N.float64)
        self.minMSEslice = N.zeros((nslice,), N.float64)
        self.meanMSEslice = N.zeros((nslice,), N.float64)

        if self.mean or self.sd or self.mse:
            grid = self.fmri_image.grid.subgrid(0)
            allslice = [slice(0,i,1) for i in self.fmri_image.grid.shape[1:]]
            if self.mean:
                self.mean_image = Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.sd:
                self.sd_image = Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.mse:
                self.mse_image = Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)

        npixel = {}
        if self.mask is not None:
            nvoxel = self.mask.readall().sum()
        else:
            nvoxel = N.product(self.fmri_image.grid.shape[1:])

        for i in range(ntime-1):
            tmp1 = N.squeeze(self.fmri_image[slice(i,i+1)])
            tmp = (tmp1 -
                   N.squeeze(self.fmri_image[slice(i+1,i+2)]))

            if self.mask is not None:
                tmp *= self.mask.readall()
                tmp1 *= self.mask.readall()
            
            tmp3 = N.power(tmp, 2)
            tmp2 = self.mse_image.readall()
            self.mse_image[allslice] = tmp2 + tmp3

            self.MSEtime[i] = tmp3.sum() / nvoxel
            self.mean_signal[i] = tmp1.sum() / nvoxel

            for j in range(nslice):
                if self.mask is not None:
                    if not npixel.has_key(j):
                        npixel[j] = self.mask.getslice(slice(j,j+1)).sum()
                else:
                    if not npixel.has_key(j):
                        npixel[j] = N.product(self.fmri_image.grid.shape[2:])
                self.MSEslice[i,j] = N.power(tmp[j], 2).sum() / npixel[j]

        if self.mean:
            self.mean_image[allslice] = N.sum(self.fmri_image.readall(), axis=0) / nvoxel
        if self.sd:
            if self.mask is not None:
                mask = self.mask.readall()
                self.sd_image[allslice] = N.std(mask * self.fmri_image.readall(), axis=0)
            else:
                self.sd_image[allslice] =  N.std(self.fmri_image.readall(), axis=0)

        tmp = self.fmri_image[slice(i+1,i+2)]
        if self.mask is not None:
            tmp *= self.mask.readall()
        self.mean_signal[i+1] = tmp.sum() / nvoxel
        
        self.maxMSEslice = N.maximum.reduce(self.MSEslice, axis=0)
        self.minMSEslice = N.minimum.reduce(self.MSEslice, axis=0)
        self.meanMSEslice = N.mean(self.MSEslice, axis=0)
        
        self.mse_image[allslice] = N.sqrt(self.mse_image.readall()/ (ntime-1))

