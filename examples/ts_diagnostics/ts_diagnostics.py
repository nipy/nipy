import neuroimaging
import enthought.traits as traits
import numpy as N

class TSDiagnostics(traits.HasTraits):

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

        self.MSEtime = N.zeros((ntime-1,), N.Float)
        self.MSEslice = N.zeros((ntime-1, nslice), N.Float)
        self.mean_signal = N.zeros((ntime,), N.Float)
        self.maxMSEslice = N.zeros((nslice,), N.Float)
        self.minMSEslice = N.zeros((nslice,), N.Float)
        self.meanMSEslice = N.zeros((nslice,), N.Float)

        if self.mean or self.sd or self.mse:
            grid = self.fmri_image.grid.subgrid(0)
            allslice = [slice(0,i,1) for i in self.fmri_image.grid.shape[1:]]
            if self.mean:
                self.mean_image = neuroimaging.image.Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.sd:
                self.sd_image = neuroimaging.image.Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)
            if self.mse:
                self.mse_image = neuroimaging.image.Image(N.zeros(self.fmri_image.shape[1:]), grid=grid)

        npixel = {}
        if self.mask is not None:
            nvoxel = self.mask.readall().sum()
        else:
            nvoxel = N.product(self.fmri_image.grid.shape[1:])

        for i in range(ntime-1):
            tmp1 = N.squeeze(self.fmri_image.getslice(slice(i,i+1)))
            tmp = (tmp1 -
                   N.squeeze(self.fmri_image.getslice(slice(i+1,i+2))))

            if self.mask is not None:
                tmp *= self.mask.readall()
                tmp1 *= self.mask.readall()
            
            tmp3 = N.power(tmp, 2)
            tmp2 = self.mse_image.readall()
            self.mse_image.writeslice(allslice, tmp2 + tmp3)

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
            self.mean_image.writeslice(allslice, N.sum(self.fmri_image.readall(), axis=0) / nvoxel)
        if self.sd:
            if self.mask is not None:
                mask = self.mask.readall()
                self.sd_image.writeslice(allslice, N.std(mask * self.fmri_image.readall(), axis=0))
            else:
                self.sd_image.writeslice(allslice, N.std(self.fmri_image.readall(), axis=0))

        tmp = self.fmri_image.getslice(slice(i+1,i+2))
        if self.mask is not None:
            tmp *= self.mask.readall()
        self.mean_signal[i+1] = tmp.sum() / nvoxel

        self.maxMSEslice = N.maximum.reduce(self.MSEslice, axis=0)
        self.minMSEslice = N.minimum.reduce(self.MSEslice, axis=0)
        self.meanMSEslice = N.mean(self.MSEslice, axis=0)


sample = neuroimaging.fmri.fMRIImage('http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/filtered_func_data.img')

tsdiag = TSDiagnostics(sample)
tsdiag.compute()

        
