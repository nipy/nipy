from neuroimaging import traits
import numpy as N
import pylab as P
from matplotlib.axes import Subplot
from matplotlib.figure import Figure

from neuroimaging.core.image.image import Image
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.utils.tests.data import repository

#matplotlib.use('WXAgg')
###########
# to run plot window without halting execution
# should add check to make sure wxPython is available,
# if not maybe import corresponding classes from gtk backends?

## import gtk
## from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
## from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar

from wxPython.wx import *

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, \
     NavigationToolbar2WxAgg, FigureManager

###########


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
        v = BoxViewer(self.mse_image)
        v.draw(); P.show()

    def plotData(self):
        win = wxFrame(None, -1, "")
        fig = Figure((8,8), 75)
        canvas = FigureCanvasWxAgg(win, -1, fig)
        toolbar = NavigationToolbar2WxAgg(canvas)
        toolbar.Realize()
        #figmgr = FigureManager(canvas, 1, win)
        sizer = wxBoxSizer(wxVERTICAL)
        sizer.Add(canvas, 1, wxLEFT|wxTOP|wxGROW)
        sizer.Add(toolbar, 0, wxLEFT|wxGROW)
        win.SetSizer(sizer)
        win.Fit()
        
        colors = ['b','g','r','c','m','y','k']
        ax = fig.add_subplot(411)
        ax.plot(self.MSEtime)
        ax = fig.add_subplot(412)
        for j in range(self.MSEslice.shape[1]):
            ax.plot(self.MSEslice[:,j], colors[j%7]+'.-')
        ax = fig.add_subplot(413)
        ax.plot(self.mean_signal)
        ax = fig.add_subplot(414)
        ax.plot(self.maxMSEslice)
        ax.plot(self.minMSEslice)
        ax.plot(self.meanMSEslice)
        win.Show()            

if __name__ == '__main__':
    app = wxPySimpleApp(0)
    sample = fMRIImage("test_fmri.hdr", datasource=repository)
    #sample = fMRIImage('http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/fiac0_fonc1.img')
    
    tsdiag = TSDiagnostics(sample)
    tsdiag.compute()
    tsdiag.plotData()
    app.MainLoop()
    tsdiag.sd_image.tofile('diag_sd.img', clobber=True)
    tsdiag.mean_image.tofile('diag_mean.img', clobber=True)
    tsdiag.mse_image.tofile('diag_mse.img', clobber=True)

