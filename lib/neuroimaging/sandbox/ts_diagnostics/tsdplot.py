#import pylab as P
#from neuroimaging.ui.visualization.viewer import BoxViewer
#        v = BoxViewer(self.mse_image)
#        v.draw(); P.show()

from wxPython.wx import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, \
     NavigationToolbar2WxAgg

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import TimeSeriesDiagnosticsStats

class TimeSeriesDiagnostics(object):

    def __init__(self, fmri_image, **keywords):
        self.tsdiag = TimeSeriesDiagnosticsStats(fmri_image)
        self.tsdiag.compute()

    def plotData(self):
        win = wxFrame(None, -1, "")
        fig = Figure((8,8), 75)
        canvas = FigureCanvasWxAgg(win, -1, fig)
        toolbar = NavigationToolbar2WxAgg(canvas)
        toolbar.Realize()
        sizer = wxBoxSizer(wxVERTICAL)
        sizer.Add(canvas, 1, wxLEFT|wxTOP|wxGROW)
        sizer.Add(toolbar, 0, wxLEFT|wxGROW)
        win.SetSizer(sizer)
        win.Fit()
        
        colors = ['b','g','r','c','m','y','k']
        ax = fig.add_subplot(411)
        ax.plot(self.tsdiag.MSEtime)
        ax = fig.add_subplot(412)
        for j in range(self.tsdiag.MSEslice.shape[1]):
            ax.plot(self.tsdiag.MSEslice[:,j], colors[j%7]+'.-')
        ax = fig.add_subplot(413)
        ax.plot(self.tsdiag.mean_signal)
        ax = fig.add_subplot(414)
        ax.plot(self.tsdiag.maxMSEslice)
        ax.plot(self.tsdiag.minMSEslice)
        ax.plot(self.tsdiag.meanMSEslice)
        win.Show()            

if __name__ == '__main__':
    app = wxPySimpleApp(0)
    sample = fMRIImage("test_fmri.hdr", datasource=repository)
    #sample = fMRIImage('http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/fiac0_fonc1.img')
    tsdiag = TimeSeriesDiagnostics(sample)
    tsdiag.plotData()
    app.MainLoop()
#    tsdiag.sd_image.tofile('diag_sd.img', clobber=True)
#    tsdiag.mean_image.tofile('diag_mean.img', clobber=True)
#    tsdiag.mse_image.tofile('diag_mse.img', clobber=True)

