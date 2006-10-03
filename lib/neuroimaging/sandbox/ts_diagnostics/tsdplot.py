#import pylab as P
#from neuroimaging.ui.visualization.viewer import BoxViewer
#        v = BoxViewer(self.mse_image)
#        v.draw(); P.show()

from wxPython.wx import wxPySimpleApp, wxFrame, wxBoxSizer, wxVERTICAL, \
  wxLEFT, wxTOP, wxGROW
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, \
  NavigationToolbar2WxAgg

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

class TimeSeriesDiagnostics(object):

    def __init__(self, fmri_image):
        self.tsdiag = TimeSeriesDiagnosticsStats(fmri_image)
        self.tsdiag.compute()

    def plot_data(self):
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
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        ax = fig.add_subplot(411)
        ax.plot(self.tsdiag.mse_time)
        ax = fig.add_subplot(412)
        for j in range(self.tsdiag.mse_slice.shape[1]):
            ax.plot(self.tsdiag.mse_slice[:,j], colors[j%7]+'.-')
        ax = fig.add_subplot(413)
        ax.plot(self.tsdiag.mean_signal)
        ax = fig.add_subplot(414)
        ax.plot(self.tsdiag.max_mse_slice)
        ax.plot(self.tsdiag.min_mse_slice)
        ax.plot(self.tsdiag.mean_mse_slice)
        win.Show()            

if __name__ == '__main__':
    app = wxPySimpleApp(0)
    sample = fMRIImage("test_fmri.img", datasource=repository)
    tsdiag = TimeSeriesDiagnostics(sample)
    tsdiag.plot_data()
    app.MainLoop()
#    tsdiag.sd_image.tofile('diag_sd.img', clobber=True)
#    tsdiag.mean_image.tofile('diag_mean.img', clobber=True)
#    tsdiag.mse_image.tofile('diag_mse.img', clobber=True)

