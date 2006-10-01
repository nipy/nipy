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

        axes_1 = fig.add_subplot(411)
        axes_1.plot(self.tsdiag.mse_time)
        axes_2 = fig.add_subplot(412)
        for j in range(self.tsdiag.mse_slice.shape[1]):
            axes_2.plot(self.tsdiag.mse_slice[:,j], colors[j%7]+'.-')
        axes_3 = fig.add_subplot(413)
        axes_3.plot(self.tsdiag.mean_signal)
        axes_4 = fig.add_subplot(414)
        axes_4.plot(self.tsdiag.max_mse_slice)
        axes_4.plot(self.tsdiag.min_mse_slice)
        axes_4.plot(self.tsdiag.mean_mse_slice)
        win.Show()

if __name__ == '__main__':
    APP = wxPySimpleApp(0)
    SAMPLE = fMRIImage("test_fmri.img", datasource=repository)
    TS_DIAG = TimeSeriesDiagnostics(SAMPLE)
    TS_DIAG.plot_data()
    APP.MainLoop()
#    TS_DIAG.sd_image.tofile('diag_sd.img', clobber=True)
#    TS_DIAG.mean_image.tofile('diag_mean.img', clobber=True)
#    TS_DIAG.mse_image.tofile('diag_mse.img', clobber=True)
