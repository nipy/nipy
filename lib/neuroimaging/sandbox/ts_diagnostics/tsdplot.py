import sys
from optparse import OptionParser, Option

from wxPython.wx import wxPySimpleApp, wxFrame, wxBoxSizer, wxVERTICAL, \
  wxLEFT, wxTOP, wxGROW
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, \
  NavigationToolbar2WxAgg

from neuroimaging.data_io import DataSource
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

class TimeSeriesDiagnostics(OptionParser):
    "Command-line tool for getting and setting Analyze header values."
	   
    _usage= "%prog [options] <hdrfile>\n"+__doc__
#    options = (
#      Option('-a', '--attribute', dest="attname",
#        help="Get or set this attribute"),
#      Option('-v', '--value', dest="value",
#        help="Set attribute to this value"))

    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage(self._usage)
#        self.add_options(self.options)

    def _error(self, message):
        print message
        self.print_help()
        sys.exit(0)

    def _plot_data(self):
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

        axes_1 = fig.add_subplot(411)
        axes_2 = fig.add_subplot(412)
        axes_3 = fig.add_subplot(413)
        axes_4 = fig.add_subplot(414)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        axes_1.plot(self._tsdiagstats.mse_time)
        for j in range(self._tsdiagstats.mse_slice.shape[1]):
            axes_2.plot(self._tsdiagstats.mse_slice[:,j], colors[j%7]+'.-')
        axes_3.plot(self._tsdiagstats.mean_signal)
        axes_4.plot(self._tsdiagstats.max_mse_slice)
        axes_4.plot(self._tsdiagstats.min_mse_slice)
        axes_4.plot(self._tsdiagstats.mean_mse_slice)
        win.Show()

    def run(self):
        options, args = self.parse_args()
        if len(args) != 1: self._error("Please provide a file name")
#        filename = "test_fmri.img"
        filename = args[0]
        if not DataSource().exists(filename):
            self._error("File not found: %s"%filename)
#        fmri_image = fMRIImage(filename, datasource=repository)
        fmri_image = fMRIImage(filename)
        self._tsdiagstats = TimeSeriesDiagnosticsStats(fmri_image)
        app = wxPySimpleApp(0)
        self._plot_data()
        app.MainLoop()

if __name__ == '__main__':
   TimeSeriesDiagnostics().run() 
#    TS_DIAG.sd_image.tofile('diag_sd.img', clobber=True)
#    TS_DIAG.mean_image.tofile('diag_mean.img', clobber=True)
#    TS_DIAG.mse_image.tofile('diag_mse.img', clobber=True)
