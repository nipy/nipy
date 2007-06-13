from optparse import Option

from wxPython.wx import wxPySimpleApp, wxFrame, wxBoxSizer, wxVERTICAL, \
  wxLEFT, wxTOP, wxGROW
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, \
  NavigationToolbar2WxAgg

from neuroimaging.data_io.api import DataSource
from neuroimaging.modalities.fmri.api import FmriImage
from neuroimaging.ui.tools import BaseTool
from neuroimaging.utils import wxmpl
#from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

class TimeSeriesDiagnostics(BaseTool):
    """
    Command-line tool for displaying four plots useful for diagnosing
    potential problems in the time series.

    The top plot displays the scaled variance from image to image;
    the second plot shows the scaled variance per slice;
    the third plot shows the scaled mean voxel intensity for each image;
    while, the bottom one plots the maximum, mean, and minimum scaled
    slice variances per slice.
    """

    _options = (
      Option('-w', '--wxmpl', action="store_true", dest="wxmpl",
        help="use wxmpl for plotting"),
      Option('-f', '--file', dest='file',
        help="input file to read data from"))

    def _plot_data(self):
        """
        Plot data using mpl.
        """
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

    def _wxmpl_plot_data(self):
        """
        Plot data using wxmpl.
        """
        fig = self._app.get_figure()

        # Create the subplot Axes
        axes_1 = fig.add_subplot(4, 1, 1)
        axes_2 = fig.add_subplot(4, 1, 2)
        axes_3 = fig.add_subplot(4, 1, 3)
        axes_4 = fig.add_subplot(4, 1, 4)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        axes_1.plot(self._tsdiagstats.mse_time)
        for j in range(self._tsdiagstats.mse_slice.shape[1]):
            axes_2.plot(self._tsdiagstats.mse_slice[:,j], colors[j%7]+'.-')
        axes_3.plot(self._tsdiagstats.mean_signal)
        axes_4.plot(self._tsdiagstats.max_mse_slice)
        axes_4.plot(self._tsdiagstats.min_mse_slice)
        axes_4.plot(self._tsdiagstats.mean_mse_slice)

        # Subplots must be labeled carefully, since labels
        # can be accidentally hidden by other subplots
        #axes_1.set_title('Time Series Diagnostics')
        axes_1.set_xlabel('Difference image number')
        axes_1.set_ylabel('Scaled variance')
        axes_2.set_xlabel('Difference image number')
        axes_2.set_ylabel('Slice by slice variance')
        axes_3.set_xlabel('Image number')
        axes_3.set_ylabel('Scaled mean voxel intensity')
        axes_4.set_xlabel('Slice number')
        axes_4.set_ylabel('Max/mean/min slice variance')

    def run(self):
        options, args = self.parse_args()
        if options.file is None:
            self._error("Please provide a file name.")
        if not DataSource().exists(options.file):
            self._error("File not found: %s"%options.file)
#        fmri_image = FmriImage(file, datasource=repository)
        fmri_image = FmriImage(options.file)
        self._tsdiagstats = TimeSeriesDiagnosticsStats(fmri_image)
        if not options.wxmpl:
            self._app = wxPySimpleApp(0)
            self._plot_data()
        else:
            self._app = wxmpl.PlotApp('Time Series Diagnostics', \
                                      size=(10.0, 11.5))
            self._wxmpl_plot_data()
        self._app.MainLoop()

if __name__ == '__main__':
    TimeSeriesDiagnostics().run() 
#    TS_DIAG.sd_image.tofile('diag_sd.img', clobber=True)
#    TS_DIAG.mean_image.tofile('diag_mean.img', clobber=True)
#    TS_DIAG.mse_image.tofile('diag_mse.img', clobber=True)
