import sys
from optparse import OptionParser, Option

from neuroimaging.data_io import DataSource
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils import wxmpl
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
        if len(args) != 1: self._error("Please provide a file name")
#        filename = "test_fmri.img"
        filename = args[0]
        if not DataSource().exists(filename):
            self._error("File not found: %s"%filename)
#        fmri_image = fMRIImage(filename, datasource=repository)
        fmri_image = fMRIImage(filename)
        self._tsdiagstats = TimeSeriesDiagnosticsStats(fmri_image)
        self._app = wxmpl.PlotApp('Time Series Diagnostics', size=(10.0, 11.5))
        self._plot_data()
        self._app.MainLoop()

if __name__ == '__main__':
    TimeSeriesDiagnostics().run() 
#    self.tsdiagstats.sd_image.tofile('diag_sd.img', clobber=True)
#    self.tsdiagstats.mean_image.tofile('diag_mean.img', clobber=True)
#    self.tsdiagstats.mse_image.tofile('diag_mse.img', clobber=True)
