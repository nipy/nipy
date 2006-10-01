from neuroimaging.utils import wxmpl

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

# Create the PlotApp instance.
# The title string is one of several optional arguments.
APP = wxmpl.PlotApp('Time Series Diagnostics', size=(10.0, 11.5))

### Create the data to plot ###
FMRI_IMAGE = fMRIImage("test_fmri.img", datasource=repository)
TS_DIAG = TimeSeriesDiagnosticsStats(FMRI_IMAGE)

### Plot it ###
FIG = APP.get_figure()

# Create the subplot Axes
AXES_1 = FIG.add_subplot(4, 1, 1)
AXES_2 = FIG.add_subplot(4, 1, 2)
AXES_3 = FIG.add_subplot(4, 1, 3)
AXES_4 = FIG.add_subplot(4, 1, 4)

AXES_1.plot(TS_DIAG.mse_time)

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for j in range(TS_DIAG.mse_slice.shape[1]):
    AXES_2.plot(TS_DIAG.mse_slice[:,j], COLORS[j%7]+'.-')

AXES_3.plot(TS_DIAG.mean_signal)

AXES_4.plot(TS_DIAG.max_mse_slice)
AXES_4.plot(TS_DIAG.min_mse_slice)
AXES_4.plot(TS_DIAG.mean_mse_slice)

# Subplots must be labeled carefully, since labels
# can be accidentally hidden by other subplots
#AXES_1.set_title('Time Series Diagnostics')
AXES_1.set_xlabel('Difference image number')
AXES_1.set_ylabel('Scaled variance')

AXES_2.set_xlabel('Difference image number')
AXES_2.set_ylabel('Slice by slice variance')

AXES_3.set_xlabel('Image number')
AXES_3.set_ylabel('Scaled mean voxel intensity')

AXES_4.set_xlabel('Slice number')
AXES_4.set_ylabel('Max/mean/min slice variance')

# Let wxPython do its thing.
APP.MainLoop()
