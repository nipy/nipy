from neuroimaging.utils import wxmpl

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

# Create the PlotApp instance.
# The title string is one of several optional arguments.
app = wxmpl.PlotApp('Time Series Diagnostics',size=(10.0, 11.5))

### Create the data to plot ###
fmri_image = fMRIImage("test_fmri.img", datasource=repository)
tsdiag = TimeSeriesDiagnosticsStats(fmri_image)
tsdiag.compute()

### Plot it ###

fig = app.get_figure()

# Create the subplot Axes
axes1 = fig.add_subplot(4, 1, 1)
axes2 = fig.add_subplot(4, 1, 2)
axes3 = fig.add_subplot(4, 1, 3)
axes4 = fig.add_subplot(4, 1, 4)

axes1.plot(tsdiag.mse_time)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for j in range(tsdiag.mse_slice.shape[1]):
    axes2.plot(tsdiag.mse_slice[:,j], colors[j%7]+'.-')

axes3.plot(tsdiag.mean_signal)

axes4.plot(tsdiag.max_mse_slice)
axes4.plot(tsdiag.min_mse_slice)
axes4.plot(tsdiag.mean_mse_slice)

# Subplots must be labeled carefully, since labels
# can be accidentally hidden by other subplots
#axes1.set_title('Time Series Diagnostics')
axes1.set_xlabel('Difference image number')
axes1.set_ylabel('Scaled variance')

axes2.set_xlabel('Difference image number')
axes2.set_ylabel('Slice by slice variance')

axes3.set_xlabel('Image number')
axes3.set_ylabel('Scaled mean voxel intensity')

axes4.set_xlabel('Slice number')
axes4.set_ylabel('Max/mean/min slice variance')

# Let wxPython do its thing.
app.MainLoop()
