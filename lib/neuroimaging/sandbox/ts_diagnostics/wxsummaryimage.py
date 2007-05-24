import numpy as N
import matplotlib.cm as cm

from neuroimaging.modalities.fmri.api import fMRIImage
from neuroimaging.utils import wxmpl
from neuroimaging.utils.tests.data import repository

from neuroimaging.sandbox.ts_diagnostics.tsdstats import \
  TimeSeriesDiagnosticsStats

def main():
    fmri_image = fMRIImage("test_fmri.img", datasource=repository)
    ts_diag = TimeSeriesDiagnosticsStats(fmri_image)
    image = N.nan_to_num(ts_diag.mse_image.readall())

    app = wxmpl.PlotApp('Time Series Diagnostics', size=(10.0, 11.5))
    fig = app.get_figure()
    axes_1 = fig.add_subplot(1, 1, 1)
    axes_1.imshow(image[5,:,:], cmap=cm.bone)
    app.MainLoop()

if __name__ == '__main__':
    main()
