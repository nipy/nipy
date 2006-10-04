from neuroimaging.core.image.image import Image
from neuroimaging.ui.visualization.viewer import BoxViewer
import numpy as N
import pylab

def overall(subject=0, run=1):
    return Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/fmristat_run/contrasts/overall/effect.img' % (subject,run))

overall_eff = [overall(subject=i) for i in range(4)]

def mean_overall():
    out = N.zeros(overall_eff[0].grid.shape, N.float64)
    for i in range(len(overall_eff)):
        out += overall_eff[i][:]
    out /= len(overall_eff)
    return Image(out, grid=overall_eff[0].grid)

overall_mean = mean_overall()
overall_mean.tofile('overall_mean.img', clobber=True)

v = BoxViewer(overall_mean, colormap='spectral'); v.draw()
pylab.show()
