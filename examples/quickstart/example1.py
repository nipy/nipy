import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.utils.tests.data import repository

def overall(subject=0, run=1):
    return Image('FIAC/fiac%d/fonc%d/fsl/fmristat_run/contrasts/overall/effect.img' % (subject,run), repository)

OVERALL_EFF = [overall(subject=i) for i in range(4)]

def mean_overall():
    out = N.zeros(OVERALL_EFF[0].grid.shape, N.float64)
    for i in range(len(OVERALL_EFF)):
        out += OVERALL_EFF[i][:]
    out /= len(OVERALL_EFF)
    return Image(out, grid=OVERALL_EFF[0].grid)

OVERALL_MEAN = mean_overall()
OVERALL_MEAN.tofile('OVERALL_MEAN.img', clobber=True)

VIEWER = BoxViewer(OVERALL_MEAN, colormap='spectral')
VIEWER.draw()
pylab.show()
