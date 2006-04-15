import neuroimaging as ni
import numpy as N
import pylab

def overall(subject=0, run=1):
    return ni.image.Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/fmristat_run/contrasts/overall/effect.img' % (subject,run))

overall_eff = [overall(subject=i) for i in range(4)]

def mean_overall():
    out = N.zeros(overall_eff[0].grid.shape, N.Float)
    for i in range(len(overall_eff)):
        out += overall_eff[i].readall()
    out /= len(overall_eff)
    return ni.image.Image(out, grid=overall_eff[0].grid)

overall_mean = mean_overall()
overall_mean.tofile('overall_mean.img', clobber=True)

v = ni.visualization.viewer.BoxViewer(overall_mean); v.draw()
pylab.show()
