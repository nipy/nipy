import numpy as np

from os.path import join as pjoin
from nipy.io.api import load_image
from nipy.core import api
import enthought.mayavi.mlab as ML

from fiac_example import datadir

mask = load_image(pjoin(datadir, 'group', 'mask.nii'))
avganat = load_image(pjoin(datadir, 'group', 'avganat.nii'))

def view_thresholdedT(design, contrast, threshold, inequality=np.greater):
    """
    A mayavi isosurface view of thresholded t-statistics

    Parameters
    ----------

    design: one of ['block', 'event']

    contrast: str
    
    threshold: float

    inequality: one of [np.greater, np.less]

    """

    maska = np.asarray(mask)
    tmap = np.array(load_image(pjoin(datadir, 'group', design, contrast, 't.nii')))
    test = inequality(tmap, threshold)
    tval = np.zeros(tmap.shape)
    tval[test] = tmap[test]
    tval[~test]

    # XXX make the array axes agree with mayavi2

    avganata = np.array(avganat)
    avganat_iso = ML.contour3d(avganata * maska, opacity=0.3, contours=[3600], color=(0.8,0.8,0.8))

    avganat_iso.actor.property.backface_culling = True
    avganat_iso.actor.property.ambient = 0.3

    tval_iso = ML.contour3d(tval * mask, color=(0.8,0.3,0.3), contours=[threshold])
    return avganat_iso, tval_iso
