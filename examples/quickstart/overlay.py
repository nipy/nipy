"""
This short script shows how to overlay a functional image onto
an anatomical image.

In this example, the images have the same mapping instance, but this
need not be the case: the important thing is that the output coordinates
of standard.grid.mapping are the same as T.grid.mapping.

The output coordinates are sampled by CoronalPlot, which need not be a
hyperplane of any dataset.

"""

import numpy as N
import pylab, matplotlib

from neuroimaging.core.api import Image
from neuroimaging.ui.visualization.slices import CoronalPlot
from neuroimaging.core.reference.mapping import Affine

# This is the correct 4x4 transform for images
#
# The 4x4 in the file are FSL coordinates, i.e. with origin 0

mapping = Affine(N.array([[   2.,    0.,    0.,  -72.],
                          [   0.,    2.,    0., -126.],
                          [   0.,    0.,    2.,  -90.],
                          [   0.,    0.,    0.,    1.]]))
                         
standard = Image("http://kff.stanford.edu/FIAC/avg152T1_brain.img")
standard.grid.mapping = mapping

T = Image("http://kff.stanford.edu/FIAC/multi/block/contrasts/average/t.nii")
T.grid.mapping = mapping

def overlay(threshold=4, y=-14):
    """
    Display T image over atlas standard, thresholded at threshold.
    """
    slices = {'T':CoronalPlot(T, y=y, colormap='spectral'),
              'standard':CoronalPlot(standard, y=y, colormap='gray')
              }

    # set T min and max for colormap

    slices['T'].vmin = -10; slices['T'].vmax = 10

    RGBA = {'T':slices['T'].RGBA(),
            'standard':slices['standard'].RGBA()}

    mask = N.squeeze(N.greater(N.fabs(slices['T'].numdata()), threshold))
    mask = N.multiply.outer(mask, N.ones(4))

    outRGBA = mask * RGBA['T'] + (1. - mask) * RGBA['standard']

    del(mask); del(RGBA); del(slices)
    return outRGBA

def colorbar(lower, upper, cmap=pylab.cm.spectral, shrink=0.4, fraction=0.4):
    """
    Colorbar for a given cmap
    """
    
    x = N.linspace(lower, upper, 91**2)
    x.shape = (91,91)
    im = pylab.imshow(x)
    pylab.colorbar(im, shrink=shrink, fraction=fraction)

if __name__ == '__main__':

    RGBA = overlay()
    colorbar(-10,10)

    im = pylab.imshow(RGBA, origin='lower')
    a = pylab.gca()
    a.set_xticklabels([]); a.set_yticklabels([])

    pylab.draw()
    pylab.savefig("overlay.png")
    pylab.savefig("/home/jtaylo/public_html/test.png")
    

