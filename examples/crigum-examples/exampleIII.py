"""
A mayavi example showing 3 different isosurfaces


"""

from enthought.mayavi.scripts import mayavi2
mayavi2.standalone(globals())

# Standard library imports
from os.path import join, dirname
import gc

# Enthought library imports
import enthought.mayavi
from enthought.mayavi.modules.iso_surface import IsoSurface
from enthought.mayavi.sources.array_source import ArraySource
from enthought.mayavi.modules.outline import Outline
from enthought.mayavi.modules.vectors import Vectors

import numpy as N

def iso(img, contour, color=(1.,1.,1.), sample=None, spacing=[1,1,1],
        opacity=1):
    src = ArraySource(transpose_input_array=False, spacing=spacing)
    mayavi.add_source(src)
    if sample is not None:
        img = img[sample]
    src.scalar_data = N.nan_to_num(img[:])
    i = IsoSurface(compute_normals=True)
    mayavi.add_module(i)
    i.actor.property.color = color
    i.actor.mapper.scalar_visibility = False
    i.contour.contours[0] = contour
    i.actor.property.opacity = 0.5
    return i

if __name__ == "__main__":
    mayavi.new_scene()
    from fiac import anat, sentence, average, avg152
    iso(avg152, 230000.)
    iso(sentence, 4, color=(1,0,0))
    iso(average, 4, color=(0,0,1), opacity=0.5)
    o = Outline()
    mayavi.add_module(o)


