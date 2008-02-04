"""
Another mayavi example: "hedgehog" style
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

from scipy import mgrid

import sys
sys.path.append("./trauma-study/python")

from hotelling import effect, T2

T2 = N.clip(T2[:], 0, 10000)
T2.shape = T2.shape[::-1]

effect = effect[:]
effect.shape = (3,) + effect.shape[1:][::-1]

mayavi.new_scene()

effect_src = ArraySource(transpose_input_array=False, spacing=[3,2,2])
effect_src.vector_data = N.transpose(effect[:,::3,::2,::2], (1,2,3,0))
mayavi.add_source(effect_src)

# An isosurface module.

v = Vectors()
mayavi.add_module(v)

T2_src = ArraySource(transpose_input_array=False)
T2_src.scalar_data = T2
mayavi.add_source(T2_src)
iso = IsoSurface(compute_normals=True)
mayavi.add_module(iso)
iso.actor.property.color = (0.9,0.2,0.2)
iso.actor.mapper.scalar_visibility = False

o = Outline()
mayavi.add_module(o)

import struct

wm = N.array(struct.unpack("<%df" % (181*217*181,), file("./trauma-study/python/CNT_AVG_wm.bin").read()))
wm.shape = (181,217,181)
wm = wm[::2,::2,::2]

wm_src = ArraySource(transpose_input_array=False)
wm_src.scalar_data = wm
mayavi.add_source(wm_src)
wm_iso = IsoSurface(compute_normals=True)
wm_iso.actor.mapper.scalar_visibility = False
wm_iso.actor.property.opacity = 0.5
mayavi.add_module(wm_iso)

wm_iso.contour.contours[0] = 0.05
#wm_iso.contour.contours.append(0.5)

