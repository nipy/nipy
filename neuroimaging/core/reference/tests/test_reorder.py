import numpy as np

import nose.tools
from neuroimaging.core.api import Axis, CoordinateSystem
from neuroimaging.core.reference.coordinate_map import CoordinateMap
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.coordinate_map import reorder_input, reorder_output

def setup_cmap():
    shape = (3,4,5)
    step = [1,2,3]
    output_axes = [Axis(s) for i, s in enumerate('xyz')]
    input_axes = [Axis(s, length=shape[i]) for i, s in enumerate('ijk')]
    input_coords = CoordinateSystem('input', input_axes)
    output_coords = CoordinateSystem('output', output_axes)
    cmap = CoordinateMap(Affine(np.diag([1,2,3,1])), input_coords, output_coords)
    return cmap

def test_reorder1():
    cmap = setup_cmap()
    cmap2 = reorder_input(cmap)
    nose.tools.assert_true(hasattr(cmap2, 'shape'))
    nose.tools.assert_equal(cmap2.input_coords.name, 'input-reordered')
    nose.tools.assert_equal(cmap2.output_coords.name, 'output')
    nose.tools.assert_equal(cmap2.shape, cmap.shape[::-1])

def test_reorder2():

    cmap = setup_cmap()
    cmap2 = reorder_output(cmap)
    nose.tools.assert_true(hasattr(cmap2, 'shape'))
    nose.tools.assert_equal(cmap2.input_coords.name, 'input')
    nose.tools.assert_equal(cmap2.output_coords.name, 'output-reordered')
    nose.tools.assert_equal(cmap2.shape, cmap.shape)
