import numpy as np

import nose.tools
from neuroimaging.core.api import CoordinateSystem, Coordinate
from neuroimaging.core.reference.coordinate_map import CoordinateMap, Affine
from neuroimaging.core.reference.coordinate_map import reorder_input, reorder_output

def setup_cmap():
    step = [1,2,3]
    output_axes = [Coordinate(s) for i, s in enumerate('xyz')]
    input_axes = [Coordinate(s) for i, s in enumerate('ijk')]
    input_coords = CoordinateSystem('input', input_axes)
    output_coords = CoordinateSystem('output', output_axes)
    cmap = Affine(np.diag([1,2,3,1]), input_coords, output_coords)
    return cmap

def test_reorder1():
    cmap = setup_cmap()
    cmap2 = reorder_input(cmap)
    nose.tools.assert_equal(cmap2.input_coords.name, 'input-reordered')
    nose.tools.assert_equal(cmap2.output_coords.name, 'output')

def test_reorder2():

    cmap = setup_cmap()
    cmap2 = reorder_output(cmap)
    nose.tools.assert_equal(cmap2.input_coords.name, 'input')
    nose.tools.assert_equal(cmap2.output_coords.name, 'output-reordered')
