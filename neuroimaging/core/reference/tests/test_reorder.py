import numpy as np

from neuroimaging.core.api import VoxelAxis, RegularAxis, VoxelCoordinateSystem, CoordinateSystem, StartStepCoordinateSystem
from neuroimaging.core.reference.coordinate_map import CoordinateMap
from neuroimaging.core.reference.mapping import Affine

def setup_cmap():
    shape = (3,4,5)
    output_axes = [RegularAxis(s, step=i+1) for i, s in enumerate('xyz')]
    input_axes = [VoxelAxis(s, length=shape[i]) for i, s in enumerate('ijk')]
    input_coords = VoxelCoordinateSystem('input', input_axes)
    output_coords = StartStepCoordinateSystem('output', output_axes)
    cmap = CoordinateMap(Affine(output_coords.affine), input_coords,
                         output_coords)
    return cmap

def test_reorder1():
    cmap = setup_cmap()
    cmap2 = cmap.reorder_input()
    assert hasattr(cmap2, 'shape')
    assert cmap2.input_coords.name == 'input-reordered'
    assert cmap2.output_coords.name == 'output'
    assert cmap2.shape == cmap.shape[::-1]

def test_reorder2():

    cmap = setup_cmap()
    cmap2 = cmap.reorder_output()
    assert hasattr(cmap2, 'shape')
    assert cmap2.input_coords.name == 'input'
    assert cmap2.output_coords.name == 'output-reordered'
    assert cmap2.shape == cmap.shape
