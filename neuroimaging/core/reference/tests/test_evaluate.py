from neuroimaging.core.api import CoordinateSystem, CoordinateMap, Grid, Evaluator, Axis
import numpy as np
import nose.tools

def test_grid():
    input = CoordinateSystem('input', [Axis(n) for n in 'ij'])
    output = CoordinateSystem('output', [Axis(n) for n in 'xy'])
    def f(ij):
        i = ij[:,0]
        j = ij[:,1]
        return np.array([i**2+j,j**3+i]).T
    cmap = CoordinateMap(f, input, output)
    grid = Grid(cmap)
    eval = Evaluator.from_shape(cmap, (50,40))
    np.allclose(grid[0:50,0:40].values, eval.values)

def test_eval_slice():
    input = CoordinateSystem('input', [Axis(n) for n in 'ij'])
    output = CoordinateSystem('input', [Axis(n) for n in 'xy'])
    def f(ij):
        i = ij[:,0]
        j = ij[:,1]
        return np.array([i**2+j,j**3+i]).T

    cmap = CoordinateMap(f, input, output)

    cmap = CoordinateMap(f, input, output)
    grid = Grid(cmap)
    e = grid[0:50,0:40]
    ee = e[0:20:3]

    nose.tools.assert_equal(ee.shape, (7,40))
    nose.tools.assert_equal(ee.values.shape, (280,2))
    nose.tools.assert_equal(ee.transposed_values.shape, (2,7,40))

    ee = e[0:20:2,3]
    nose.tools.assert_equal(ee.values.shape, (10,2))
    nose.tools.assert_equal(ee.transposed_values.shape, (2,10))
    nose.tools.assert_equal(ee.shape, (10,))
