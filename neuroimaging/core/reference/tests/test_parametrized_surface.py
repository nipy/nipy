"""
Polar coordinate systems using a CoordinateMap
"""

import numpy as np
import nose.tools

from neuroimaging.core.api import CoordinateMap, CoordinateSystem, Axis, Affine
from neuroimaging.core.api import Grid

uv = CoordinateSystem('input', [Axis(l) for l in ['u', 'v']])
xyz = CoordinateSystem('output', [Axis(l) for l in 'xyz'])

def parametric(vals):

    """
    Parametrization of the surface x**2-y**2*z**2+z**3=0
    """
    vals = uv.typecast(vals, uv.dtype)
    u = vals['u']
    v = vals['v']
    
    o = np.array([v*(u**2-v**2),
                  u,
                  u**2-v**2]).T
    return o

"""
Let's check that indeed this is a parametrization of that surface
"""

def implicit(vals):
    vals = xyz.typecast(vals, xyz.dtype)
    x = vals['x']; y = vals['y']; z = vals['z']
    return x**2-y**2*z**2+z**3

surface_param = CoordinateMap(parametric, uv, xyz)
assert np.allclose(implicit(parametric(np.random.standard_normal((40,2)))), 0)

def test_grid():
    g = Grid(surface_param)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    nose.tools.assert_equal(x.shape, (201,101))
    nose.tools.assert_equal(y.shape, (201,101))
    nose.tools.assert_equal(z.shape, (201,101))
    return x, y, z

if __name__ == '__main__':
    from enthought.mayavi import mlab
    mlab.mesh(*test_grid())
    mlab.draw()
