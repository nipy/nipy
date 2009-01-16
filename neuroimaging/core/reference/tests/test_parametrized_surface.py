"""
Parametrized surfaces using a CoordinateMap
"""
import numpy as np
import nose.tools

from neuroimaging.core.api import CoordinateMap, CoordinateSystem, Affine, \
    Coordinate
from neuroimaging.core.api import Grid

uv = CoordinateSystem('input', [Coordinate(l, dtype=np.float) for l in ['u', 'v']])
xyz = CoordinateSystem('output', [Coordinate(l, dtype=np.float) for l in 'xyz'])

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

def test_grid32():
    """
    Check that we can use a float32 input and output
    """

    uv32 = CoordinateSystem('input', [Coordinate(l, dtype=np.float32) for l in ['u', 'v']])
    xyz32 = CoordinateSystem('output', [Coordinate(l, dtype=np.float32) for l in 'xyz'])

    def par32(vals):
        """
        float32 version of the same thing
        """
        vals = uv32.typecast(vals, uv32.dtype)
        u = vals['u']
        v = vals['v']
    
        o = np.array([v*(u**2-v**2),
                      u,
                      u**2-v**2]).T
        return o
    surface32 = CoordinateMap(par32, uv32, xyz32)
    g = Grid(surface32)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    nose.tools.assert_equal(x.shape, (201,101))
    nose.tools.assert_equal(y.shape, (201,101))
    nose.tools.assert_equal(z.shape, (201,101))
    nose.tools.assert_equal(x.dtype, np.dtype(np.float32))
    return x, y, z

def test_grid32_c128():
    """
    Check that we can use a float32 input and complex128 output
    """

    uv32 = CoordinateSystem('input', [Coordinate(l, dtype=np.float32) for l in ['u', 'v']])
    xyz128 = CoordinateSystem('output', [Coordinate(l, dtype=np.complex128) for l in 'xyz'])

    def par32_c128(vals):
        """
        float32 version of the same thing
        """
        vals = uv32.typecast(vals, uv32.dtype)
        u = vals['u']
        v = vals['v']
    
        o = np.array([v*(u**2-v**2),
                      u,
                      u**2-v**2]).T
        return o.astype(np.complex128)

    surface = CoordinateMap(par32_c128, uv32, xyz128)
    g = Grid(surface)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    nose.tools.assert_equal(x.shape, (201,101))
    nose.tools.assert_equal(y.shape, (201,101))
    nose.tools.assert_equal(z.shape, (201,101))
    nose.tools.assert_equal(x.dtype, np.dtype(np.complex128))
    return x, y, z


def view_surface():
    from enthought.mayavi import mlab
    mlab.mesh(*test_grid())
    mlab.draw()
