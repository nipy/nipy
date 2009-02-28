"""
Parametrized surfaces using a CoordinateMap
"""
import numpy as np
import nose.tools

from neuroimaging.core.api import CoordinateMap, CoordinateSystem, Affine
from neuroimaging.core.api import Grid

uv = CoordinateSystem('uv', 'input')
xyz = CoordinateSystem('xyz', 'output')

def parametric(vals):
    """
    Parametrization of the surface x**2-y**2*z**2+z**3=0
    """
    u = vals[:,0]
    v = vals[:, 1]
    o = np.array([v*(u**2-v**2),
                  u,
                  u**2-v**2]).T
    return o

"""
Let's check that indeed this is a parametrization of that surface
"""

def implicit(vals):
    x = vals[:,0]; y = vals[:,1]; z = vals[:,2]
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

    uv32 = CoordinateSystem('uv', 'input', np.float32)
    xyz32 = CoordinateSystem('xyz', 'output', np.float32)

    def par32(vals):
        """
        float32 version of the same thing
        """
        u = vals[:,0]
        v = vals[:,1]
    
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

    uv32 = CoordinateSystem('uv', 'input', np.float32)
    xyz128 = CoordinateSystem('xyz', 'output', np.complex128)

    def par32_c128(vals):
        """
        float32 version of the same thing
        """
        u = vals[:,0]
        v = vals[:,1]
    
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
