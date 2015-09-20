# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parametrized surfaces using a CoordinateMap
"""
from __future__ import absolute_import
import numpy as np

from nose.tools import assert_equal

from nipy.core.api import CoordinateMap, CoordinateSystem
from nipy.core.api import Grid

uv = CoordinateSystem('uv', 'input')
xyz = CoordinateSystem('xyz', 'output')

def parametric_mapping(vals):
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

surface_param = CoordinateMap(uv, xyz, parametric_mapping)

def test_surface():
    assert np.allclose(
        implicit(
            parametric_mapping(
                np.random.standard_normal((40,2))
                )
            ), 
        0)

def test_grid():
    g = Grid(surface_param)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    yield assert_equal, x.shape, (201,101)
    yield assert_equal, y.shape, (201,101)
    yield assert_equal, z.shape, (201,101)

def test_grid32():
    # Check that we can use a float32 input and output
    uv32 = CoordinateSystem('uv', 'input', np.float32)
    xyz32 = CoordinateSystem('xyz', 'output', np.float32)
    surface32 = CoordinateMap(uv32, xyz32, parametric_mapping)
    g = Grid(surface32)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    yield assert_equal, x.shape, (201,101)
    yield assert_equal, y.shape, (201,101)
    yield assert_equal, z.shape, (201,101)
    yield assert_equal, x.dtype, np.dtype(np.float32)

def test_grid32_c128():
    # Check that we can use a float32 input and complex128 output
    uv32 = CoordinateSystem('uv', 'input', np.float32)
    xyz128 = CoordinateSystem('xyz', 'output', np.complex128)
    def par_c128(x):
        return parametric_mapping(x).astype(np.complex128)
    surface = CoordinateMap(uv32, xyz128, par_c128)
    g = Grid(surface)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    yield assert_equal, x.shape, (201,101)
    yield assert_equal, y.shape, (201,101)
    yield assert_equal, z.shape, (201,101)
    yield assert_equal, x.dtype, np.dtype(np.complex128)


def view_surface():
    from enthought.mayavi import mlab
    g = Grid(surface_param)
    xyz_grid = g[-1:1:201j,-1:1:101j]
    x, y, z = xyz_grid.transposed_values
    mlab.mesh(x, y, z)
    mlab.draw()
