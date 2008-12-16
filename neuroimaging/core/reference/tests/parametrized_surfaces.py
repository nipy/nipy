"""
Polar coordinate systems using a CoordinateMap
"""

import numpy as np
from scipy.linalg import expm

from neuroimaging.core.api import CoordinateMap, CoordinateSystem, Axis, Affine
from neuroimaging.core.reference.coordinate_map import Grid, values

"""
A parametric representation of a sphere
"""

tp = CoordinateSystem('input', [Axis(l) for l in ['theta', 'phi']])
xyz = CoordinateSystem('output', [Axis(l) for l in ['x', 'y', 'z']])

def sphere(vals, r=3.):
    vals = tp.typecast(vals, tp.dtype)
    theta = vals['theta']
    phi = vals['phi']

    o = np.array([r*np.cos(theta)*np.sin(phi),
                  r*np.sin(theta)*np.sin(phi),
                  r*np.cos(phi)]).T
    return o

sphere_parametrization = CoordinateMap(sphere, tp, xyz)

uv = CoordinateSystem('input', [Axis(l) for l in ['u', 'v']])
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

if __name__ == '__main__':
    from enthought.mayavi import mlab
    g = Grid(uv)
    x, y, z = values(g[-1:1:101j,-2:2:101j], transpose=True)
    mlab.mesh(x,y,z)
    mlab.draw()
