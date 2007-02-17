"""
A set of methods to get sampling grids which represent slices in space.
"""

__docformat__ = 'restructuredtext'

from neuroimaging.core.reference import grid, axis, mapping, mni
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem
import numpy.linalg as L
import numpy as N
import numpy.random as R

def from_origin_and_columns(origin, colvectors, shape, output_coords=None):
    """
    Return a grid representing a slice based on a given origin, a pair of direction
    vectors which span the slice, and a shape.

    By default the output coordinate system is the MNI world.

    :Returns: `grid.SamplingGrid`
    """

    nout = colvectors.shape[1]
    ndim = colvectors.shape[0]

    f = N.zeros((nout,)*2)
    for i in range(ndim):
        f[0:nout,i] = colvectors[i]
    
    p = N.identity(nout) - N.dot(f, L.pinv(f))
    tmp = R.standard_normal((nout, nout-ndim))
    tmp = N.dot(p, tmp)
    f[0:nout, ndim:] = tmp
    for i in range(nout-ndim):
        f[0:nout,ndim+i] = f[0:nout,ndim+i] / N.sqrt(N.add.reduce(f[0:nout,ndim+i]**2))

    t = N.zeros((nout+1,)*2)
    t[0:nout, 0:nout] = f
    t[nout, nout] = 1.
    t[0:nout, nout] = origin

    input_coords = VoxelCoordinateSystem('slice', axis.generic,
                                         shape=shape + (1,))
    if output_coords is None:
        output_coords = mni.MNI_world

    w = mapping.Affine(t)
    g = grid.SamplingGrid(list(shape + (1,) * (nout-ndim)), w, input_coords, output_coords)
    return g


def box_slices(zlim, ylim, xlim, shape, x=N.inf, y=N.inf, z=N.inf):
    """
    Create a set of 3 sampling grids representing slices along each plane.
    """
    if x == N.inf:
        x = (xlim[0]+xlim[1])/2.

    if y == N.inf:
        y = (ylim[0]+ylim[1])/2.

    if z == N.inf:
        z = (zlim[0]+zlim[1])/2.

    # yslice, xslice, zslice
    origins = [[zlim[0], y,       xlim[0]],
               [zlim[0], ylim[0], x],
               [z,       ylim[0], xlim[0]]]

    step = [(zlim[1] - zlim[0]) / (shape[0] - 1.),
            (ylim[1] - ylim[0]) / (shape[1] - 1.),
            (xlim[1] - xlim[0]) / (shape[2] - 1.)]

    # yslice, xslice, zslice
    columns = [N.array([[0,0,step[2]], [step[0],0,0]]),
               N.array([[0,step[1],0], [step[0],0,0]]),
               N.array([[0,step[1],0], [0,0,step[2]]])]

    # yslice, xslice, zslice
    shapes = [(shape[0], shape[2]),
              (shape[0], shape[1]),
              (shape[2], shape[1])]

    # yslice, xslice, zslice
    slices = []
    for i in range(3):
        slices.append(from_origin_and_columns(origins[i],
                                              columns[i][::-1],
                                              shapes[i]))
    return slices

def yslice(y, zlim, ylim, xlim, shape):
    """
    Return a slice through a 3d box with y fixed.
    Defaults to a slice through MNI coordinates.
    """
    return box_slices(zlim, ylim, xlim, shape, y=y)[0]

def xslice(x, zlim, ylim, xlim, shape):
    """
    Return a slice through a 3d box with x fixed.
    Defaults to a slice through MNI coordinates.
    """
    return box_slices(zlim, ylim, xlim, shape, x=x)[1]

def zslice(z, zlim, ylim, xlim, shape):    
    """
    Return a slice through a 3d box with z fixed.
    Defaults to a slice through MNI coordinates.
    """
    return box_slices(zlim, ylim, xlim, shape, z=z)[2]

def bounding_box(grid):
    """
    Determine a valid bounding box from a SamplingGrid instance.
    """
    return [[r.min(), r.max()] for r in grid.range()]
    
