
import grid, axis, coordinate_system, mapping
import numpy.linalg as L
import numpy as N
import numpy.random as R

def from_origin_and_columns(origin, colvectors, shape, output_coords=None):
    """
    Return a grid representing a slice based on a given origin, direction vectors and shape.

    By default the output coordinate system is the MNI world.
    """

    nout = colvectors.shape[1]
    ndim = colvectors.shape[0]

    f = N.zeros((nout,)*2, N.Float)
    for i in range(ndim):
        f[0:nout,i] = colvectors[i]
    
    tmp = R.standard_normal((nout, nout-ndim))
    p = N.identity(nout) - N.dot(f, L.generalized_inverse(f))
    tmp = N.dot(p, tmp)
    f[0:nout,ndim:] = N.transpose(tmp)

    for i in range(nout-ndim):
        f[0:nout,ndim+i] = f[0:nout,ndim+i] / N.sqrt(N.add.reduce(f[0:nout,ndim+i]**2))

    t = N.zeros((nout+1,)*2, N.Float)
    t[0:nout,0:nout] = f
    t[nout,nout] = 1.
    t[0:nout,nout] = origin

    input_coords = coordinate_system.VoxelCoordinateSystem('slice',
                                                           axis.generic,
                                                           shape=shape)
    if output_coords is None:
        output_coords = coordinate_system.MNI_world

    w = mapping.Affine(input_coords, output_coords, t)
    g = grid.SamplingGrid(mapping=w, shape=list(shape + (1,) * (nout-ndim)))
    return g

# MNI default

default_xlim = [-90.,90.]
default_ylim = [-126.,90.]
default_zlim = [-72.,108.]
default_shape = (91,109,91)

def box_slices(zlim=default_zlim, ylim=default_ylim, xlim=default_xlim,
               shape=default_shape, x=N.inf, y=N.inf, z=N.inf):

    if x == N.inf:
        x = (xlim[0]+xlim[1])/2.

    if y == N.inf:
        y = (ylim[0]+ylim[1])/2.

    if z == N.inf:
        z = (zlim[0]+zlim[1])/2.

    origin = (zlim[0], ylim[0], xlim[0])

    # yslice, xslice, zslice

    origins = []

    origins.append([zlim[0], y, xlim[0]])
    origins.append([zlim[0], ylim[0], x])
    origins.append([z, ylim[0], xlim[0]])

    step = N.zeros((3,), N.float64)
    step[0] = (zlim[1] - zlim[0]) / (shape[0] - 1.)
    step[1] = (ylim[1] - ylim[0]) / (shape[1] - 1.)
    step[2] = (xlim[1] - xlim[0]) / (shape[2] - 1.)
    
    # yslice, xslice, zslice

    columns = []

    columns.append(N.array([[0,0,step[2]],[step[0],0,0]]))
    columns.append(N.array([[0,step[1],0],[step[0],0,0]]))
    columns.append(N.array([[0,step[1],0],[0,0,step[2]]]))

    # yslice, xslice, zslice

    shapes = []

    shapes.append((shape[0], shape[2]))
    shapes.append((shape[0], shape[1]))
    shapes.append((shape[2], shape[1]))

    # yslice, xslice, zslice

    slices = []

    for i in range(3):
        slices.append(from_origin_and_columns(origins[i],
                                              columns[i][::-1],
                                              shapes[i]))
    return slices

def yslice(y=0,
           zlim=default_zlim,
           ylim=default_ylim,
           xlim=default_xlim,
           shape=default_shape):
    """
    Return a slice through a 3d box with y fixed.
    Defaults to a slice through MNI coordinates.
    """

    return box_slices(xlim=xlim,
                      ylim=ylim,
                      zlim=zlim,
                      shape=shape,
                      y=y)[0]
def xslice(x=0,
           zlim=default_zlim,
           ylim=default_ylim,
           xlim=default_xlim,
           shape=default_shape):

    """
    Return a slice through a 3d box with x fixed.
    Defaults to a slice through MNI coordinates.
    """

    return box_slices(xlim=xlim,
                      ylim=ylim,
                      zlim=zlim,
                      shape=shape,
                      x=x)[1]

def zslice(z=0,
           zlim=default_zlim,
           ylim=default_ylim,
           xlim=default_xlim,
           shape=default_shape):
    
    """
    Return a slice through a 3d box with z fixed.
    Defaults to a slice through MNI coordinates.
    """

    return box_slices(xlim=xlim,
                      ylim=ylim,
                      zlim=zlim,
                      shape=shape,
                      z=z)[2]

def bounding_box(grid):
    """
    Determine a valid bounding box from a SamplingGrid instance.
    """

    r = grid.range()
    m = []
    M = []

    ndim = r.shape[0]
    for i in range(ndim):
        m.append(r[i].min())
        M.append(r[i].max())

    return [[m[i], M[i]] for i in range(ndim)]

def squeezeshape(shape):
    s = N.array(shape)
    keep = N.not_equal(s, 1)
    return tuple(s[keep])
    
