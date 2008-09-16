"""
A set of methods to get coordinate maps which represent slices in space.

"""

__docformat__ = 'restructuredtext'

from neuroimaging.core.reference import coordinate_map, axis, mapping, mni
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem
import numpy.linalg as L
import numpy as np
import numpy.random as R

def from_origin_and_columns(origin, colvectors, shape, output_coords):
    """
    Return a coordinate_map representing a slice based on a given origin, a pair of direction
    vectors which span the slice, and a shape.

    By default the output coordinate system is the MNI world.

    :Parameters:
        origin : the corner of the output coordinates, i.e. the [0]*ndimin
                 point
        colvectors : the steps in each voxel direction
        shape : how many steps in each voxel direction
        output_coords : a CoordinateSystem for the output

    :Returns: `coordinate_map.CoordinateMap`
    """
    colvectors = np.asarray(colvectors)
    nout = colvectors.shape[1]
    nin = colvectors.shape[0]

    f = np.zeros((nout+1,nin+1))
    for i in range(nin):
        f[0:nout,i] = colvectors[i]
    f[0:nout,-1] = origin
    f[nout, nin] = 1.

    input_coords = VoxelCoordinateSystem('slice', \
       [axis.VoxelAxis('voxel%d' % d, length=shape[d])
        for d in range(len(shape))])

    w = mapping.Affine(f)
    g = coordinate_map.CoordinateMap(w, input_coords, output_coords)
    return g


def xslice(x, zlim, ylim, output_coords, shape):
    """
    Return a slice through a 3d box with x fixed.
    Defaults to a slice through MNI coordinates.

    :Parameters:
        y : TODO
            TODO
        zlim : TODO
            TODO
        ylim : TODO
            TODO
        xlim : TODO
            TODO
        shape : TODO
            TODO
        output_coords : TODO
            TODO
    """
    origin = [zlim[0],ylim[0],x]
    colvectors = [[(zlim[1]-zlim[0])/(shape[1] - 1.),0,0],
                  [0,(ylim[1]-ylim[0])/(shape[0] - 1.),0]]
    return from_origin_and_columns(origin, colvectors, shape, output_coords)

def yslice(y, zlim, xlim, output_coords, shape):
    """
    Return a slice through a 3d box with y fixed.
    Defaults to a slice through MNI coordinates.

    :Parameters:
        x : TODO
            TODO
        zlim : TODO
            TODO
        ylim : TODO
            TODO
        xlim : TODO
            TODO
        shape : TODO
            TODO
        output_coords : TODO
            TODO
    """
    origin = [zlim[0],y,xlim[0]]
    colvectors = [[(zlim[1]-zlim[0])/(shape[1] - 1.),0,0],
                  [0,0,(xlim[1]-xlim[0])/(shape[0] - 1.)]]
    return from_origin_and_columns(origin, colvectors, shape, output_coords)

def zslice(z, ylim, xlim, output_coords, shape):    
    """
    Return a slice through a 3d box with z fixed.
    Defaults to a slice through MNI coordinates.

    :Parameters:
        z : TODO
            TODO
        ylim : TODO
            TODO
        xlim : TODO
            TODO
        shape : TODO
            TODO
        output_coords : TODO
            TODO
    """
    origin = [z,xlim[0],ylim[0]]
    colvectors = [[0,(ylim[1]-ylim[0])/(shape[1] - 1.),0],
                  [0,0,(xlim[1]-xlim[0])/(shape[0] - 1.)]]
    return from_origin_and_columns(origin, colvectors, shape, output_coords)



def bounding_box(coordmap):
    """
    Determine a valid bounding box from a CoordinateMap instance.

    :Parameters:
        coordmap : `coordinate_map.CoordinateMap`
            TODO
    """
    return [[r.min(), r.max()] for r in coordmap.range()]
    
