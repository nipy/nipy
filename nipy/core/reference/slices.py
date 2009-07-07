"""
A set of methods to get coordinate maps which represent slices in space.

"""
from nipy.core.reference.coordinate_system import CoordinateSystem
from nipy.core.reference.coordinate_map import AffineTransform
from nipy.core.reference.array_coords import ArrayCoordMap
import numpy as np

from nipy.core.transforms.affines import from_matrix_vector

lps_output_coordnames = ('x+LR', 'y+PA', 'z+SI')

__docformat__ = 'restructuredtext'

def xslice(x, y_spec, z_spec, output_space=''):
    """
    Return an LPS slice through a 3d box with x fixed.

    Parameters
    ----------
    x : float
       The value at which x is fixed.

    yspec : ([float,float], int)
       Tuple representing the y-limits of the bounding 
       box and the number of points.

    zspec : ([float,float], int)
       Tuple representing the z-limits of the bounding 
       box and the number of points.

    output_space : str, optional
       Origin of the range CoordinateSystem.

    Returns
    -------

    affine_transform : AffineTransform
       An affine transform that describes an plane in 
       LPS coordinates with x fixed.

    >>> y_spec = ([-114,114], 115) # voxels of size 2 in y, starting at -114, ending at 114
    >>> z_spec = ([-70,100], 86) # voxels of size 2 in z, starting at -70, ending at 100
    >>> x30 = xslice(30, y_spec, z_spec)
    >>> x30([0,0])
    array([  30., -114.,  -70.])
    >>> x30([114,85])
    array([  30.,  114.,  100.])
    >>> x30
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i_y', 'i_z'), name='slice', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       affine=array([[   0.,    0.,   30.],
                     [   2.,    0., -114.],
                     [   0.,    2.,  -70.],
                     [   0.,    0.,    1.]])
    )
    >>> 

    >>> bounding_box(x30, (y_spec[1], z_spec[1]))
    ([30.0, 30.0], [-114.0, 114.0], [-70.0, 100.0])

    """
    zlim = z_spec[0]
    ylim = y_spec[0]
    shape = (z_spec[1], y_spec[1])
    origin = [x,ylim[0],zlim[0]]
    colvectors = [[0,(ylim[1]-ylim[0])/(shape[1] - 1.),0],
                  [0,0,(zlim[1]-zlim[0])/(shape[0] - 1.)]]

    T = from_matrix_vector(np.vstack(colvectors).T, origin)
    affine_domain = CoordinateSystem(['i_y', 'i_z'], 'slice')
    affine_range = CoordinateSystem(lps_output_coordnames, output_space)
    return AffineTransform(affine_domain,
                           affine_range,
                           T)

def yslice(y, x_spec, z_spec, output_space=''):
    """
    Return a slice through a 3d box with y fixed.

    Parameters
    ----------
    y : float
       The value at which y is fixed.

    xspec : ([float,float], int)
       Tuple representing the x-limits of the bounding 
       box and the number of points.

    zspec : ([float,float], int)
       Tuple representing the z-limits of the bounding 
       box and the number of points.

    output_space : str, optional
       Origin of the range CoordinateSystem.

    Returns
    -------

    affine_transform : AffineTransform
       An affine transform that describes an plane in 
       LPS coordinates with y fixed.

    Examples
    --------

    >>> x_spec = ([-92,92], 93) # voxels of size 2 in x, starting at -92, ending at 92
    >>> z_spec = ([-70,100], 86) # voxels of size 2 in z, starting at -70, ending at 100
    >>> 

    >>> y70 = yslice(70, x_spec, z_spec)
    >>> y70
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i_x', 'i_z'), name='slice', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       affine=array([[  2.,   0., -92.],
                     [  0.,   0.,  70.],
                     [  0.,   2., -70.],
                     [  0.,   0.,   1.]])
    )

    >>> y70([0,0])
    array([-92.,  70., -70.])
    >>> y70([92,85])
    array([  92.,   70.,  100.])
    >>> 
    >>> bounding_box(y70, (x_spec[1], z_spec[1]))
    ([-92.0, 92.0], [70.0, 70.0], [-70.0, 100.0])

    """
    xlim = x_spec[0]
    zlim = z_spec[0]
    shape = (z_spec[1], x_spec[1])

    origin = [xlim[0],y,zlim[0]]
    colvectors = [[(xlim[1]-xlim[0])/(shape[1] - 1.),0,0],
                  [0,0,(zlim[1]-zlim[0])/(shape[0] - 1.)]]

    T = from_matrix_vector(np.vstack(colvectors).T, origin)
    affine_domain = CoordinateSystem(['i_x', 'i_z'], 'slice')
    affine_range = CoordinateSystem(lps_output_coordnames, output_space)
    return AffineTransform(affine_domain,
                           affine_range,
                           T)

def zslice(z, x_spec, y_spec, output_space=''):
    """
    Return a slice through a 3d box with z fixed.

    Parameters
    ----------
    z : float
       The value at which z is fixed.

    x_spec : ([float,float], int)
       Tuple representing the x-limits of the bounding 
       box and the number of points.

    y_spec : ([float,float], int)
       Tuple representing the y-limits of the bounding 
       box and the number of points.

    output_space : str, optional
       Origin of the range CoordinateSystem.

    Returns
    -------

    affine_transform : AffineTransform
       An affine transform that describes an plane in 
       LPS coordinates with z fixed.

    Examples
    --------

    >>> x_spec = ([-92,92], 93) # voxels of size 2 in x, starting at -92, ending at 92
    >>> y_spec = ([-114,114], 115) # voxels of size 2 in y, starting at -114, ending at 114
    >>> z40 = zslice(40, x_spec, y_spec)
    >>> z40
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i_x', 'i_y'), name='slice', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       affine=array([[   2.,    0.,  -92.],
                     [   0.,    2., -114.],
                     [   0.,    0.,   40.],
                     [   0.,    0.,    1.]])
    )
    >>> 
    >>> z40([0,0])
    array([ -92., -114.,   40.])
    >>> z40([92,114])
    array([  92.,  114.,   40.])

    >>> bounding_box(z40, (x_spec[1], y_spec[1]))
    ([-92.0, 92.0], [-114.0, 114.0], [40.0, 40.0])

    """
    xlim = x_spec[0]
    ylim = y_spec[0]
    shape = (y_spec[1], x_spec[1])
    origin = [xlim[0],ylim[0],z]
    colvectors = [[(xlim[1]-xlim[0])/(shape[1] - 1.),0,0],
                  [0,(ylim[1]-ylim[0])/(shape[0] - 1.),0]]

    T = from_matrix_vector(np.vstack(colvectors).T, origin)
    affine_domain = CoordinateSystem(['i_x', 'i_y'], 'slice')
    affine_range = CoordinateSystem(lps_output_coordnames, output_space)
    return AffineTransform(affine_domain,
                           affine_range,
                           T)


def bounding_box(coordmap, shape):
    """
    Determine a valid bounding box from a CoordinateMap 
    and a shape.

    Parameters
    ----------
    coordmap : CoordinateMap or AffineTransform

    shape : (int)
       Tuple of ints.
       
    Returns
    -------

    limits : (float)
       Tuple of floats.


    Examples
    --------

    >>> A = AffineTransform.from_start_step('ijk', lps_output_coordnames, [2,4,6], [1,3,5])
    >>> bounding_box(A, (30,40,20))
    ([2.0, 31.0], [4.0, 121.0], [6.0, 101.0])
    >>> 

    """
    e = ArrayCoordMap.from_shape(coordmap, shape)
    return tuple([[r.min(), r.max()] for r in e.transposed_values])
    
