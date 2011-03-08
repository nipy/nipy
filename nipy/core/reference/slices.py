# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A set of methods to get coordinate maps which represent slices in space.
"""
import numpy as np

from nipy.core.reference.coordinate_system import CoordinateSystem
from nipy.core.reference.coordinate_map import AffineTransform
from nipy.core.reference.array_coords import ArrayCoordMap
from nipy.core.transforms.affines import from_matrix_vector

lps_output_coordnames = ('x+LR', 'y+PA', 'z+SI')

def xslice(x, y_spec, z_spec, output_space=''):
    """
    Return an LPS slice through a 3d box with x fixed.

    Parameters
    ----------
    x : float
       The value at which x is fixed.
    y_spec : sequence
       A sequence with 2 values of form ((float, float), int). The
       (float, float) components are the min and max y values; the int
       is the number of points.
    z_spec : sequence
       As for `y_spec` but for z
    output_space : str, optional
       Origin of the range CoordinateSystem.

    Returns
    -------
    affine_transform : AffineTransform
       An affine transform that describes an plane in
       LPS coordinates with x fixed.

    Examples
    --------
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
    >>> bounding_box(x30, (y_spec[1], z_spec[1]))
    ((30.0, 30.0), (-114.0, 114.0), (-70.0, 100.0))
    """
    (ymin, ymax), yno = y_spec
    y_tick = (ymax-ymin) / (yno - 1.0)
    (zmin, zmax), zno = z_spec
    z_tick = (zmax-zmin) / (zno - 1.0)
    origin = [x, ymin, zmin]
    colvectors = np.asarray([[0, 0],
                             [y_tick, 0],
                             [0, z_tick]])
    T = from_matrix_vector(colvectors, origin)
    affine_domain = CoordinateSystem(['i_y', 'i_z'], 'slice')
    affine_range = CoordinateSystem(lps_output_coordnames, output_space)
    return AffineTransform(affine_domain,
                           affine_range,
                           T)


def yslice(y, x_spec, z_spec, output_space=''):
    """ Return a slice through a 3d box with y fixed.

    Parameters
    ----------
    y : float
       The value at which y is fixed.
    x_spec : sequence
       A sequence with 2 values of form ((float, float), int). The
       (float, float) components are the min and max x values; the int
       is the number of points.
    z_spec : sequence
       As for `x_spec` but for z
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
    ((-92.0, 92.0), (70.0, 70.0), (-70.0, 100.0))
    """
    (xmin, xmax), xno = x_spec
    x_tick = (xmax-xmin) / (xno - 1.0)
    (zmin, zmax), zno = z_spec
    z_tick = (zmax-zmin) / (zno - 1.0)
    origin = [xmin, y, zmin]
    colvectors = np.asarray([[x_tick, 0],
                             [0, 0],
                             [0, z_tick]])
    T = from_matrix_vector(colvectors, origin)
    affine_domain = CoordinateSystem(['i_x', 'i_z'], 'slice')
    affine_range = CoordinateSystem(lps_output_coordnames, output_space)
    return AffineTransform(affine_domain,
                           affine_range,
                           T)


def zslice(z, x_spec, y_spec, output_space=''):
    """ Return a slice through a 3d box with z fixed.

    Parameters
    ----------
    z : float
       The value at which z is fixed.
    x_spec : sequence
       A sequence with 2 values of form ((float, float), int). The
       (float, float) components are the min and max x values; the int
       is the number of points.
    y_spec : sequence
       As for `x_spec` but for y
    output_space : str, optional
       Origin of the range CoordinateSystem.

    Returns
    -------
    affine_transform : AffineTransform
       An affine transform that describes a plane in LPS coordinates with z
       fixed.

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
    >>> z40([0,0])
    array([ -92., -114.,   40.])
    >>> z40([92,114])
    array([  92.,  114.,   40.])
    >>> bounding_box(z40, (x_spec[1], y_spec[1]))
    ((-92.0, 92.0), (-114.0, 114.0), (40.0, 40.0))
    """
    (xmin, xmax), xno = x_spec
    x_tick = (xmax-xmin) / (xno - 1.0)
    (ymin, ymax), yno = y_spec
    y_tick = (ymax-ymin) / (yno - 1.0)
    origin = [xmin, ymin, z]
    colvectors = np.asarray([[x_tick, 0],
                             [0, y_tick],
                             [0, 0]])
    T = from_matrix_vector(colvectors, origin)
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
       Containing mapping between voxel coordinates implied by `shape` and
       physical coordinates.
    shape : sequence of int
       shape implying array

    Returns
    -------
    limits : (N,) tuple of (2,) tuples of float
       minimum and maximum coordinate values in output space (range) of
       `coordmap`. N is given by coordmap.ndim[1].

    Examples
    --------
    >>> A = AffineTransform.from_start_step('ijk', lps_output_coordnames, [2,4,6], [1,3,5])
    >>> bounding_box(A, (30,40,20))
    ((2.0, 31.0), (4.0, 121.0), (6.0, 101.0))
    """
    e = ArrayCoordMap.from_shape(coordmap, shape)
    return tuple([(r.min(), r.max()) for r in e.transposed_values])

