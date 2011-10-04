""" Useful neuroimaging coordinate map makers and utilities """

import numpy as np

from nibabel.affines import from_matvec

from ...fixes.nibabel import io_orientation

from .coordinate_system import CoordSysMaker
from .coordinate_map import CoordMapMaker

_labels = 'x=L->R', 'y=P->A', 'z=I->S'
scanner_names = ['scanner-' + label for label in _labels] + ['t']
unknown_names = ['unknown-' + label for label in _labels] + ['t']
aligned_names = ['aligned-' + label for label in _labels] + ['t']
mni_names = ['mni-' + label for label in _labels] + ['t']
talairach_names = ['talairach-' + label for label in _labels] + ['t']

# Some standard coordinate system makers
voxel_cs = CoordSysMaker('ijkl', 'array')
scanner_cs = CoordSysMaker(scanner_names, 'scanner')
unknown_cs = CoordSysMaker(unknown_names, 'unknown')
aligned_cs = CoordSysMaker(aligned_names, 'aligned')
mni_cs = CoordSysMaker(mni_names, 'mni')
talairach_cs = CoordSysMaker(talairach_names, 'talairach')

# Standard coordinate map makers
vox2scanner = CoordMapMaker(voxel_cs, scanner_cs)
vox2unknown = CoordMapMaker(voxel_cs, unknown_cs)
vox2aligned = CoordMapMaker(voxel_cs, aligned_cs)
vox2mni = CoordMapMaker(voxel_cs, mni_cs)
vox2talairach = CoordMapMaker(voxel_cs, talairach_cs)

# Register these xyzs as known
known_names = {}
for _rcs in (scanner_names, mni_names, talairach_names):
    for _name, _coord in zip(_rcs[:3], 'xyz'):
        known_names[_name] = _coord


class SpaceError(Exception):
    pass

class SpaceTypeError(SpaceError):
    pass

class AxesError(SpaceError):
    pass

class AffineError(SpaceError):
    pass

def xyz_affine(coordmap, name2xyz=None):
    """ Return (4, 4) affine mapping voxel coordinates to XYZ from `coordmap`

    If no (4, 4) affine "makes sense"(TM) for this `coordmap` then raise errors
    listed below.  A (4, 4) affine makes sense if the first three output axes
    are recognizably X, Y, and Z in that order AND they there are corresponding
    input dimensions, AND the corresponding input dimensions are the first three
    input dimension (in any order).  Thus the input axes have to be 3D.

    Parameters
    ----------
    coordmap : ``CoordinateMap`` instance
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.

    Returns
    -------
    xyz_aff : (4,4) array
        voxel to X, Y, Z affine mapping

    Raises
    ------
    SpaceTypeError : if this is not an affine coordinate map
    AxesError : if not all of x, y, z recognized in `coordmap` output, or they
    are in the wrong order, or the x, y, z axes do not correspond to the first
    three input axes.
    AffineError : if axes dropped from the affine contribute to x, y, z
    coordinates.

    Notes
    -----
    We could also try and "make sense" (TM) of a coordmap that had X, Y and Z
    outputs, but not in that order, nor all in the first three axes.  In that
    case we could just permute the affine to get the output order we need.  But,
    that could become confusing if the returned affine has different output
    coordinates than the passed `coordmap`.  And it's more complicated.  So,
    let's not do that for now.

    Examples
    --------
    >>> cmap = vox2mni(np.diag([2,3,4,5,1]))
    >>> cmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> xyz_affine(cmap)
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  3.,  0.,  0.],
           [ 0.,  0.,  4.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    if name2xyz is None:
        name2xyz = known_names
    try:
        affine = coordmap.affine
    except AttributeError:
        raise SpaceTypeError('Need affine coordinate map')
    order = xyz_order(coordmap.function_range, name2xyz)
    if order[:3] != [0, 1, 2]:
        raise AxesError('First 3 output axes must be X, Y, Z')
    # Check equivalent input axes
    ornt = io_orientation(affine)
    if set(ornt[:3, 0]) != set((0, 1, 2)):
        raise AxesError('First 3 input axes must correspond to X, Y, Z')
    # Check that dropped dimensions don't provide xyz coordinate info
    extra_cols = affine[:3,3:-1]
    if not np.allclose(extra_cols, 0):
        raise AffineError('Dropped dimensions not orthogonal to xyz')
    return from_matvec(affine[:3,:3], affine[:3,-1])


def xyz_order(coordsys, name2xyz=None):
    """ Vector of orders for sorting coordsys axes in xyz first order

    Parameters
    ----------
    coordsys : ``CoordinateSystem`` instance
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.

    Returns
    -------
    xyz_order : list
        Ordering of axes to get xyz first ordering.  See the examples.

    Raises
    ------
    AxesError : if there are not all of x, y and z axes

    Examples
    --------
    >>> from nipy.core.api import CoordinateSystem
    >>> xyzt_cs = mni_cs(4) # coordsys with t (time) last
    >>> xyzt_cs
    CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64)
    >>> xyz_order(xyzt_cs)
    [0, 1, 2, 3]
    >>> tzyx_cs = CoordinateSystem(xyzt_cs.coord_names[::-1], 'reversed')
    >>> tzyx_cs
    CoordinateSystem(coord_names=('t', 'mni-z=I->S', 'mni-y=P->A', 'mni-x=L->R'), name='reversed', coord_dtype=float64)
    >>> xyz_order(tzyx_cs)
    [3, 2, 1, 0]
    """
    if name2xyz is None:
        name2xyz = known_names
    names = coordsys.coord_names
    N = len(names)
    axvals = np.zeros(N, dtype=int)
    for i, name in enumerate(names):
        try:
            xyz_char = name2xyz[name]
        except KeyError:
            axvals[i] = N+i
        else:
            axvals[i] = 'xyz'.index(xyz_char)
    if not set(axvals).issuperset(range(3)):
        raise AxesError("Not all of x, y, z recognized in coordinate map")
    return list(np.argsort(axvals))


def is_xyz_affable(coordmap, name2xyz=None):
    """ Return True if the coordap has an xyz affine

    Parameters
    ----------
    coordmap : ``CoordinateMap`` instance
        Coordinate map to test
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.

    Returns
    -------
    tf : bool
        True if `coordmap` has an xyz affine, False otherwise

    Examples
    --------
    >>> cmap = vox2mni(np.diag([2,3,4,5,1]))
    >>> cmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> is_xyz_affable(cmap)
    True
    >>> time0_cmap = cmap.reordered_domain([3,0,1,2])
    >>> time0_cmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.,  0.],
                     [ 0.,  0.,  0.,  4.,  0.],
                     [ 5.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> is_xyz_affable(time0_cmap)
    False
    """
    try:
        xyz_affine(coordmap, name2xyz)
    except SpaceError:
        return False
    return True
