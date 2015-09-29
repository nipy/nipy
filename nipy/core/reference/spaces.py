""" Useful neuroimaging coordinate map makers and utilities """
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from nibabel.affines import from_matvec

from ...fixes.nibabel import io_orientation

from .coordinate_system import CoordSysMaker, is_coordsys, is_coordsys_maker
from .coordinate_map import CoordMapMaker

from ...externals.six import string_types

class XYZSpace(object):
    """ Class contains logic for spaces with XYZ coordinate systems

    >>> sp = XYZSpace('hijo')
    >>> print(sp)
    hijo: [('x', 'hijo-x=L->R'), ('y', 'hijo-y=P->A'), ('z', 'hijo-z=I->S')]
    >>> csm = sp.to_coordsys_maker()
    >>> cs = csm(3)
    >>> cs
    CoordinateSystem(coord_names=('hijo-x=L->R', 'hijo-y=P->A', 'hijo-z=I->S'), name='hijo', coord_dtype=float64)
    >>> cs in sp
    True
    """
    x_suffix = 'x=L->R'
    y_suffix = 'y=P->A'
    z_suffix = 'z=I->S'

    def __init__(self, name):
        self.name = name

    @property
    def x(self):
        """ x-space coordinate name """
        return "%s-%s" % (self.name, self.x_suffix)

    @property
    def y(self):
        """ y-space coordinate name """
        return "%s-%s" % (self.name, self.y_suffix)

    @property
    def z(self):
        """ z-space coordinate name """
        return "%s-%s" % (self.name, self.z_suffix)

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.name)

    def __str__(self):
        return "%s: %s" % (self.name, sorted(self.as_map().items()))

    def __eq__(self, other):
        """ Equality defined as having the same xyz names """
        try:
            otuple = other.as_tuple()
        except AttributeError:
            return False
        return self.as_tuple() == otuple

    def __ne__(self, other):
        return not self == other

    def as_tuple(self):
        """ Return xyz names as tuple

        >>> sp = XYZSpace('hijo')
        >>> sp.as_tuple()
        ('hijo-x=L->R', 'hijo-y=P->A', 'hijo-z=I->S')
        """
        return self.x, self.y, self.z

    def as_map(self):
        """ Return xyz names as dictionary

        >>> sp = XYZSpace('hijo')
        >>> sorted(sp.as_map().items())
        [('x', 'hijo-x=L->R'), ('y', 'hijo-y=P->A'), ('z', 'hijo-z=I->S')]
        """
        return dict(zip('xyz', self.as_tuple()))

    def register_to(self, mapping):
        """ Update `mapping` with key=self.x, value='x' etc pairs

        The mapping will then have keys that are names we (``self``) identify as
        being x, or y, or z, values are 'x' or 'y' or 'z'.

        Note that this is the opposite way round for keys, values, compared to
        the ``as_map`` method.

        Parameters
        ----------
        mapping : mapping
            such as a dict

        Returns
        -------
        None

        Examples
        --------
        >>> sp = XYZSpace('hijo')
        >>> mapping = {}
        >>> sp.register_to(mapping)
        >>> sorted(mapping.items())
        [('hijo-x=L->R', 'x'), ('hijo-y=P->A', 'y'), ('hijo-z=I->S', 'z')]
        """
        mapping.update(dict(zip(self.as_tuple(), 'xyz')))

    def to_coordsys_maker(self, extras=()):
        """ Make a coordinate system maker for this space

        Parameters
        ----------
        extra : sequence
            names for any further axes after x, y, z

        Returns
        -------
        csm : CoordinateSystemMaker

        Examples
        --------
        >>> sp = XYZSpace('hijo')
        >>> csm = sp.to_coordsys_maker()
        >>> csm(3)
        CoordinateSystem(coord_names=('hijo-x=L->R', 'hijo-y=P->A', 'hijo-z=I->S'), name='hijo', coord_dtype=float64)
        """
        return CoordSysMaker(self.as_tuple() + tuple(extras), name=self.name)

    def __contains__(self, obj):
        """ True if `obj` can be thought of as being 'in' this space

        `obj` is an object that is in some kind of space - it can be a
        coordinate system, a coordinate map, or an object with a ``coordmap``
        attribute.  We test the output coordinate system of `obj` against our
        own space definition.

        A coordinate system is in our space if it has all the axes of our space.

        Parameters
        ----------
        obj : object
            Usually a coordinate system, a coordinate map, or an Image (with a
            ``coordmap`` attribute)

        Returns
        -------
        tf : bool
            True if `obj` is 'in' this space

        Examples
        --------
        >>> from nipy.core.api import Image, AffineTransform, CoordinateSystem
        >>> sp = XYZSpace('hijo')
        >>> names = sp.as_tuple()
        >>> cs = CoordinateSystem(names)
        >>> cs in sp
        True
        >>> cs = CoordinateSystem(names + ('another_name',))
        >>> cs in sp
        True
        >>> cmap = AffineTransform('ijk', names, np.eye(4))
        >>> cmap in sp
        True
        >>> img = Image(np.zeros((3,4,5)), cmap)
        >>> img in sp
        True
        """
        try:
            obj = obj.coordmap
        except AttributeError:
            pass
        try:
            obj = obj.function_range
        except AttributeError:
            pass
        my_names = self.as_tuple()
        return set(my_names).issubset(obj.coord_names)


# Generic coordinate map maker for voxels (function_domain). Unlike nifti
# loading, by default the 4th axis is not time (because we don't know what it
# is).
voxel_csm = CoordSysMaker('ijklmnop', 'voxels')

# Module level mapping from key=name to values in 'x' or 'y' or 'z'
known_names = {}
known_spaces = []

# Standard spaces defined
for _name in ('unknown', 'scanner', 'aligned', 'mni', 'talairach'):
    _space = XYZSpace(_name)
    known_spaces.append(_space)
    _space.register_to(known_names)
    _csm = _space.to_coordsys_maker('tuvw')
    _cmm = CoordMapMaker(voxel_csm, _csm)
    # Put these into the module namespace
    exec('%s_space = _space' % _name)
    exec('%s_csm = _csm' % _name)
    exec('vox2%s = _cmm' % _name)


def known_space(obj, spaces=None):
    """ If `obj` is in a known space, return the space, otherwise return None

    Parameters
    ----------
    obj : object
        Object that can be tested against an XYZSpace with ``obj in sp``
    spaces : None or sequence, optional
        spaces to test against.  If None, use the module level ``known_spaces``
        list to test against.

    Returns
    -------
    sp : None or XYZSpace
        If `obj` is not in any of the `known_spaces`, return None.  Otherwise
        return the first matching space in `known_spaces`

    Examples
    --------
    >>> from nipy.core.api import CoordinateSystem
    >>> sp0 = XYZSpace('hijo')
    >>> sp1 = XYZSpace('hija')

    Make a matching coordinate system

    >>> cs = sp0.to_coordsys_maker()(3)

    Test whether this coordinate system is in either of ``(sp0, sp1)``

    >>> known_space(cs, (sp0, sp1))
    XYZSpace('hijo')

    So, yes, it's in ``sp0``. How about another generic CoordinateSystem?

    >>> known_space(CoordinateSystem('xyz'), (sp0, sp1)) is None
    True

    So, no, that is not in either of ``(sp0, sp1)``
    """
    if spaces is None:
        # use module level global
        spaces = known_spaces
    for sp in spaces:
        if obj in sp:
            return sp
    return None


def get_world_cs(world_id, ndim=3, extras='tuvw', spaces=None):
    """ Get world coordinate system from `world_id`

    Parameters
    ----------
    world_id : str, XYZSPace, CoordSysMaker or CoordinateSystem
        Object defining a world output system.  If str, then should be a name of
        an XYZSpace in the list `spaces`.
    ndim : int, optional
        Number of dimensions in this world.  Default is 3
    extras : sequence, optional
        Coordinate (axis) names for axes > 3 that are not named by `world_id`
    spaces : None or sequence, optional
        List of known (named) spaces to compare a str `world_id` to.  If None,
        use the module level ``known_spaces``

    Returns
    -------
    world_cs : CoordinateSystem
        A world coordinate system

    Examples
    --------
    >>> get_world_cs('mni')
    CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S'), name='mni', coord_dtype=float64)

    >>> get_world_cs(mni_space, 4)
    CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64)

    >>> from nipy.core.api import CoordinateSystem
    >>> get_world_cs(CoordinateSystem('xyz'))
    CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64)
    """
    if is_coordsys(world_id):
        if world_id.ndim != ndim:
            raise SpaceError("Need %d-dimensional CoordinateSystem" % ndim)
        return world_id
    if spaces is None:
        spaces = known_spaces
    if isinstance(world_id, string_types):
        space_names = [s.name for s in spaces]
        if world_id not in space_names:
            raise SpaceError('Unkown space "%s"; known spaces are %s'
                                    % (world_id, ', '.join(space_names)))
        world_id = spaces[space_names.index(world_id)]
    if is_xyz_space(world_id):
        world_id = world_id.to_coordsys_maker(extras)
    if is_coordsys_maker(world_id):
        return world_id(ndim)
    raise ValueError('Expecting CoordinateSystem, CoordSysMaker, '
                     'XYZSpace, or str, got %s' % world_id)


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
    name2xyz : None or mapping, optional
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
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='voxels', coord_dtype=float64),
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
    name2xyz : None or mapping, optional
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
    >>> xyzt_cs = mni_csm(4) # coordsys with t (time) last
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


def is_xyz_space(obj):
    """ True if `obj` appears to be an XYZ space definition """
    return (hasattr(obj, 'x') and
            hasattr(obj, 'y') and
            hasattr(obj, 'z') and
            hasattr(obj, 'to_coordsys_maker'))


def is_xyz_affable(coordmap, name2xyz=None):
    """ Return True if the coordap has an xyz affine

    Parameters
    ----------
    coordmap : ``CoordinateMap`` instance
        Coordinate map to test
    name2xyz : None or mapping, optional
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
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='voxels', coord_dtype=float64),
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
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='voxels', coord_dtype=float64),
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
