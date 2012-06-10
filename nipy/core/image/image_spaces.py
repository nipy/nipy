""" Utilities for working with Images and common neuroimaging spaces

Images are very general things, and don't know anything about the kinds of
spaces they refer to, via their coordinate map.

There are a set of common neuroimaging spaces.  When we create neuroimaging
Images, we want to place them in neuroimaging spaces, and return information
about common neuroimaging spaces.

We do this by putting information about neuroimaging spaces in functions and
variables in the ``nipy.core.reference.spaces`` module, and in this module.

This keeps the specific neuroimaging spaces out of our Image object.

>>> from nipy.core.api import Image, vox2mni, rollimg, xyz_affine, as_xyz_image

Make a standard 4D xyzt image in MNI space.

First the data and affine:

>>> data = np.arange(24).reshape((1,2,3,4))
>>> affine = np.diag([2,3,4,1]).astype(float)

We can add the TR (==2.0) to make the full 5x5 affine we need

>>> img = Image(data, vox2mni(affine, 2.0))
>>> img.affine
array([[ 2.,  0.,  0.,  0.,  0.],
       [ 0.,  3.,  0.,  0.,  0.],
       [ 0.,  0.,  4.,  0.,  0.],
       [ 0.,  0.,  0.,  2.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])

In this case the neuroimaging 'xyz_affine' is just the 4x4 from the 5x5 in the image

>>> xyz_affine(img)
array([[ 2.,  0.,  0.,  0.],
       [ 0.,  3.,  0.,  0.],
       [ 0.,  0.,  4.,  0.],
       [ 0.,  0.,  0.,  1.]])

However, if we roll time first in the image array, we can't any longer get an
xyz_affine that makes sense in relationship to the voxel data:

>>> img_t0 = rollimg(img, 't')
>>> xyz_affine(img_t0)
Traceback (most recent call last):
    ...
AxesError: First 3 input axes must correspond to X, Y, Z

But we can fix this:

>>> img_t0_affable = as_xyz_image(img_t0)
>>> xyz_affine(img_t0_affable)
array([[ 2.,  0.,  0.,  0.],
       [ 0.,  3.,  0.,  0.],
       [ 0.,  0.,  4.,  0.],
       [ 0.,  0.,  0.,  1.]])

It also works with nibabel images, which can only have xyz_affines:

>>> import nibabel as nib
>>> nimg = nib.Nifti1Image(data, affine)
>>> xyz_affine(nimg)
array([[ 2.,  0.,  0.,  0.],
       [ 0.,  3.,  0.,  0.],
       [ 0.,  0.,  4.,  0.],
       [ 0.,  0.,  0.,  1.]])
"""

import sys

import numpy as np

from ...fixes.nibabel import io_orientation

from ..image.image import Image
from ..reference import spaces as rsp
from ..reference.coordinate_map import AffineTransform


def xyz_affine(img, name2xyz=None):
    """ Return xyz affine from image `img` if possible, or raise error

    Parameters
    ----------
    img : ``Image`` instance or nibabel image
        It has a ``coordmap`` or method ``get_affine``
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.
        Not used for nibabel `img` input.

    Returns
    -------
    xyz_aff : (4,4) array
        voxel to X, Y, Z affine mapping

    Raises
    ------
    SpaceTypeError : if `img` does not have an affine coordinate map
    AxesError : if not all of x, y, z recognized in `img` ``coordmap`` range
    AffineError : if axes dropped from the affine contribute to x, y, z
    coordinates

    Examples
    --------
    >>> from nipy.core.api import vox2mni, Image
    >>> arr = np.arange(24).reshape((2,3,4,1)).astype(float)
    >>> img = Image(arr, vox2mni(np.diag([2,3,4,5,1])))
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='voxels', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> xyz_affine(img)
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  3.,  0.,  0.],
           [ 0.,  0.,  4.,  0.],
           [ 0.,  0.,  0.,  1.]])

    Nibabel images always have xyz affines

    >>> import nibabel as nib
    >>> nimg = nib.Nifti1Image(arr, np.diag([2,3,4,1]))
    >>> xyz_affine(nimg)
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  3.,  0.,  0.],
           [ 0.,  0.,  4.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    try:
        return img.get_affine()
    except AttributeError:
        return rsp.xyz_affine(img.coordmap, name2xyz)


def is_xyz_affable(img, name2xyz=None):
    """ Return True if the image `img` has an xyz affine

    Parameters
    ----------
    img : ``Image`` or nibabel ``SpatialImage``
        If ``Image`` test ``img.coordmap``.  If a nibabel image, return True
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.
        Not used for nibabel `img` input.

    Returns
    -------
    tf : bool
        True if `img` has an xyz affine, False otherwise

    Examples
    --------
    >>> from nipy.core.api import vox2mni, Image, rollimg
    >>> arr = np.arange(24).reshape((2,3,4,1))
    >>> img = Image(arr, vox2mni(np.diag([2,3,4,5,1])))
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='voxels', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> is_xyz_affable(img)
    True
    >>> time0_img = rollimg(img, 't')
    >>> time0_img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='voxels', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.,  0.],
                     [ 0.,  0.,  0.,  4.,  0.],
                     [ 5.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> is_xyz_affable(time0_img)
    False

    Nibabel images always have xyz affines

    >>> import nibabel as nib
    >>> nimg = nib.Nifti1Image(arr, np.diag([2,3,4,1]))
    >>> is_xyz_affable(nimg)
    True
    """
    try:
        xyz_affine(img, name2xyz)
    except rsp.SpaceError:
        return False
    return True


def as_xyz_image(img, name2xyz=None):
    """ Return version of `img` that has a valid xyz affine, or raise error

    Parameters
    ----------
    img : ``Image`` instance or nibabel image
        It has a ``coordmap`` attribute (``Image``) or a ``get_affine`` method
        (nibabel image object)
    name2xyz : None or mapping
        Object such that ``name2xyz[ax_name]`` returns 'x', or 'y' or 'z' or
        raises a KeyError for a str ``ax_name``.  None means use module default.
        Not used for nibabel `img` input.

    Returns
    -------
    reo_img : ``Image`` instance or nibabel image
        Returns image of same type as `img` input. If necessary, `reo_img` has
        its data and coordmap changed to allow it to return an xyz affine.  If
        `img` is already xyz affable we return the input unchanged (``img is
        reo_img``).

    Raises
    ------
    SpaceTypeError : if `img` does not have an affine coordinate map
    AxesError : if not all of x, y, z recognized in `img` ``coordmap`` range
    AffineError : if axes dropped from the affine contribute to x, y, z
    coordinates
    """
    try:
        aff = xyz_affine(img, name2xyz)
    except (rsp.AxesError, rsp.AffineError):
        pass
    else:
        return img
    cmap = img.coordmap
    order = rsp.xyz_order(cmap.function_range, name2xyz)
    # Reorder reference to canonical order
    reo_img = img.reordered_reference(order)
    # Which input axes correspond?
    ornt = io_orientation(reo_img.coordmap.affine)
    current_in_order = ornt[:,0]
    # Set nan to inf to make np.argsort work for old numpy versions
    current_in_order[np.isnan(current_in_order)] = np.inf
    # Do we have the first three axes somewhere?
    if not set((0,1,2)).issubset(current_in_order):
        raise rsp.AxesError("One of x, y or z outputs missing a "
                            "corresponding input axis")
    desired_input_order = np.argsort(current_in_order)
    reo_img = reo_img.reordered_axes(list(desired_input_order))
    try:
        aff = xyz_affine(reo_img, name2xyz)
    except rsp.SpaceError:
        # Python 2.5 / 3 compatibility
        e = sys.exc_info()[1]
        raise e.__class__("Could not reorder so xyz coordinates did not "
                          "depend on the other axis coordinates: " +
                          str(e))
    return reo_img


def make_xyz_image(data, xyz_affine, world, metadata=None):
    """ Create 3D+ image embedded in space named in `world`

    Parameters
    ----------
    data : object
        Object returning array from ``np.asarray(obj)``, and having ``shape``
        attribute.  Should have at least 3 dimensions (``len(shape) >= 3``), and
        these three first 3 dimensions should be spatial
    xyz_affine : (4, 4) array-like or tuple
        if (4, 4) array-like (the usual case), then an affine relating spatial
        dimensions in data (dimensions 0:3) to mm in XYZ space given in `world`.
        If a tuple, then contains two values: the (4, 4) array-like, and a
        sequence of scalings for the dimensions greater than 3.  See examples.
    world : str or XYZSpace or CoordSysMaker or CoordinateSystem
        World 3D space to which affine refers.  See ``spaces.get_world_cs()``
    metadata : None or mapping, optional
        metadata for created image.  Defaults to None, giving empty metadata.

    Returns
    -------
    img : Image
        image containing `data`, with coordmap constructed from `affine` and
        `world`, and with default voxel input coordinates.  If the data has more
        than 3 dimensions, and you didn't specify the added zooms with a tuple
        `xyz_affine` parameter, the coordmap affine gets filled out with extra
        ones on the diagonal to give an (N+1, N+1) affine, with ``N =
        len(data.shape)``

    Examples
    --------
    >>> data = np.arange(24).reshape((2, 3, 4))
    >>> aff = np.diag([4, 5, 6, 1])
    >>> img = make_xyz_image(data, aff, 'mni')
    >>> img
    Image(
      data=array([[[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]],
    <BLANKLINE>
                  [[12, 13, 14, 15],
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]]),
      coordmap=AffineTransform(
                function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxels', coord_dtype=float64),
                function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S'), name='mni', coord_dtype=float64),
                affine=array([[ 4.,  0.,  0.,  0.],
                              [ 0.,  5.,  0.,  0.],
                              [ 0.,  0.,  6.,  0.],
                              [ 0.,  0.,  0.,  1.]])
             ))

    Now make data 4D; we just add 1. to the diagonal for the new dimension

    >>> data4 = data[..., None]
    >>> img = make_xyz_image(data4, aff, 'mni')
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='voxels', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x=L->R', 'mni-y=P->A', 'mni-z=I->S', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 4.,  0.,  0.,  0.,  0.],
                     [ 0.,  5.,  0.,  0.,  0.],
                     [ 0.,  0.,  6.,  0.,  0.],
                     [ 0.,  0.,  0.,  1.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )

    We can pass in a scalar or tuple to specify scaling for the extra dimension

    >>> img = make_xyz_image(data4, (aff, 2.0), 'mni')
    >>> img.coordmap.affine
    array([[ 4.,  0.,  0.,  0.,  0.],
           [ 0.,  5.,  0.,  0.,  0.],
           [ 0.,  0.,  6.,  0.,  0.],
           [ 0.,  0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]])
    >>> data5 = data4[..., None]
    >>> img = make_xyz_image(data5, (aff, (2.0, 3.0)), 'mni')
    >>> img.coordmap.affine
    array([[ 4.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  5.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  6.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  2.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  1.]])
    """
    N = len(data.shape)
    if N < 3:
        raise ValueError('Need data with at least 3 dimensions')
    if type(xyz_affine) is tuple:
        xyz_affine, added_zooms = xyz_affine
        # Could be scalar added zooms
        try:
            len(added_zooms)
        except TypeError:
            added_zooms = (added_zooms,)
        if len(added_zooms) != (N - 3):
            raise ValueError('Wrong number of added zooms')
    else:
        added_zooms = (1,) * (N - 3)
    xyz_affine = np.asarray(xyz_affine)
    if not xyz_affine.shape == (4, 4):
        raise ValueError("Expecting 4 x 4 affine")
    # Make coordinate map
    world_cm = rsp.get_world_cs(world, N)
    voxel_cm = rsp.voxel_csm(N)
    if N > 3:
        affine = np.diag((1, 1, 1) + added_zooms + (1,))
        affine[:3, :3] = xyz_affine[:3, :3]
        affine[:3, -1] = xyz_affine[:3, 3]
    else:
        affine = xyz_affine
    cmap = AffineTransform(voxel_cm, world_cm, affine)
    return Image(data, cmap, metadata)
