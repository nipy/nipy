""" Utilities for working with Images and common neuroimaging spaces
"""

import numpy as np

from nibabel.orientations import io_orientation

from ..reference import spaces as rsp


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
    >>> arr = np.arange(24).reshape((2,3,4,1))
    >>> img = Image(arr, vox2mni(np.diag([2,3,4,5,1])))
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x', 'mni-y', 'mni-z', 't'), name='mni', coord_dtype=float64),
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
    array([[2, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 1]])
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
    >>> from nipy.core.api import vox2mni, Image, img_rollaxis
    >>> arr = np.arange(24).reshape((2,3,4,1))
    >>> img = Image(arr, vox2mni(np.diag([2,3,4,5,1])))
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('mni-x', 'mni-y', 'mni-z', 't'), name='mni', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> is_xyz_affable(img)
    True
    >>> time0_img = img_rollaxis(img, 't')
    >>> time0_img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='array', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('t', 'mni-x', 'mni-y', 'mni-z'), name='mni', coord_dtype=float64),
       affine=array([[ 5.,  0.,  0.,  0.,  0.],
                     [ 0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.,  0.],
                     [ 0.,  0.,  0.,  4.,  0.],
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


def as_xyz_affable(img, name2xyz=None):
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
    except rsp.AffineError:
        pass
    else:
        return img
    cmap = img.coordmap
    order = rsp.xyz_order(cmap.function_range, name2xyz)
    # Reorder reference to canonical order
    reo_img = img.reordered_reference(order)
    # Which input axes correspond?
    ornt = io_orientation(reo_img.coordmap.affine)
    desired_input_order = np.argsort(ornt[:,0])
    reo_img = reo_img.reordered_axes(list(desired_input_order))
    try:
        aff = xyz_affine(reo_img, name2xyz)
    except rsp.AffineError:
        raise rsp.AffineError("Could not reorder so xyz coordinates did not "
                              "depend on the other axis coordinates")
    return reo_img
