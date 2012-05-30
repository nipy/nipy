# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
An implementation of the dimension info as desribed in:

http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

A version of the same file is in the nibabel repisitory at
``doc/source/external/nifti1.h``.

Background
==========

We (nipystas) make an explicit distinction between

* an input coordinate system of an image (the array == voxel coordinates)
* output coordinate system (usually millimeters in some world)
* the mapping between the two.

The collection of these three is the ``coordmap`` attribute of a NIPY image.

There is no constraint that the number of input and output coordinates should be
the same.

We don't specify the units of our output coordinate system, but assume spatial
units are millimeters and time units are seconds.

NIFTI is mostly less explicit, but more constrained.

NIFTI input coordinate system
-----------------------------

NIFTI files can have up to seven voxel dimensions (7 axes in the input
coordinate system).

The first 3 voxel dimensions of a NIFTI file must be spatial but can be in any
order in relationship to directions in mm space (the output coordinate system)

The 4th voxel dimension is assumed to be time.  In particular, if you have some
other meaning for a non-spatial dimension, the NIFTI standard suggests you set
the length of the 4th dimension to be 1, and use the 5th dimension of the image
instead, and set the NIFTI "intent" fields to state the meaning. If the
``intent`` field is set correctly then it should be possible to set meaningful
input coordinate axis names for dimensions > (0, 1, 2).

There's a wrinkle to the 4th axis is time story; the ``xyxt_units`` field in the
NIFTI header can specify the 4th dimension units as Hz (frequency), PPM
(concentration) or Radians / second.

NIFTI also has a 'dim_info' header attribute that optionally specifies that 0 or
more of the first three voxel axes are 'frequency', 'phase' or 'slice'.  These
terms refer to 2D MRI acquisition encoding, where 'slice's are collected
sequentially, and the two remaining dimensions arose from frequency and phase
encoding.  The ``dim_info`` fields are often not set.  3D acquisitions don't have
a 'slice' dimension.

NIFTI output coordinate system
------------------------------

In the NIFTI specification, the order of the output coordinates (at least the
first 3) are fixed to be what might be called RAS+, that is ('x=L->R', 'y=P->A',
'z=I->S'). This RAS+ output order is not allowed to change and there is no way of
specifying such a change in the nifti header.

The world in which these RAS+ X, Y, Z axes exist can be one of the recognized
spaces, which are: scanner, aligned (to another file's world space), Talairach,
MNI 152 (aligned to the MNI 152 atlas).

By implication, the 4th output dimension is likely to be seconds (given the 4th
input dimension is likley time), but there's a field ``xyzt_units`` (see above)
that can be used to imply the 4th output dimension is actually frequency,
concentration or angular velocity.

NIFTI input / output mapping
----------------------------

NIFTI stores the relationship between the first 3 (spatial) voxel axes and the
RAS+ coordinates in an *XYZ affine*.  This is a homogenous coordinate affine,
hence 4 by 4 for 3 (spatial) dimensions.

NIFTI also stores "pixel dimensions" in a ``pixdim`` field. This can give you
scaling for individual axes.  We ignore the values of ``pixdim`` for the first 3
axes if we have a full ("sform") affine stored in the header, otherwise they
form part of the affine above.  ``pixdim``[3:] provide voxel to output calings
for later axes.  The units for the 4th dimension can come from ``xyzt_units`` as
above.

We take the convention that the output coordinate names are ('x=L->R', 'y=P->A',
'z=I->S','t','u','v','w') unless there is no time axis (see below) in which case
we just omit 't'.  The first 3 axes are also named after the output space
('scanner-x=L->R', 'mni-x=L-R' etc).

The input axes are 'ijktuvw' unless there is no time axis (see below), in which case they are 'ijkuvw' (remember, nifti only allows 7 dimensions,
and one is used up by the time length 1 axis).

Time-like axes
--------------

A time-like axis is an axis that is any of time, Hz, PPM or radians / second.

We recognize time in a NIPY coordinate map by an input or an output axis named
't' or 'time'.  If it's an output axis we work out the corresponding input axis.

A Hz axis can be called 'hz' or 'frequency-hz'.

A PPM axis can be called 'ppm' or 'concentration-ppm'.

A radians / second axis can be called 'rads' or 'radians/s'.

Does this nifti image have a time-like axis?
--------------------------------------------

We take there to be no time axis if there are only three nifti dimensions, or
if:

* the length of the fourth nifti dimension is 1 AND
* There are more than four dimensions AND
* The ``xyzt_units`` field does not indicate time or time-like units.

What we do about all this
=========================

On saving a NIPY image to NIFTI
-------------------------------

First, we need to create a valid XYZ Affine.  We check if this can be done by
checking if there are recognizable X, Y, Z output axes and corresponding input
(voxel) axes.  This requires the input image to be at least 3D. If we find these
requirements, we reorder the image axes to have XYZ output axes and 3 spatial
input axes first, and get the corresponding XYZ affine.

We check if the XYZ output fits with the the NIFTI named spaces of scanner,
aligned, Talairach, MNI.  If not we raise an error.

If the non-spatial dimensions are not orthogonal to each other, raise an error.

If any of the first three input axes are named ('slice', 'freq', 'phase') set
the ``dim_info`` field accordingly.

Set the ``xyzt_units`` field to indicate millimeters and seconds, if there is a
't' axis, otherwise millimeters and (Hz, PPM, rads) if there's are other time-like
axes), otherwise millimeters and zero (unknown).

We look to see if we have a time-like axis in the inputs or the outputs. If we
do, roll that axis to be the 4th axis.  If this axis is actually time, take the
``affine[3, -1]`` and put into the ``toffset`` field.  If there's no time-like
axis, but there are other non-spatial axes, make a length 1 input axis to
indicate this.

Set ``pixdim`` for axes >= 3 using vector length of corresponding affine
columns.

We don't set the intent-related fields for now.

On loading a NIPY image from NIFTI
----------------------------------

Lacking any other information, we take the input coordinate names for
axes 0:7 to be  ('i', 'j', 'k', 't', 'u', 'v', 'w').

If there is a time-like axis, name the input and corresponding output axis for
the type of axis ('t', 'hz', 'ppm', 'rads').

Otherwise remove the 't' axis from both input and output, and squeeze the length
1 dimension from the nifti.

If there's a 't' axis get ``toffset`` and put into affine at position [3, -1].

If ``dim_info`` is set coherently, set input axis names to 'slice', 'freq',
'phase' from ``dim_info``.

Get the output spatial coordinate names from the 'scanner', 'aligned',
'talairach', 'mni' XYZ spaces (see :mod:`nipy.core.reference.spaces`).

We construct the N-D affine by taking the XYZ affine and adding scaling diagonal
elements from ``pixdim``.

Ignore the intent-related fields for now.

"""

import warnings

import numpy as np

from ..core.reference.coordinate_system import CoordinateSystem as CS
from ..core.reference.coordinate_map import AffineTransform as AT, compose
from ..core.reference import spaces as ncrs
from ..core.api import (lps_output_coordnames, ras_output_coordnames)


valid_input_axisnames = tuple('ijktuvw')
valid_output_axisnames = tuple('xyztuvw')
fps = ('frequency', 'phase', 'slice')
valid_spatial_axisnames = valid_input_axisnames[:3] + fps
valid_nonspatial_axisnames = valid_input_axisnames[3:]


def ni_affine_pixdim_from_affine(affine_transform, strict=False):
    """

    Given a square affine_transform, return a new 3-dimensional AffineTransform
    and the pixel dimensions in dimensions greater than 3.

    If strict is True, then an exception is raised if the affine matrix is not
    diagonal with positive entries in dimensions greater than 3.

    If strict is True, then the names of the range coordinates must be
    LPS:('x+LR','y+PA','z+SI') or RAS:('x+RL','y+AP','z+SI'). If strict is
    False, and the names are not either of these, LPS:('x+LR','y+PA','z+SI') are
    used.

    If the names are RAS:('x+RL','y+AA','z+SI'), then the affine is flipped so
    the result is in LPS:('x+LR','y+PA','z+SI').

    NIFTI images have the first 3 dimensions as spatial, and the remaining as
    non-spatial, with the 4th typically being time.

    Parameters
    ----------
    affine_transform : `AffineTransform`

    Returns
    -------
    nifti_transform: `AffineTransform`
       A 3-dimensional or less AffineTransform
    pixdim : ndarray(np.float)
       The pixel dimensions greater than 3.

    Examples
    --------
    >>> from nipy.core.api import CoordinateSystem as CS
    >>> from nipy.core.api import AffineTransform as AT
    >>> outnames = CS(('x+LR','y+PA','z+SI') + ('t',))
    >>> innames = CS(['phase', 'j', 'frequency', 't'])
    >>> af_tr = AT(outnames, innames, np.diag([2,-2,3,3.5,1]))
    >>> print af_tr
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI', 't'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('phase', 'j', 'frequency', 't'), name='', coord_dtype=float64),
       affine=array([[ 2. ,  0. ,  0. ,  0. ,  0. ],
                     [ 0. , -2. ,  0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  3. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ,  3.5,  0. ],
                     [ 0. ,  0. ,  0. ,  0. ,  1. ]])
    )

    >>> af_tr3dorless, p = ni_affine_pixdim_from_affine(af_tr)
    >>> print af_tr3dorless
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.],
                     [ 0., -2.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.],
                     [ 0.,  0.,  0.,  1.]])
    )
    >>> print p
    [ 3.5]
    """
    if ((not isinstance(affine_transform, AT)) or
        (affine_transform.ndims[0] != affine_transform.ndims[1])):
        raise ValueError('affine_transform must be a square AffineTransform' + 
                         ' to save as a NIFTI file')
    ndim = affine_transform.ndims[0]
    ndim3 = min(ndim, 3)
    range_names = affine_transform.function_range.coord_names
    if range_names[:ndim3] not in [lps_output_coordnames[:ndim3],
                                   ras_output_coordnames[:ndim3]]:
        if strict:
            raise ValueError('strict is true and the range is not LPS or RAS')
        warnings.warn('range is not LPS or RAS, assuming LPS')
        range_names = list(range_names)
        range_names[:ndim3] = lps_output_coordnames[:ndim3]
        range_names = tuple(range_names)

    ndim = affine_transform.ndims[0]
    nifti_indim = 'ijk'[:ndim] + 'tuvw'[ndim3:ndim]
    nifti_outdim = range_names[:ndim3] + ('t', 'u', 'v', 'w' )[ndim3:ndim]

    nifti_transform = AT(CS(nifti_indim),
                         CS(nifti_outdim),
                         affine_transform.affine)

    domain_names = affine_transform.function_domain.coord_names[:ndim3]
    nifti_transform = nifti_transform.renamed_domain(dict(zip('ijk'[:ndim3],
                                                         domain_names)))


    # now find the pixdims
    A = nifti_transform.affine[3:,3:]
    if (not np.allclose(np.diag(np.diag(A)), A)
        or not np.all(np.diag(A) > 0)):
        msg = "affine transformation matrix is not diagonal " + \
              " with positive entries on diagonal, some information lost"
        if strict:
            raise ValueError('strict is true and %s' % msg)
        warnings.warn(msg)
    pixdim = np.fabs(np.diag(A)[:-1])

    # find the 4x4 (or smaller)
    A3d = np.identity(ndim3+1)
    A3d[:ndim3,:ndim3] = nifti_transform.affine[:ndim3, :ndim3]
    A3d[:ndim3,-1] = nifti_transform.affine[:ndim3, -1]

    range_names = nifti_transform.function_range.coord_names[:ndim3]
    nifti_3dorless_transform = AT(CS(domain_names),
                                  CS(range_names),
                                  A3d)
    # if RAS, we flip, with a warning
    if range_names[:ndim3] == ras_output_coordnames[:ndim3]:
        signs = [-1,-1,1,1][:(ndim3+1)]
        # silly, but 1d case is handled for consistency
        if signs == [-1,-1]:
            signs = [-1,1]
        ras_to_lps = AT(CS(ras_output_coordnames[:ndim3]),
                        CS(lps_output_coordnames[:ndim3]),
                        np.diag(signs))
        warnings.warn('affine_transform has RAS output_range, flipping to LPS')
        nifti_3dorless_transform = compose(ras_to_lps, nifti_3dorless_transform)
    return nifti_3dorless_transform, pixdim


def get_input_cs(hdr):
    """ Get input (function_domain) coordinate system from `hdr`

    Look at the header `hdr` to see if we have information about the image axis
    names.  So far this is ony true of the nifti header, which can use the
    ``dim_info`` field for this.  If we can't find any information, use the
    default names from 'ijklmnop'

    Parameters
    ----------
    hdr : object
        header object, having at least a ``get_data_shape`` method

    Returns
    -------
    cs : ``CoordinateSystem``
        Input (function_domain) Coordinate system

    Example
    -------
    >>> class C(object):
    ...     def get_data_shape(self):
    ...         return (2,3)
    ...
    >>> hdr = C()
    >>> get_input_cs(hdr)
    CoordinateSystem(coord_names=('i', 'j'), name='voxel', coord_dtype=float64)
    """
    ndim = len(hdr.get_data_shape())
    all_names = list('ijklmno')
    try:
        freq, phase, slice = hdr.get_dim_info()
    except AttributeError:
        pass
    else: # Nifti - maybe we have named axes
        if not freq is None:
            all_names[freq] = 'freq'
        if not phase is None:
            all_names[phase] = 'phase'
        if not slice is None:
            all_names[slice] = 'slice'
    return CS(all_names[:ndim], 'voxel')


_xform2csm = {'scanner': ncrs.scanner_csm,
              'aligned': ncrs.aligned_csm,
              'talairach': ncrs.talairach_csm,
              'mni': ncrs.mni_csm}


def get_output_cs(hdr):
    """ Calculate output (function range) coordinate system from `hdr`

    With our current use of nibabel for image loading, there is always an xyz
    output, because nibabel images always have 4x4 xyz affines.  So, the output
    coordinate system has a least 3 coordinates (those for x, y, z), regardless
    of the array shape implied by `hdr`.  If `hdr` implies a larger array shape
    N (where N>3), then the output coordinate system will be length N.

    Nifti also allows us to specify one of 4 named output spaces (scanner,
    aligned, talairach and mni).

    Parameters
    ----------
    hdr : object
        header object, having at least a ``get_data_shape`` method

    Returns
    -------
    cs : ``CoordinateSystem``
        Input (function_domain) Coordinate system

    Example
    -------
    >>> class C(object):
    ...     def get_data_shape(self):
    ...         return (2,3)
    ...
    >>> hdr = C()
    >>> get_output_cs(hdr)
    CoordinateSystem(coord_names=('unknown-x=L->R', 'unknown-y=P->A', 'unknown-z=I->S'), name='unknown', coord_dtype=float64)
    """
    # Affines from nibabel always have 3 dimensions of output
    ndim = max((len(hdr.get_data_shape()), 3))
    try:
        label = hdr.get_value_label('sform_code')
    except AttributeError: # not nifti
        return ncrs.unknown_csm(ndim)
    csm = _xform2csm.get(label, None)
    if not csm is None:
        return csm(ndim)
    label = hdr.get_value_label('qform_code')
    csm = _xform2csm.get(label, None)
    if not csm is None:
        return csm(ndim)
    return ncrs.unknown_csm(ndim)
