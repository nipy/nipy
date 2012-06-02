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

If the non-spatial dimensions are not orthogonal to each other, raise a
NiftiError.

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

If ``dim_info`` is set coherently, set input axis names to 'slice', 'freq',
'phase' from ``dim_info``.

If there is a time-like axis, name the input and corresponding output axis for
the type of axis ('t', 'hz', 'ppm', 'rads').

Otherwise remove the 't' axis from both input and output names, and squeeze the
length 1 dimension from the nifti.

If there's a 't' axis get ``toffset`` and put into affine at position [3, -1].

Get the output spatial coordinate names from the 'scanner', 'aligned',
'talairach', 'mni' XYZ spaces (see :mod:`nipy.core.reference.spaces`).

We construct the N-D affine by taking the XYZ affine and adding scaling diagonal
elements from ``pixdim``.

If the space units in Nifti's ``xyxt_units`` are 'micron' or 'meter', scale the
affine by 0.001 and 1000 respectively, but warn.

Ignore the intent-related fields for now.

"""

import sys

import warnings
from copy import copy

import numpy as np

import nibabel as nib
from nibabel.affines import to_matvec, from_matvec

from ..core.reference.coordinate_system import CoordinateSystem as CS
from ..core.reference.coordinate_map import (AffineTransform as AT,
                                             axid2axes,
                                             product as cm_product)
from ..core.reference import spaces as ncrs
from ..core.image.image import Image
from ..core.image.image_spaces import as_xyz_affable


XFORM2SPACE = {'scanner': ncrs.scanner_space,
               'aligned': ncrs.aligned_space,
               'talairach': ncrs.talairach_space,
               'mni': ncrs.mni_space}

TIME_LIKE_AXES = ( # name, matcher, units
    ('t', lambda n : n == 't' or n == 'time', 'sec'),
    ('hz', lambda n : n == 'hz' or n == 'frequency-hz', 'hz'),
    ('ppm', lambda n : n == 'ppm' or n == 'concentration-ppm', 'ppm'),
    ('rads', lambda n : n == 'rads' or n == 'radians/s', 'rads'),
)

# Threshold for near-zero affine values
TINY = 1e-5


class NiftiError(Exception):
    pass


def nipy2nifti(img, strict=None, fix0=False):
    """ Return nifti image from nipy image `img`

    Parameters
    ----------
    img : object
         An object, usually a NIPY ``Image``,  having attributes `coordmap` and
         `shape`
    strict : bool, optional
        Whether to use strict checking of input image for creating nifti
    fix0: bool, optional
        Whether to fix potential 0 column / row in affine. This option only used
        when trying to find time etc axes in the coordmap output names.  In
        order to find matching input names, we need to use the corresponding
        rows and columns in the affine.  Sometimes time, in particular, has 0
        scaling, and thus all 0 in the corresponding row / column.  In that case
        it's hard to work out which input corresponds. If `fix0` is True, and
        there is only one all zero (matrix part of the) affine row, and only one
        all zero (matrix part of the) affine column, fix scaling for that
        combination to zero, assuming this a zero scaling for time.

    Returns
    -------
    ni_img : ``nibabel.Nifti1Image``
        Nifti image

    Notes
    -----
    First, we need to create a valid XYZ Affine.  We check if this can be done
    by checking if there are recognizable X, Y, Z output axes and corresponding
    input (voxel) axes.  This requires the input image to be at least 3D. If we
    find these requirements, we reorder the image axes to have XYZ output axes
    and 3 spatial input axes first, and get the corresponding XYZ affine.

    If the non-spatial dimensions are not orthogonal to each other, raise an
    error.

    We check if the XYZ output fits with the the NIFTI named spaces of scanner,
    aligned, Talairach, MNI.  If not we raise an error.

    If any of the first three input axes are named ('slice', 'freq', 'phase')
    set the ``dim_info`` field accordingly.

    Set the ``xyzt_units`` field to indicate millimeters and seconds, if there
    is a 't' axis, otherwise millimeters and 0 (unknown).

    We look to see if we have an output axis named 't'. If we do, roll that axis
    to be the 4th axis. Take the ``affine[3, -1]`` and put into the ``toffset``
    field.  If there's no 't' axis, but there are other non-spatial axes, make a
    length 1 input axis to indicate this.

    If there is an axis named any of frequency-hz', 'concentration-ppm' or
    'radians/s' and there is no 't' axis, move the axis to the 4th position and
    set ``xyz_units``.

    Set ``pixdim`` for axes >= 3 using vector length of corresponding affine
    columns.

    We don't set the intent-related fields for now.
    """
    strict_none = strict is None
    if strict_none:
        warnings.warn('Default `strict` currently False; this will change to '
                      'True in a future version of nipy',
                      FutureWarning,
                      stacklevel = 2)
        strict = False
    known_names = ncrs.known_names
    if not strict: # add simple 'xyz' to acceptable spatial names
        known_names = copy(known_names) # copy module global dict
        for c in 'xyz':
            known_names[c] = c
    try:
        img = as_xyz_affable(img, known_names)
    except (ncrs.AxesError, ncrs.AffineError):
        # Python 2.5 / 3 compatibility
        e = sys.exc_info()[1]
        raise NiftiError('Image cannot be reordered to XYZ because: "%s"'
                         % e)
    coordmap = img.coordmap
    # Get useful information from old header
    in_hdr = img.metadata.get('header', None)
    hdr = nib.Nifti1Header.from_header(in_hdr)
    # Default behavior is to take datatype from old header, unless there was no
    # header, in which case we try to use the data dtype.  If that fails, we
    # fall back to np.float32
    data = None
    if in_hdr is None:
        data = img.get_data()
        try:
            hdr.set_data_dtype(data.dtype)
        except nib.HeaderDataError:
            hdr.set_data_dtype(np.float32)
    # Remaining axes orthogonal?
    rzs, trans = to_matvec(coordmap.affine)
    if (not np.allclose(rzs[3:, :3], 0) or
        not np.allclose(rzs[:3, 3:], 0)):
        raise NiftiError('Non space axes not orthogonal to space')
    # And to each other?
    nsp_affine = rzs[3:,3:]
    nsp_nzs = np.abs(nsp_affine) > TINY
    n_in_col = np.sum(nsp_nzs, axis=0)
    n_in_row = np.sum(nsp_nzs, axis=1)
    if np.any(n_in_col > 1) or np.any(n_in_row > 1):
        raise NiftiError('Non space axes not orthogonal to each other')
    # Affine seems OK, check for space
    xyz_affine = ncrs.xyz_affine(coordmap, known_names)
    spatial_output_names = coordmap.function_range.coord_names[:3]
    out_space = CS(spatial_output_names)
    for name, space in XFORM2SPACE.items():
        if out_space in space:
            hdr.set_sform(xyz_affine, name)
            hdr.set_qform(xyz_affine, name)
            break
    else:
        if not strict and spatial_output_names == ('x', 'y', 'z'):
            warnings.warn('Default `strict` currently False; '
                          'this will change to True in a future version of '
                          'nipy; output names of "x", "y", "z" will raise '
                          'an error.  Please use canonical output names from '
                          'nipy.core.reference.spaces',
                          FutureWarning,
                          stacklevel = 2)
            hdr.set_sform(xyz_affine, 'scanner')
            hdr.set_qform(xyz_affine, 'scanner')
        else:
            raise NiftiError('Image world not a Nifti world')
    # Set dim_info
    # Use list() to get .index method for python < 2.6
    input_names = list(coordmap.function_domain.coord_names)
    spatial_names = input_names[:3]
    dim_infos = []
    for fps in 'freq', 'phase', 'slice':
        dim_infos.append(
            spatial_names.index(fps) if fps in spatial_names else None)
    hdr.set_dim_info(*dim_infos)
    # Set units without knowing time
    hdr.set_xyzt_units(xyz='mm')
    # Done if we only have 3 input dimensions
    non_space_inames = input_names[3:]
    non_space_onames = coordmap.function_range.coord_names[3:]
    n_ns = len(non_space_inames)
    if n_ns == 0: # No non-spatial dimensions
        return nib.Nifti1Image(img.get_data(), xyz_affine, hdr)
    # Go now to data, pixdims
    if data is None:
        data = img.get_data()
    rzs, trans = to_matvec(img.coordmap.affine)
    ns_pixdims = np.sqrt(np.sum(rzs[3:, 3:] ** 2, axis=0))
    # Look for time and time-related axes in input and then maybe output names
    out_no = None
    for name, matcher, units in TIME_LIKE_AXES:
        for in_ns_no, in_ax_name in enumerate(non_space_inames):
            if matcher(in_ax_name):
                in_no = in_ns_no + 3
                break
        else: # This axis not found inputs, look in outputs
            for out_ns_no, out_ax_name in enumerate(non_space_onames):
                if matcher(out_ax_name):
                    break
            else: # Go check for the next time-like
                continue
            # Find matching input axis
            in_no, out_no = axid2axes(coordmap, out_ax_name)
            if in_no is None: # No matching input, keep trying
                continue
            in_ns_no = in_no - 3
        # xyzt_units
        hdr.set_xyzt_units(xyz='mm', t=units)
        # If this is time, set toffset
        if name == 't':
            # Which output axis corresponds?
            if out_no is None:
                _, out_no = axid2axes(coordmap, in_no)
            if out_no is None:
                raise NiftiError('Time input and output do not match')
            hdr['toffset'] = trans[out_no]
        # Make sure this time-like axis is first non-space axis
        if in_ns_no != 0:
            data = np.rollaxis(data, 3 + in_ns_no, 3)
            order = range(n_ns)
            order.pop(in_ns_no)
            order.insert(0, in_ns_no)
            ns_pixdims = [ns_pixdims[i] for i in order]
        break # once we've found a time-like, stop
    else: # no time-like axis
        # add new 1-length axis
        data = img.get_data()[:, :, :, None, ...]
    hdr['pixdim'][4:(4 + n_ns)] = ns_pixdims
    return nib.Nifti1Image(data, xyz_affine, hdr)


TIME_LIKE_UNITS = dict(
    sec = dict(name='t', scaling = 1),
    msec = dict(name='t', scaling = 1 / 1000.),
    usec = dict(name='t', scaling = 1 / 1000000.),
    hz = dict(name='hz', scaling = 1),
    ppm = dict(name='ppm', scaling = 1),
    rads = dict(name='rads', scaling = 1))


def nifti2nipy(ni_img):
    """ Return NIPY image from nifti image `ni_image`

    Parameters
    ----------
    ni_img : nibabel.Nifti1Image
        Nifti image

    Returns
    -------
    img : :class:`Image`
        nipy image

    Notes
    -----
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
    hdr = ni_img.get_header()
    affine = ni_img.get_affine()
    data = ni_img.get_data()
    shape = list(ni_img.shape)
    ndim = len(shape)
    # For now we only warn if intent is set to an unexpected value
    intent, _, _ = hdr.get_intent()
    if intent != 'none':
        warnings.warn('Ignoring intent field meaning "%s"' % intent,
                        UserWarning)
    # Which space?
    world_label = hdr.get_value_label('sform_code')
    if world_label == 'unknown':
        world_label = hdr.get_value_label('qform_code')
    world_space = XFORM2SPACE.get(world_label, ncrs.unknown_space)
    # Promote 1 and 2D
    if ndim <= 3:
        if ndim == 1:
            data = data[:, None, None]
        elif ndim == 2:
            data = data[:, :, None]
        ndim = 3
    # Get information from dim_info
    input_names3 = list('ijk')
    freq, phase, slice = hdr.get_dim_info()
    if not freq is None:
        input_names3[freq] = 'freq'
    if not phase is None:
        input_names3[phase] = 'phase'
    if not slice is None:
        input_names3[slice] = 'slice'
    # Add to mm scaling, with warning
    space_units, time_like_units = hdr.get_xyzt_units()
    if space_units in ('micron', 'meter'):
        warnings.warn('"%s" space scaling in Nifti ``xyt_units field; '
                      'applying scaling to affine, but this may not be what '
                      'you want' % space_units, UserWarning)
        if space_units == 'micron':
            affine[:3] /= 1000.
        elif space_units == 'meter':
            affine[:3] *= 1000.
    input_cs3 = CS(input_names3, name='voxels')
    output_cs3 = world_space.to_coordsys_maker()(3)
    cmap3 = AT(input_cs3, output_cs3, affine)
    if ndim == 3:
        # Warn for ignoring intent
        return Image(data, cmap3, {'header': hdr})
    space_units, time_like_units = hdr.get_xyzt_units()
    units_info = TIME_LIKE_UNITS.get(time_like_units, None)
    n_ns = ndim - 3
    ns_zooms = list(hdr.get_zooms()[3:])
    ns_trans = [0] * n_ns
    # Have we got a time axis?
    if (shape[3] == 1 and ndim > 4 and units_info is None):
        # Squeeze length 1 no-time axis
        shape.pop(3)
        ns_zooms.pop(0)
        ns_trans.pop(0)
        data = data.reshape(shape)
        ndim -= 1
        time_name = None
    else: # have time-like
        if units_info is None:
            units_info = TIME_LIKE_UNITS['sec']
        time_name = units_info['name']
        if units_info['scaling'] != 1:
            ns_zooms[0] *= units_info['scaling']
        if time_name == 't':
            # Get time offset
            ns_trans[0] = hdr['toffset']
    ns_names = tuple('uvw')
    if not time_name is None:
        ns_names = (time_name,) + ns_names
    output_cs = CS(ns_names[:n_ns])
    input_cs = CS(ns_names[:n_ns])
    aff = from_matvec(np.diag(ns_zooms), ns_trans)
    ns_cmap = AT(input_cs, output_cs, aff)
    cmap = cm_product(cmap3, ns_cmap,
                      input_name=cmap3.function_domain.name,
                      output_name=cmap3.function_range.name)
    return Image(data, cmap, {'header': hdr})
