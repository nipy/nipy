# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
An implementation of the dimension info as desribed in:

http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

In particular, it allows one to take a (possibly 4 or higher-dimensional)
AffineTransform instance and return a valid NIFTI 3-dimensional NIFTI
AffineTransform instance.

Axes:
-----

NIFTI files can have up to seven dimensions. We take the convention that
the output coordinate names are ('x+LR','y+PA','z+SI','t','u','v','w')
and the input coordinate names are ('i','j','k','t','u','v','w').

In the NIFTI specification, the order of the output coordinates (at least the
first 3) are fixed to be LPS:('x+LR','y+PA','z+SI') and their order is not
allowed to change. If the output coordinates are RAS:('x+RL','y+AP','z+SI'),
then the function ni_affine_pixdim_from_affine flips them to maintain NIFTI's
standard of LPS:('x+LR','y+PA','z+SI') coordinates.

NIFTI has a 'diminfo' header attribute that optionally specifies that
some of 'i', 'j', 'k' are renamed 'frequency', 'phase' or 'axis'.
"""

import warnings

import numpy as np

from nipy.core.api import CoordinateSystem as CS, AffineTransform as AT

from nipy.core.reference.coordinate_map import (product as mapping_product,
                                                compose)
from nipy.core.api import (lps_output_coordnames, ras_output_coordnames)


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


def affine_transform_from_array(affine, ijk, pixdim):
    """Generate a AffineTransform from an affine transform.

    This is a convenience function to create a AffineTransform from image
    attributes.  It assumes that the first three axes in the image (and
    therefore affine) are spatial (in 'ijk' in input and equal to 'xyz'
    in output), and appends the standard names for further dimensions
    (e.g. 'l' as the 4th in input, 't' as the 4th in output).

    Parameters
    ----------
    affine : array
       affine for affine_transform
    ijk : sequence
       sequence, some permutation of 'ijk', giving spatial axis
       ordering.  These are the spatial input axis names
    pixdim : sequence of floats
       Pixdims for dimensions beyond 3.

    Returns
    -------
    3daffine_transform : ``AffineTransform``
       affine transform corresponding to `affine` and `ijk` domain names
       with LPS range names
    full_transform: ``AffineTransform``
       affine transform corresponding to `affine` and `ijk` domain names
       for first 3 coordinates, diagonal with pixdim values beyond

    Examples
    --------
    >>> af_tr3d, af_tr = affine_transform_from_array(np.diag([2,3,4,1]), 'ijk', [])
    >>> af_tr.function_domain.coord_names
    ('i', 'j', 'k')
    >>> af_tr3d.function_domain.coord_names
    ('i', 'j', 'k')
    >>> af_tr.function_range.coord_names
    ('x+LR', 'y+PA', 'z+SI')
    >>> af_tr3d.function_range.coord_names
    ('x+LR', 'y+PA', 'z+SI')
    >>> af_tr3d, af_tr = affine_transform_from_array(np.diag([2,3,4,1]), 'kij', [3.5])
    >>> af_tr.function_domain.coord_names
    ('k', 'i', 'j', 't')
    >>> af_tr.function_range.coord_names
    ('x+LR', 'y+PA', 'z+SI', 't')
    >>> af_tr3d.function_domain.coord_names
    ('k', 'i', 'j')
    >>> af_tr3d.function_range.coord_names
    ('x+LR', 'y+PA', 'z+SI')
    >>> print af_tr3d
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI'), name='', coord_dtype=float64),
       affine=array([[ 2.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.],
                     [ 0.,  0.,  0.,  1.]])
    )

    >>> print af_tr
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('k', 'i', 'j', 't'), name='product', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x+LR', 'y+PA', 'z+SI', 't'), name='product', coord_dtype=float64),
       affine=array([[ 2. ,  0. ,  0. ,  0. ,  0. ],
                     [ 0. ,  3. ,  0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  4. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ,  3.5,  0. ],
                     [ 0. ,  0. ,  0. ,  0. ,  1. ]])
    )

    FIXME: This is an internal function and should be revisited when
    the AffineTransform is refactored.
    """
    if affine.shape != (4, 4) or len(ijk) != 3:
        raise ValueError('affine must be square, 4x4, ijk of length 3')
    innames = tuple(ijk) + tuple('tuvw'[:len(pixdim)])
    incoords = CS(innames, 'voxel')
    outnames = lps_output_coordnames + tuple('tuvw'[:len(pixdim)])
    outcoords = CS(outnames, 'world')
    transform3d = AT(CS(incoords.coord_names[:3]), 
                     CS(outcoords.coord_names[:3]), affine)
    if pixdim:
        nonspatial = AT.from_params(incoords.coord_names[3:], 
                                    outcoords.coord_names[3:],
                                    np.diag(list(pixdim) + [1]))
        transform_full = mapping_product(transform3d, nonspatial)
    else:
        transform_full = transform3d
    return transform3d, transform_full
