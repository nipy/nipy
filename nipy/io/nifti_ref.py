"""
An implementation of the dimension info as desribed in 

http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

In particular, it allows one to check if a `CoordinateMap` instance
can be coerced into a valid NIFTI CoordinateMap instance. For a 
valid NIFTI coordmap, we can then ask which axes correspond to time,
slice, phase and frequency.

Axes:
-----

NIFTI files can have up to seven dimensions. We take the convention that
the output coordinate names are ['x','y','z','t','u','v','w']
and the input coordinate names are ['i','j','k','l','m','n','o'].

In the NIFTI specification, the order of the output coordinates (at
least the first 3) are fixed to be ['x','y','z'] and their order is not
meant to change.  As for input coordinates, the first three can be
reordered, so ['j','k','i','l'] is valid, for instance.

NIFTI has a 'diminfo' header attribute that optionally specifies the
order of the ['i', 'j', 'k'] axes. To use similar terms to those in the
nifti1.h header, 'frequency' corresponds to 'i'; 'phase' to 'j' and
'slice' to 'k'. We use ['i','j','k'] instead because there are images
for which the terms 'phase' and 'frequency' have no proper meaning.

Voxel coordinates:
------------------

NIFTI's voxel convention is what can best be described as 0-based
FORTRAN indexing. For example: suppose we want the x=20-th, y=10-th
pixel of the third slice of an image with 30 64x64 slices. This

>>> from nipy.testing import anatfile
>>> from nipy.io.api import load_image
>>> nifti_ijk = [19,9,2]
>>> fortran_ijk = [20,10,3]
>>> c_kji = [2,9,19]
>>> imgarr = np.asarray(load_image(anatfile))
>>> request1 = imgarr[nifti_ijk[2], nifti_ijk[1], nifti_ijk[0]]
>>> request2 = imgarr[fortran_ijk[2]-1,fortran_ijk[1]-1, fortran_ijk[0]-1]
>>> request3 = imgarr[c_kji[0],c_kji[1],c_kji[2]]
>>> request1 == request2
True
>>> request2 == request3
True

FIXME: (finish this thought.... Are we going to open NIFTI files with
NIFTI input coordinates?)  For this reason, we have to consider whether
we should transpose the memmap from pynifti.
"""
import warnings

import numpy as np

from nipy.core.api import CoordinateSystem as CS, AffineTransform as AT

# Renamings that have to be done: 
#
# coordmap-> affine_transform in nipy.core.image, nipy.io
# lpi-> lps
# affine_transform.affine->affine_transform.matrix ?

from nipy.core.image.lpi_image import lps_output_coordnames, \
   ras_output_coordnames
from nipy.core.reference.coordinate_map import product as mapping_product, \
    compose


valid_input_axisnames = tuple('ijktuvw')
valid_output_axisnames = tuple('xyztuvw')
fps = ('frequency', 'phase', 'slice')
valid_spatial_axisnames = valid_input_axisnames[:3] + fps
valid_nonspatial_axisnames = valid_input_axisnames[3:]

def ni_affine_pixdim_from_affine(affine_transform, strict=False):
    """

    Given a square affine_transform,
    return a new 3-dimensional AffineTransform
    and the pixel dimensions in dimensions 
    greater than 3.

    If strict is True, then an exception is raised if
    the affine matrix is not diagonal with
    positive entries in dimensions 
    greater than 3. 

    If strict is True, then the names of the range coordinates
    must be LPS:%(lps)s or RAS:%(ras)s. If strict is False, and the names
    are not either of these, LPS:%(lps)s are used.

    If the names are RAS:%(ras)s, then the affine is flipped
    so the result is in LPS:%(lps)s

    NIFTI images have the first 3 dimensions as spatial, and the
    remaining as non-spatial, with the 4th typically being time.

    Parameters
    ----------
    affine_transform : `AffineTransform`

    Returns
    -------
    nifti_transform: `AffineTransform`
       A 3-dimensional or less AffineTransform

    pixdim : ndarray(np.float)
       The pixel dimensions greater than 3.

    """ %{'lps':lps_output_coordnames,
          'ras':ras_output_coordnames}

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
            raise ValueError('strict is true and the range is not LPS or RAS, assuming LPS')
        warnings.warn('range is not LPS or RAS, assuming LPS')
        range_names = list(range_names)
        range_names[:ndim3] = lps_output_coordnames[:ndim3]
        range_names = tuple(range_names)

    ndim = affine_transform.ndims[0]
    nifti_indim = 'ijk'[:ndim] + 'tuvw'[ndim3:ndim]
    nifti_outdim = range_names[:ndim3] + \
        ('t', 'u', 'v', 'w' )[ndim3:ndim]

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
    pixdim = np.fabs(np.diag(A)[3:])

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
    >>> cmap = affine_transform_from_array(np.eye(4), 'ijk')
    >>> cmap.function_domain.coord_names
    ('i', 'j', 'k')
    >>> cmap.function_range.coord_names
    ('x', 'y', 'z')
    >>> cmap = affine_transform_from_array(np.eye(5), 'kij')
    >>> cmap.function_domain.coord_names
    ('k', 'i', 'j', 't')
    >>> cmap.function_range.coord_names
    ('x', 'y', 'z', 't')
    
    FIXME: This is an internal function and should be revisited when
    the AffineTransform is refactored.

    JT: This encapsulates a lot of the logic in LPI/RASTransform,
    and returns both transforms.
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

