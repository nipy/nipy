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
we should transpose the memmap from nifti input.
"""
import warnings

import numpy as np

from nipy.core.api import CoordinateSystem, Affine

# (i,j,k) = ('phase', 'frequency', 'slice')
valid_input_axisnames = tuple('ijklmno')
valid_output_axisnames = tuple('xyztuvw')


def iscoerceable(coordmap):
    """
    Determine if a given CoordinateMap instance can be used as 
    a valid coordmap for a NIFTI image, so that an Image can be saved.
    
    This may raise various warnings about the coordmap.

    Parameters
    ----------
    coordmap : ``CoordinateMap``

    Returns
    -------
    tf : bool
       True if `coordmap` is coerceable to NIFTI

    Examples
    --------
    >>> from nipy.core.api import CoordinateSystem as CS, Affine
    >>> cmap = Affine(np.eye(5), CS('kjil'), CS('xyzt'))
    >>> iscoerceable(cmap)
    True
    >>> cmap = Affine(np.eye(5), CS('lijk'), CS('xyzt'))
    >>> iscoerceable(cmap)
    True
    >>> cmap = Affine(np.eye(5), CS('ijkq'), CS('xyzt'))
    >>> iscoerceable(cmap)
    False
    """
    try: 
        coerce_coordmap(coordmap)
    except ValueError:
        return False
    return True

def coerce_coordmap(coordmap):
    """
    If necessary, reorder CoordinateMap as valid for a NIFTI image

    NIFTI images have the first 3 dimensions as spatial, and the
    remaining as non-spatial, with the 4th typically being time.

    If the input coordinates must be reordered, then the order defaults
    to the standard NIFTI order ['i','j','k','l','m','n','o'].

    Parameters
    ----------
    coordmap : `CoordinateMap`

    Returns
    -------
    newcmap: `CoordinateMap`
       a new CoordinateMap that can be used with a (possibly transposed
       array) in a proper Image.
    transp_order : list
       a list that should be used to transpose data matching this
       coordmap to allow it to be saved as NIFTI

    Notes
    -----
    If the input coordinates do not have the proper order, the image
    would have to be transposed to be saved. The i,j,k can be in any
    order in the first three slots, but the remaining ones should be in
    order because there is no NIFTI header attribute that can tell us
    anything about this order. Also, the NIFTI header says that the
    phase, freq, slice values all have to be less than 3 in position.
    """
    if not hasattr(coordmap, 'affine'):
        raise ValueError, 'coordmap must be affine to save as a NIFTI file'
    affine = coordmap.affine
    if affine.shape[0] != affine.shape[1]:
        raise ValueError, 'affine must be square to save as a NIFTI file'
    ndim = affine.shape[0] - 1
    # Verify input coordinates are a valid set (independent of order)
    innames = coordmap.input_coords.coord_names
    vinput = valid_input_axisnames[:ndim]
    if set(vinput) != set(innames):
        raise ValueError('input coordinate axisnames of a %d-dimensional'
                         'Image must come from %s' % (ndim, vinput))
    # Verify output coordinates are a valid set (independent of order)
    voutput = valid_output_axisnames[:ndim]
    outnames = coordmap.output_coords.coord_names
    if set(voutput) != set(outnames):
        raise ValueError('output coordinate axisnames of a %d-dimensional'
                         'Image must come from %s' % (ndim, voutput))
    # if the input coordinates do not have the proper order, the image
    # would have to be transposed to be saved
    reinput = False
    # Check if the first 3 input coordinates need to be reordered
    if innames != vinput:
        ndimm = min(ndim, 3)
        # Check if first 3 coords are valid input nifti coords set('ijk')
        if set(innames[:ndimm]) != set(vinput[:ndimm]):
            warnings.warn('an Image with this coordmap has to be '
                          'transposed to be saved because the first '
                          '%d input axes are not from %s' %
                          (ndimm, set(vinput[:ndimm])))
            reinput = True
        # Check if subsequent coords match the correct nifti order ('lmno')
        if innames[ndimm:] != vinput[ndimm:]:
            warnings.warn('an Image with this coordmap has to be '
                          'transposed because the last %d axes are '
                          'not in the NIFTI order' % (ndim-3,))
            reinput = True
    # if the output coordinates are not in the NIFTI order, they will
    # have to be put in NIFTI order, affecting the affine matrix
    reoutput = False
    if outnames != voutput:
        warnings.warn('The order of the output coordinates is not the '
                      'NIFTI order, this will change the affine '
                      'transformation by reordering the output coordinates.')
        reoutput = True
    # Create the appropriate reorderings, if necessary
    inperm = np.identity(ndim+1)
    if reinput:
        inperm[:ndim,:ndim] = np.array([[int(vinput[i] == innames[j]) 
                                         for j in range(ndim)] 
                                        for i in range(ndim)])
    intrans = tuple(np.dot(inperm, range(ndim+1)).astype(np.int))[:-1]
    outperm = np.identity(ndim+1)
    if reoutput:
        outperm[:ndim,:ndim] = np.array([[int(voutput[i] == outnames[j]) 
                                          for j in range(ndim)] 
                                         for i in range(ndim)])
    outtrans = tuple(np.dot(outperm, range(ndim+1)).astype(np.int))[:-1]
    # Create the new affine
    A = np.dot(outperm, np.dot(affine, inperm))
    # If the affine beyond the 3 coordinate is not diagonal
    # some information will be lost saving to NIFTI
    if not np.allclose(np.diag(np.diag(A))[3:,3:], A[3:,3:]):
        warnings.warn("the affine is not diagonal in the "
                      "non 'ijk','xyz' coordinates, information "
                      "will be lost in saving to NIFTI")
    # Create new coordinate systems
    if not np.allclose(inperm, np.identity(ndim+1)):
        inname = coordmap.input_coords.name + '-reordered'
    else:
        inname = coordmap.input_coords.name
    if not np.allclose(outperm, np.identity(ndim+1)):
        outname = coordmap.output_coords.name + '-reordered'
    else:
        outname = coordmap.output_coords.name
    coords = coordmap.input_coords.coord_names
    newincoords = CoordinateSystem([coords[i] for i in intrans], inname)
    coords = coordmap.output_coords.coord_names
    newoutcoords = CoordinateSystem([coords[i] for i in outtrans], outname)
    return Affine(A, newincoords, newoutcoords), intrans


def coordmap_from_affine(affine, ijk):
    """Generate a CoordinateMap from an affine transform.

    This is a convenience function to create a CoordinateMap from image
    attributes.  It assumes that the first three axes in the image (and
    therefore affine) are spatial (in 'ijk' in input and equal to 'xyz'
    in output), and appends the standard names for further dimensions
    (e.g. 'l' as the 4th in input, 't' as the 4th in output).

    Parameters
    ----------
    affine : array
       affine for coordmap
    ijk : sequence
       sequence, some permutation of 'ijk', giving spatial axis
       ordering.  These are the spatial input axis names

    Returns
    -------
    coordmap : ``CoordinateMap``
       coordinate map corresponding to `affine` and `ijk`

    Examples
    --------
    >>> cmap = coordmap_from_affine(np.eye(4), 'ijk')
    >>> cmap.input_coords.coord_names
    ('i', 'j', 'k')
    >>> cmap.output_coords.coord_names
    ('x', 'y', 'z')
    >>> cmap = coordmap_from_affine(np.eye(5), 'kij')
    >>> cmap.input_coords.coord_names
    ('k', 'i', 'j', 'l')
    >>> cmap.output_coords.coord_names
    ('x', 'y', 'z', 't')
    
    FIXME: This is an internal function and should be revisited when
    the CoordinateMap is refactored.
    """
    if len(ijk) != 3:
        raise ValueError('ijk input should be length 3')
    ndim_out = affine.shape[0] - 1
    ndim_in = affine.shape[1] - 1
    if ndim_in < 3:
        raise ValueError('Input number of dimensions should >=3')
    innames = tuple(ijk) + valid_input_axisnames[3:ndim_in]
    incoords = CoordinateSystem(innames, 'input')
    outnames = valid_output_axisnames[:ndim_out]
    outcoords = CoordinateSystem(outnames, 'output')
    return Affine(affine, incoords, outcoords)


def ijk_from_fps(fps):
    ''' Get names 'ijk' from freqency, phase, slice axis indices

    'frequency' corresponds to 'i'; 'phase' to 'j' and 'slice' to 'k'

    Parameters
    ----------
    fps : sequence
       sequence of ints in range(0,3), or None, specifying frequency,
       phase and slice axis respectively.  Integers outside this range
       raise an error.

    Returns
    -------
    ijk : string
       string giving names (characters), some permutation of 'ijk'

    Examples
    --------
    >>> ijk_from_fps((None,None,None))
    'ijk'
    >>> ijk_from_fps((None,None,0))
    'kij'
    >>> ijk_from_fps((2,None,0))
    'kji'
    '''
    remaining = []
    ijk = [' '] * 3
    for name, pos in zip('ijk', fps):
        if pos is None:
            remaining.append(name)
        else:
            ijk[pos] = name
    for pos in range(len(ijk)):
        if ijk[pos] == ' ':
            ijk[pos] = remaining.pop(0)
    return ''.join(ijk)


def fps_from_ijk(ijk):
    ''' Get axis indices for frequency, phase, slice from ijk string

    'frequency' corresponds to 'i'; 'phase' to 'j' and 'slice' to 'k'

    Parameters
    ----------
    ijk : sequence
       sequence of names (characters), usually some permutation of 'ijk'.
    Returns
    -------
    fps : sequence
       sequence of {ints in range(0,3), or None}, specifying frequency,
       phase and slice axis respectively.  If any of 'i','j' or 'k' are
       missing from `ijk` input, return None for corresponding phase
       frequency slice values.

    Examples
    --------
    >>> fps_from_ijk('ijk')
    (0, 1, 2)
    >>> fps_from_ijk('kij')
    (1, 2, 0)
    >>> fps_from_ijk('qrs')
    (None, None, None)
    >>> fps_from_ijk('qrsi')
    (3, None, None)
    '''
    ijk = list(ijk)
    fps = []
    for name in 'ijk':
        try:
            ind = ijk.index(name)
        except ValueError:
            fps.append(None)
        else:
            fps.append(ind)
    return tuple(fps)

