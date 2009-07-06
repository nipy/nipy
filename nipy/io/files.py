"""The image module provides basic functions for working with images in nipy.
Functions are provided to load, save and create image objects, along with
iterators to easily slice through volumes.

    load : load an image from a file

    save : save an image to a file

    fromarray : create an image from a numpy array

Examples
--------
See documentation for load and save functions for 'working' examples.

"""

import os

import numpy as np

import nipy.io.imageformats as formats

from nipy.core.api import Image, compose, AffineTransform
from nifti_ref import (ni_affine_pixdim_from_affine, affine_transform_from_array)
                       
                       

def load(filename):
    """Load an image from the given filename.

    Parameters
    ----------
    filename : string
        Should resolve to a complete filename path.

    Returns
    -------
    image : An `Image` object
        If successful, a new `Image` object is returned.

    See Also
    --------
    save_image : function for saving images
    fromarray : function for creating images from numpy arrays

    Examples
    --------

    >>> from nipy.io.api import load_image
    >>> from nipy.testing import anatfile
    >>> img = load_image(anatfile)
    >>> img.shape
    (25, 35, 25)
    """
    img = formats.load(filename)
    aff = img.get_affine()
    shape = img.get_shape()
    hdr = img.get_header()

    # Get info from NIFTI header, if present, to tell which axes are
    # which.  This is a NIFTI-specific kludge, that might be abstracted
    # out into the image backend in a general way.  Similarly for
    # getting zooms

    # axis_renames is a dictionary: dict([(int, str)])
    # that has keys in range(3)
    # the axes of the Image are renamed from 'ijk'
    # using these names

    try:
        axis_renames = hdr.get_axis_renames()
    except (TypeError, AttributeError):
        axis_renames = {}

    try:
        zooms = hdr.get_zooms()
    except AttributeError:
        zooms = np.ones(len(shape))

    # affine_transform is a 3-d transform

    affine_transform3d, affine_transform = \
        affine_transform_from_array(aff, 'ijk', pixdim=zooms[3:])
    img = Image(img.get_data(), affine_transform.renamed_domain(axis_renames))
    img.header = hdr
    return img


# No longer needed

# def _match_affine(aff, ndim, zooms=None):
#     ''' Fill or prune affine to given number of dimensions

#     XXX Zooms do what here?

#     >>> aff = np.arange(16).reshape(4,4)
#     >>> _match_affine(aff, 3)
#     array([[ 0,  1,  2,  3],
#            [ 4,  5,  6,  7],
#            [ 8,  9, 10, 11],
#            [12, 13, 14, 15]])
#     >>> _match_affine(aff, 2)
#     array([[ 0.,  1.,  3.],
#            [ 4.,  5.,  7.],
#            [ 0.,  0.,  1.]])
#     >>> _match_affine(aff, 4)
#     array([[  0.,   1.,   2.,   0.,   3.],
#            [  4.,   5.,   6.,   0.,   7.],
#            [  8.,   9.,  10.,   0.,  11.],
#            [  0.,   0.,   0.,   1.,   0.],
#            [  0.,   0.,   0.,   0.,   1.]])
#     >>> aff = np.arange(9).reshape(3,3)
#     >>> _match_affine(aff, 2)
#     array([[0, 1, 2],
#            [3, 4, 5],
#            [6, 7, 8]])
#     '''
#     if aff.shape[0] != aff.shape[1]:
#         raise ValueError('Need square affine')
#     aff_dim = aff.shape[0] - 1
#     if ndim == aff_dim:
#         return aff
#     aff_diag = np.ones(ndim+1)
#     if not zooms is None:
#         n = min(len(zooms), ndim)
#         aff_diag[:n] = zooms[:n]
#     mod_aff = np.diag(aff_diag)
#     n = min(ndim, aff_dim)
#     # rotations zooms shears
#     mod_aff[:n,:n] = aff[:n,:n]
#     # translations
#     mod_aff[:n,-1] = aff[:n,-1]
#     return mod_aff


def save(img, filename, dtype=None):
    """Write the image to a file.

    Parameters
    ----------
    img : An `Image` object
    filename : string
        Should be a valid filename.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load_image : function for loading images
    fromarray : function for creating images from numpy arrays

    Examples
    --------

    >>> import os
    >>> import numpy as np
    >>> from tempfile import mkstemp
    >>> from nipy.core.api import fromarray
    >>> from nipy.io.api import save_image
    >>> data = np.zeros((91,109,91), dtype=np.uint8)
    >>> img = fromarray(data, 'kji', 'zxy')
    >>> fd, fname = mkstemp(suffix='.nii.gz')
    >>> saved_img = save_image(img, fname)
    >>> saved_img.shape
    (91, 109, 91)
    >>> os.unlink(fname)
    >>> fd, fname = mkstemp(suffix='.img.gz')
    >>> saved_img = save_image(img, fname)
    >>> saved_img.shape
    (91, 109, 91)
    >>> os.unlink(fname)
    >>> fname = 'test.mnc'
    >>> saved_image = save_image(img, fname)
    Traceback (most recent call last):
       ...
    ValueError: Cannot save file type "minc"
    
    Notes
    -----
    Filetype is determined by the file extension in 'filename'.  Currently the
    following filetypes are supported:
    
    * Nifti single file : ['.nii', '.nii.gz']
    * Nifti file pair : ['.hdr', '.hdr.gz']
    * Analyze file pair : ['.img', 'img.gz']
    """
    # Get header from image
    try:
        original_hdr = img.header
    except AttributeError:
        original_hdr = None
    # Make NIFTI compatible affine_transform
    affine_3dorless_transform, pixdim = ni_affine_pixdim_from_affine(img.coordmap)

#   what are we going to do with pixdim?
#   LPIImage will all have pixdim[3:] == 1...

    aff = affine_3dorless_transform.affine 

    # rzs = Fimg.affine[:3,:], JT for Matthew, I changed this below is this correct?
    rzs = img.coordmap.affine[:-1,:-1] 
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))

    ftype = _type_from_filename(filename)
    if ftype.startswith('nifti1'):
        klass = formats.Nifti1Image
    elif ftype == 'analyze':
        klass = formats.Spm2AnalyzeImage
    else:
        raise ValueError('Cannot save file type "%s"' % ftype)
    # make new image
    out_img = klass(data=img.get_data(),
                    affine=aff,
                    header=original_hdr)
    hdr = out_img.get_header()
    # work out phase, freqency, slice from coordmap names
    axisnames = affine_3dorless_transform.function_domain.coord_names

    # let the hdr do what it wants from the axisnames
    try:
        hdr.set_dim_info_from_names(axisnames)
    except AttributeError:
        pass
    # Set zooms
    hdr.set_zooms(zooms)
    # save to disk
    out_img.to_filespec(filename)
    return img

def _type_from_filename(filename):
    ''' Return image type determined from filename
    
    Filetype is determined by the file extension in 'filename'.
    Currently the following filetypes are supported:
    
    * Nifti single file : ['.nii', '.nii.gz']
    * Nifti file pair : ['.hdr', '.hdr.gz']
    * Analyze file pair : ['.img', '.img.gz']

    >>> _type_from_filename('test.nii')
    'nifti1single'
    >>> _type_from_filename('test')
    'nifti1single'
    >>> _type_from_filename('test.hdr')
    'nifti1pair'
    >>> _type_from_filename('test.hdr.gz')
    'nifti1pair'
    >>> _type_from_filename('test.img.gz')
    'analyze'
    >>> _type_from_filename('test.mnc')
    'minc'
    '''
    if filename.endswith('.gz'):
        filename = filename[:-3]
    elif filename.endswith('.bz2'):
        filename = filename[:-4]
    _, ext = os.path.splitext(filename)
    if ext in ('', '.nii'):
        return 'nifti1single'
    if ext == '.hdr':
        return 'nifti1pair'
    if ext == '.img':
        return 'analyze'
    if ext == '.mnc':
        return 'minc'
    raise ValueError('Strange file extension "%s"' % ext)
