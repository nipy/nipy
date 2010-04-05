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

from nipy.core.api import Image, is_image
from nifti_ref import (coordmap_from_affine, coerce_coordmap, 
                       ijk_from_fps, fps_from_ijk)
                       

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
    (33, 41, 25)
    """
    img = formats.load(filename)
    aff = img.get_affine()
    shape = img.get_shape()
    hdr = img.get_header()
    # Get info from NIFTI header, if present, to tell which axes are
    # which.  This is a NIFTI-specific kludge, that might be abstracted
    # out into the image backend in a general way.  Similarly for
    # getting zooms
    try:
        fps = hdr.get_dim_info()
    except (TypeError, AttributeError):
        fps = (None, None, None)
    ijk = ijk_from_fps(fps)
    try:
        zooms = hdr.get_zooms()
    except AttributeError:
        zooms = np.ones(len(shape))
    aff = _match_affine(aff, len(shape), zooms)
    coordmap = coordmap_from_affine(aff, ijk)
    img = Image(img.get_data(), coordmap)
    img.header = hdr
    return img


def _match_affine(aff, ndim, zooms=None):
    ''' Fill or prune affine to given number of dimensions

    >>> aff = np.arange(16).reshape(4,4)
    >>> _match_affine(aff, 3)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> _match_affine(aff, 2)
    array([[ 0.,  1.,  3.],
           [ 4.,  5.,  7.],
           [ 0.,  0.,  1.]])
    >>> _match_affine(aff, 4)
    array([[  0.,   1.,   2.,   0.,   3.],
           [  4.,   5.,   6.,   0.,   7.],
           [  8.,   9.,  10.,   0.,  11.],
           [  0.,   0.,   0.,   1.,   0.],
           [  0.,   0.,   0.,   0.,   1.]])
    >>> aff = np.arange(9).reshape(3,3)
    >>> _match_affine(aff, 2)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    '''
    if aff.shape[0] != aff.shape[1]:
        raise ValueError('Need square affine')
    aff_dim = aff.shape[0] - 1
    if ndim == aff_dim:
        return aff
    aff_diag = np.ones(ndim+1)
    if not zooms is None:
        n = min(len(zooms), ndim)
        aff_diag[:n] = zooms[:n]
    mod_aff = np.diag(aff_diag)
    n = min(ndim, aff_dim)
    # rotations zooms shears
    mod_aff[:n,:n] = aff[:n,:n]
    # translations
    mod_aff[:n,-1] = aff[:n,-1]
    return mod_aff


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
    # Make NIFTI compatible version of image
    newcmap, order = coerce_coordmap(img.coordmap)
    Fimg = Image(np.transpose(np.asarray(img), order), newcmap)
    # Expand or contract affine to 4x4 (3 dimensions)
    rzs = Fimg.affine[:-1,:-1]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    aff = _match_affine(Fimg.affine, 3, zooms)
    ftype = _type_from_filename(filename)
    if ftype.startswith('nifti1'):
        klass = formats.Nifti1Image
    elif ftype == 'analyze':
        klass = formats.Spm2AnalyzeImage
    else:
        raise ValueError('Cannot save file type "%s"' % ftype)
    # make new image
    out_img = klass(data=np.asarray(Fimg),
                    affine=aff,
                    header=original_hdr)
    hdr = out_img.get_header()
    # work out phase, freqency, slice from coordmap names
    ijk = newcmap.input_coords.coord_names
    fps = fps_from_ijk(ijk)
    # put fps into header if possible
    try:
        hdr.set_dim_info(*fps)
    except AttributeError:
        pass
    # Set zooms
    hdr.set_zooms(zooms)
    # save to disk
    out_img.to_filename(filename)
    return Fimg


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


def as_image(image_input):
    ''' Load image from filename or pass through image instance

    Parameters
    ----------
    image_input : str or Image instance
       image or string filename of image.  If a string, load image and
       return.  If an image, pass through without modification

    Returns
    -------
    img : Image or Image-like instance
       Input object if `image_input` seemed to be an image, loaded Image
       object if `image_input` was a string.

    Raises
    ------
    TypeError : if neither string nor image-like passed

    Examples
    --------
    >>> from nipy.testing import anatfile
    >>> from nipy.io.api import load_image
    >>> img = as_image(anatfile)
    >>> img2 = as_image(img)
    >>> img2 is img
    True
    '''
    if is_image(image_input):
        return image_input
    if isinstance(image_input, basestring):
        return load(image_input)
    raise TypeError('Expecting an image-like object or filename string')
    
