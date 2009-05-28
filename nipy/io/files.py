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

import nifti as nf

from nipy.core.api import Image
from nifti_ref import (coordmap_from_ioimg, coerce_coordmap, 
                       get_diminfo, standard_order,
                       ijk_from_diminfo)
                       

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
    img = nf.load(filename)
    aff = img.get_affine()
    shape = img.get_shape()
    aff = _match_affine(aff, len(shape))
    hdr = img.get_header()
    # Get byte from NIFTI header, if present, to tell which axes are
    # which.  This is a NIFTI-specific kludge, that might be abstracted
    # out into the image backend in a general way
    try:
        dim_info = hdr['dim_info']
    except (TypeError, KeyError):
        dim_info = 0
    ijk = ijk_from_diminfo(dim_info)
    coordmap = coordmap_from_ioimg(aff,
                                   ijk,
                                   img.get_shape())
    return Image(img.get_data(), coordmap)


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
    if ndim > aff_dim:
        mod_aff[:aff_dim,:aff_dim] = aff[:aff_dim,:aff_dim]
        mod_aff[:aff_dim,-1] = aff[:aff_dim,-1]
    else: # required smaller than given
        mod_aff[:ndim,:ndim] = aff[:ndim,:ndim]
        mod_aff[:ndim,-1] = aff[:ndim,-1]
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

    Notes
    -----
    Filetype is determined by the file extension in 'filename'.  Currently the
    following filetypes are supported:
    
    * Nifti single file : ['.nii', '.nii.gz']
    * Nifti file pair : ['.hdr', '.hdr.gz']
    * Analyze file pair : ['.img', 'img.gz']
    """
    # Make NIFTI compatible version of image
    newcmap, order = coerce_coordmap(img.coordmap)
    Fimg = Image(np.transpose(np.asarray(img), order), newcmap)
    # Expand or contract affine to 4x4 (3 dimensions)
    rzs = Fimg.affine[:-1,:-1]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    aff = _match_affine(Fimg.affine, 3, zooms)
    ftype = _type_from_filename(filename)
    if ftype.startswith('nifti1'):
        dim_info = get_diminfo(Fimg.coordmap)
        out_img = nf.Nifti1Image(data=np.asarray(Fimg), affine=aff)
        hdr = out_img.get_header()
        hdr['dim_info'] = ord(dim_info)
    elif ftype == 'analyze':
        out_img = nf.Spm2AnalyzeImage(data=np.asarray(Fimg), affine=aff)
    else:
        raise ValueError('Cannot save file type "%s"' % ftype)
    out_img.to_filespec(filename)
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
