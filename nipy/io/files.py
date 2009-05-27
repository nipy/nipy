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

import numpy as np

import nifti as nf

from nipy.core.reference.coordinate_map import (reorder_input, 
                                                reorder_output)
from nipy.core.api import Image
from nifti_ref import (coordmap_from_ioimg, coerce_coordmap, get_pixdim, 
                       get_diminfo, standard_order)
                       

def load(filename):
    """Load an image from the given filename.

    Load an image from the file specified by ``filename``.

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
    coordmap = coordmap_from_ioimg(aff,
                                   hdr['dim_info'],
                                   hdr['pixdim'],
                                   img.get_shape())
    return Image(img.get_data(), coordmap)


def _match_affine(aff, ndim):
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
    mod_aff = np.eye(ndim+1)
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
    >>> fd, name = mkstemp(suffix='.nii.gz')
    >>> tmpfile = open(name)
    >>> saved_img = save_image(img, tmpfile.name)
    >>> saved_img.shape
    (91, 109, 91)
    >>> tmpfile.close()
    >>> os.unlink(name)

    Notes
    -----
    Filetype is determined by the file extension in 'filename'.  Currently the
    following filetypes are supported:
        Nifti single file : ['.nii', '.nii.gz']
        Nifti file pair : ['.hdr', '.hdr.gz']
        Analyze file pair : ['.img', 'img.gz']
        
    """

    # Reorder the image to 'fortran'-like input and output coordinates
    # before trying to coerce to a NIFTI like image
    rimg = Image(np.transpose(np.asarray(img)), 
                 reorder_input(reorder_output(img.coordmap)))
    Fimg = coerce2nifti(rimg) # F for '0-based Fortran', 'ijklmno' to
                              # 'xyztuvw' coordinate map
    aff = _match_affine(Fimg.affine, 3)
    dim_info = get_diminfo(Fimg.coordmap)
    nifti_img = nf.Nifti1Image(data=np.asarray(Fimg),
                               affine=aff)
    hdr = nifti_img.get_header()
    hdr['dim_info'] = dim_info
    nifti_img.to_filespec(filename)
    return Fimg
    

def coerce2nifti(img, standard=False):
    """
    Coerce an Image into a new Image with a valid NIFTI coordmap
    so that it can be saved.

    If standard is True, the resulting image has 'standard'-ordered
    input_coordinates, i.e. 'ijklmno'[:img.ndim]
    """
    newcmap, order = coerce_coordmap(img.coordmap)
    nimg = Image(np.transpose(np.asarray(img), order), newcmap)
    if standard:
        sorder, scmap = standard_order(nimg.coordmap)
        return Image(np.transpose(np.asarray(nimg), sorder), scmap)
    else:
        return nimg

