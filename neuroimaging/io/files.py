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

from neuroimaging.core.reference.coordinate_map import (reorder_input, 
                                                        reorder_output)
from neuroimaging.core.api import Image
from pyniftiio import PyNiftiIO
from nifti_ref import (coordmap_from_ioimg, coerce_coordmap, get_pixdim, 
                       get_diminfo, standard_order)
                       
def _open(source, coordmap=None, mode="r", dtype=None):
    """Create an `Image` from the given filename

    Parameters
    ----------
    source : filename or a numpy array
    coordmap : `reference.coordinate_map.CoordinateMap`
        The coordinate map for the file
    mode : ``string``
        The mode ot open the file in ('r', 'w', etc)

    Returns
    -------
    image : A new `Image` object created from the filename.

    """

    try:
        if hasattr(source, 'header'):
            hdr = source.header
        else:
            hdr = {}
        ioimg = PyNiftiIO(source, mode, dtype=dtype, header=hdr)
        if coordmap is None:
            coordmap = coordmap_from_ioimg(ioimg.affine, ioimg.header['dim_info'], ioimg.header['pixdim'], ioimg.shape)

        # Build nipy image from array-like object and coordinate map
        img = Image(ioimg, coordmap)
        return img
    except IOError:
        raise IOError, 'Unable to create image from source %s' % str(source)
        
def load(filename, mode='r'):
    """Load an image from the given filename.

    Load an image from the file specified by ``filename``.

    Parameters
    ----------
    filename : string
        Should resolve to a complete filename path.
    mode : Either 'r' or 'r+'

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

    >>> from neuroimaging.io.api import load_image
    >>> from neuroimaging.testing import anatfile
    >>> img = load_image(anatfile)
    >>> img.shape
    (25, 35, 25)

    """

    if mode not in ['r', 'r+']:
        raise ValueError, 'image opening mode must be either "r" or "r+"'
    return _open(filename, mode=mode)

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
    >>> from neuroimaging.core.api import fromarray
    >>> from neuroimaging.io.api import save_image
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
    Cimg = Image(np.transpose(np.asarray(Fimg)), reorder_input(reorder_output(Fimg.coordmap))) # C for 'C-based' 'omnlkji' to 'wvutzyx' coordinate map

    # FIXME: this smells a little bad... to save it using PyNiftiIO
    # the array should be from Cimg
    # but the easiest way to specify the affine in PyNiftiIO seems to be 
    # from Fimg
    # 
    # One possible fix, have the array in PyNiftiIO expecting FORTRAN ordering?
    # BUT, pynifti doesn't let you 'transpose' its array naturally...

    # The image we will ultimately save the data in

    outimage = _open(Cimg, coordmap=Cimg.coordmap, mode='w', dtype=dtype)

    # Cimg (the one that saves the array correctly has a 
    # 'older-nipy' standard affine
    #
    # This seems reasonable for use with pyniftiy because the 
    # ndarray of outimage._data
    # is contiguous or "C-ordered"
    # BUT, Cimg.affine reflects the 'lkji' to 'txyz' coordinate map
    # so, the save through PyNiftIO uses the affine from Fimg
    # and the data from Cimg

    v = np.identity(Cimg.ndim+1)
    v[:Cimg.ndim,:Cimg.ndim] = np.fliplr(np.identity(Cimg.ndim))
    assert np.allclose(np.dot(v, np.dot(Cimg.affine, v)), Fimg.affine)

    # At this point _data is a file-io object (like PyNiftiIO).
    # _data.save delegates the save to pynifti.
    
    # Now that the affine has the proper order,
    # it can be saved to the NIFTI header

    # PyNiftiIO only ever wants a 4x4 affine matrix...
    affine = np.identity(4)
    affine[:3,:3] = Fimg.affine[:3,:3]
    
    # PyNiftiIO save uses the 4x4 affine, pixdim and diminfo
    # to save the file

    outimage._data.save(affine, get_pixdim(Fimg.coordmap, full_length=1),
                        get_diminfo(Fimg.coordmap), filename)
    return outimage
    
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

