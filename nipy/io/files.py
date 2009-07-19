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

from nipy.core import AffineImage
                       

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
    >>> img.get_data().shape
    (25, 35, 25)
    """
    io_img = formats.load(filename)
    # XXX: Need to worry about axis name 
    img = AffineImage(data=io_img.get_data(), 
                      affine=io_img.get_affine(),
                      world_space=filename,
                      metadata=dict(header=io_img.get_header()))
    return img



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
        original_hdr = img.metadata['header']
    except AttributeError:
        original_hdr = None
    # Make NIFTI compatible version of image
    img = img.as_affine_image() 
    # Extract zoom from affine
    rzs = img.affine[:-1,:-1]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    ftype = _type_from_filename(filename)
    if ftype.startswith('nifti1'):
        klass = formats.Nifti1Image
    elif ftype == 'analyze':
        klass = formats.Spm2AnalyzeImage
    else:
        raise ValueError('Cannot save file type "%s"' % ftype)
    # make new image
    io_img = klass(data=img.get_data(),
                    affine=img.affine,
                    header=original_hdr)
    hdr = io_img.get_header()
    # XXX:
    # Need to work out axis naming if possible.
    #try:
    #    hdr.set_dim_info(*fps)
    #except AttributeError:
    #    pass
    # Set zooms
    hdr.set_zooms(zooms)
    # save to disk
    io_img.to_filespec(filename)
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
