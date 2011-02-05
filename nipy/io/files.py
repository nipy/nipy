# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

import nibabel as nib

from nipy.core.api import Image, is_image
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
    (33, 41, 25)
    """
    img = nib.load(filename)
    aff = img.get_affine()
    shape = img.get_shape()
    hdr = img.get_header()
    # If the header implements it, get a list of names, one per axis,
    # and put this into the coordinate map.  In fact, no image format
    # implements this at the moment, so in practice, the following code
    # is not currently called. 
    axis_renames = {}
    try:
        axis_names = hdr.axis_names
    except AttributeError:
        pass
    else:
        # axis_renames is a dictionary: dict([(int, str)]) that has keys
        # in range(3). The axes of the Image are renamed from 'ijk' using
        # these names
        for i in range(min([len(axis_names), 3])):
            name = axis_names[i]
            if not (name is None or name == ''):
                axis_renames[i] = name
    zooms = hdr.get_zooms()
    # affine_transform is a 3-d transform
    affine_transform3d, affine_transform = \
        affine_transform_from_array(aff, 'ijk', pixdim=zooms[3:])
    img = Image(img.get_data(), affine_transform.renamed_domain(axis_renames))
    img.header = hdr
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
        klass = nib.Nifti1Image
    elif ftype == 'analyze':
        klass = nib.Spm2AnalyzeImage
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
    
