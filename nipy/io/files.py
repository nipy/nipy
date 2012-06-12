# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" The io.files module provides basic functions for working with file-based
images in nipy.

* load : load an image from a file
* save : save an image to a file

Examples
--------
See documentation for load and save functions for worked examples.
"""

import os

import numpy as np

import nibabel as nib
from nibabel.spatialimages import HeaderDataError

from ..core.image.image import is_image

from .nifti_ref import (nipy2nifti, nifti2nipy)


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
    Image : image object

    Examples
    --------
    >>> from nipy.io.api import load_image
    >>> from nipy.testing import anatfile
    >>> img = load_image(anatfile)
    >>> img.shape
    (33, 41, 25)
    """
    img = nib.load(filename)
    ni_img = nib.Nifti1Image(img._data, img.get_affine(), img.get_header())
    return nifti2nipy(ni_img)


def save(img, filename, dtype_from='data'):
    """Write the image to a file.

    Parameters
    ----------
    img : An `Image` object
    filename : string
        Should be a valid filename.
    dtype_from : {'data', 'header'} or dtype specifier, optional
        Method of setting dtype to save data to disk. Value of 'data' (default),
        means use data dtype to save.  'header' means use data dtype specified
        in header, if available, otherwise use data dtype.  Can also be any
        valid specifier for a numpy dtype, e.g. 'i4', ``np.float32``.  Not every
        format supports every dtype, so some values of this parameter or data
        dtypes will raise errors.

    Returns
    -------
    image : An `Image` object
        Possibly modified by saving.

    See Also
    --------
    load_image : function for loading images
    Image : image object

    Examples
    --------
    Make a temporary directory to store files

    >>> import os
    >>> from tempfile import mkdtemp
    >>> tmpdir = mkdtemp()

    Make some some files and save them

    >>> import numpy as np
    >>> from nipy.core.api import Image, AffineTransform
    >>> from nipy.io.api import save_image
    >>> data = np.zeros((91,109,91), dtype=np.uint8)
    >>> cmap = AffineTransform('kji', 'zxy', np.eye(4))
    >>> img = Image(data, cmap)
    >>> fname1 = os.path.join(tmpdir, 'img1.nii.gz')
    >>> saved_img1 = save_image(img, fname1)
    >>> saved_img1.shape
    (91, 109, 91)
    >>> fname2 = os.path.join(tmpdir, 'img2.img.gz')
    >>> saved_img2 = save_image(img, fname2)
    >>> saved_img2.shape
    (91, 109, 91)
    >>> fname = 'test.mnc'
    >>> saved_image3 = save_image(img, fname)
    Traceback (most recent call last):
       ...
    ValueError: Sorry, we cannot yet save as format "minc"

    Finally, we clear up our temporary files:

    >>> import shutil
    >>> shutil.rmtree(tmpdir)

    Notes
    -----
    Filetype is determined by the file extension in 'filename'.  Currently the
    following filetypes are supported:

    * Nifti single file : ['.nii', '.nii.gz']
    * Nifti file pair : ['.hdr', '.hdr.gz']
    * SPM Analyze : ['.img', '.img.gz']
    """
    # Try and get nifti
    dt_from_is_str = isinstance(dtype_from, basestring)
    if dt_from_is_str and dtype_from == 'header':
        # All done
        io_dtype = None
    elif dt_from_is_str and dtype_from == 'data':
        io_dtype = img.get_data().dtype
    else:
        io_dtype = np.dtype(dtype_from)
    # make new image
    ni_img = nipy2nifti(img, data_dtype = io_dtype)
    ftype = _type_from_filename(filename)
    if ftype.startswith('nifti1'):
        ni_img.to_filename(filename)
    elif ftype == 'analyze':
        try:
            ana_img = nib.Spm2AnalyzeImage.from_image(ni_img)
        except HeaderDataError:
            raise HeaderDataError('SPM analyze does not support datatype %s' %
                                  ni_img.get_header().get_data_dtype())
        ana_img.to_filename(filename)
    else:
        raise ValueError('Sorry, we cannot yet save as format "%s"' % ftype)
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
