""" Compatibility functions for older versions of nibabel

Nibabel <= 1.3.0 do not have these attributes:

* header
* affine
* dataobj

The equivalents for these older versions of nibabel are:

* obj.get_header()
* obj.get_affine()
* obj._data

With old nibabel, getting unscaled data used `read_img_data(img,
prefer="unscaled").  Newer nibabel should prefer the `get_unscaled` method on
the image proxy object
"""

import numpy as np

import nibabel as nib


def get_dataobj(img):
    """ Return data object for nibabel image

    Parameters
    ----------
    img : ``SpatialImage`` instance
        Instance of nibabel ``SpatialImage`` class

    Returns
    -------
    dataobj : object
        ``ArrayProxy`` or ndarray object containing data for `img`
    """
    try:
        return img.dataobj
    except AttributeError:
        return img._data


def get_header(img):
    """ Return header from nibabel image

    Parameters
    ----------
    img : ``SpatialImage`` instance
        Instance of nibabel ``SpatialImage`` class

    Returns
    -------
    header : object
        header object from `img`
    """
    try:
        return img.header
    except AttributeError:
        return img.get_header()


def get_affine(img):
    """ Return affine from nibabel image

    Parameters
    ----------
    img : ``SpatialImage`` instance
        Instance of nibabel ``SpatialImage`` class

    Returns
    -------
    affine : object
        affine object from `img`
    """
    try:
        return img.affine
    except AttributeError:
        return img.get_affine()


def get_unscaled_data(img):
    """ Get the data from a nibabel image, maybe without applying scaling

    Parameters
    ----------
    img : ``SpatialImage`` instance
        Instance of nibabel ``SpatialImage`` class

    Returns
    -------
    data : ndarray
        Data as loaded from image, not applying scaling if this can be avoided
    """
    if hasattr(nib.AnalyzeImage.ImageArrayProxy, 'get_unscaled'):
        try:
            return img.dataobj.get_unscaled()
        except AttributeError:
            return np.array(img.dataobj)
    return nib.loadsave.read_img_data(img, prefer='unscaled')
