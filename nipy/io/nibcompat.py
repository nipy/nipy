""" Compatibility functions for older versions of nibabel

Nibabel <= 1.3.0 do not have these attributes:

* header
* affine
* dataobj

The equivalents for these older versions of nibabel are:

* obj.get_header()
* obj.get_affine()
* obj._data
"""

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
