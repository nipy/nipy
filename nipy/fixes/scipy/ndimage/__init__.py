""" Patches for scipy.ndimage

Patched affine_transform to work round np.intp bug.

Some versions of scipy.ndimage interpolation routines can't handle the np.intp
type. I (MB) have only seen this on a 32-bit machine runing scipy 0.9.0
"""

import numpy as np

import scipy.ndimage as spnd

# int dtype corresponding to intp
_INT_DTYPE = np.dtype(np.int32 if np.dtype(np.intp).itemsize == 4 else
                      np.int64)


# Do not run doctests on this module via nose; scipy doctests unreliable
__test__ = False


def _proc_array(array):
    """ Change array dtype from intp to int32 / int64

    Parameters
    ----------
    array : ndarray

    Returns
    -------
    output_array : ndarray
        `array` unchanged or view of array where array dtype has been changed
        from ``np.intp`` to ``np.int32`` or ``np.int64`` depending on whether
        this is a 32 or 64 bit numpy.  All other dtypes unchanged.
    """
    if array.dtype == np.dtype(np.intp):
        return array.view(_INT_DTYPE)
    return array


def _proc_output(output):
    """ Change dtype from intp to int32 / int64 for ndimage output parameter

    Allowed values to `output` are the same as listed for the ``output``
    parameter to ``scipy.ndimage.affine_transform``.

    Parameters
    ----------
    output : None or ndarray or or dtype or dtype specifier
        Can be ndarray (will have ``.dtype`` attribute), or a numpy dtype, or
        something that can be converted to a numpy dtype such as a string dtype
        code or numpy type.

    Returns
    -------
    output_fixed : None or ndarray or dtype or dtype specifier
        `output` where array dtype or dtype specifier has been changed from
        ``np.intp`` to ``np.int32`` or ``np.int64`` depending on whether this
        is a 32 or 64 bit numpy.  All other dtypes unchanged.  None returned
        unchanged
    """
    if output is None:
        return None
    if hasattr(output, 'dtype'):  # output was ndarray
        return _proc_array(output)
    # output can also be a dtype specifier
    if np.dtype(output) == np.dtype(np.intp):  # dtype specifier for np.intp
        return _INT_DTYPE
    return output


def affine_transform(input, matrix, offset=0.0, output_shape=None, output=None,
                     order=3, mode='constant', cval=0.0, prefilter=True):
    return spnd.affine_transform(_proc_array(input), matrix, offset,
                                 output_shape, _proc_output(output), order,
                                 mode, cval, prefilter)

affine_transform.__doc__ = spnd.affine_transform.__doc__


def map_coordinates(input, coordinates, output=None, order=3,
                    mode='constant', cval=0.0, prefilter=True):
    return spnd.map_coordinates(_proc_array(input), coordinates,
                                _proc_output(output), order, mode, cval,
                                prefilter)

map_coordinates.__doc__ = spnd.map_coordinates.__doc__
