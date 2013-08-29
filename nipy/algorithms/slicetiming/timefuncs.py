""" Utility functions for returning slice times from number of slices and TR

Slice timing routines in nipy need a vector of slice times.

Slice times are vectors $t_i i = 0 ... N$ of times, one for each slice, where
$t_i% gives the time at which slice number $i$ was acquired, relative to the
beginning of the volume acquisition.

We like these vectors because they are unambiguous; the indices $i$ refer to
positions in space, and the values $t_i$ refer to times.

But, there are many common slice timing regimes for which it's easy to get the
slice times once you know the volume acquisition time (the TR) and the number of
slices.

For example, if you acquired the slices in a simple ascending order, and you
have 10 slices and the TR was 2.0, then the slice times are:

>>> import numpy as np
>>> np.arange(10) / 10.  * 2.0
array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8])

These are small convenience functions that accept the number of slices and the
TR as input, and return a vector of slice times:

>>> ascending(10, 2.)
array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8])
"""
from __future__ import division, print_function, absolute_import

import numpy as np

# Dictionary (key, value) == (name, function) for slice timing functions
SLICETIME_FUNCTIONS = {}

def _dec_filldoc(func):
    """ Fill docstring of slice time function
    """
    func._doc_template = func.__doc__
    func.__doc__ = func.__doc__.format(
        **dict(
            name = func.__name__,
            pstr=
"""Note: slice 0 is the first slice in the voxel data block

    Parameters
    ----------
    n_slices : int
        Number of slices in volume
    TR : float
        Time to acquire one full volume

    Returns
    -------
    slice_times : (n_slices,) ndarray
        Vectors $t_i i = 0 ... N$ of times, one for each slice, where $t_i$
        gives the time at which slice number $i$ was acquired, relative to the
        beginning of the volume acquisition.
    """))
    return func


def _dec_register_stf(func):
    """ Register slice time function in module dictionary """
    name = func.__name__
    SLICETIME_FUNCTIONS[name] = func
    if name.startswith('st_'):
        short_name = name[3:]
        if short_name in SLICETIME_FUNCTIONS:
            raise ValueError(
                "Duplicate short / long function name {0}".format(short_name))
        SLICETIME_FUNCTIONS[short_name] = func
    return func


def _dec_stfunc(func):
    return _dec_register_stf(_dec_filldoc(func))


def _derived_func(name, func):
    def derived(n_slices, TR):
        return func(n_slices, TR)
    derived.__name__ = name
    derived.__doc__ = func._doc_template
    return _dec_stfunc(derived)


@_dec_stfunc
def st_01234(n_slices, TR):
    """ Simple ascending slice sequence

    slice 0 first, slice 1 second etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0. ,  0.2,  0.4,  0.6,  0.8])

    {pstr}
    """
    return np.arange(n_slices) / n_slices * TR

ascending = _derived_func('ascending', st_01234)


@_dec_stfunc
def st_43210(n_slices, TR):
    """ Simple descending slice sequence

    slice ``n_slices-1`` first, slice ``n_slices - 2`` second etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0.8,  0.6,  0.4,  0.2,  0. ])

    {pstr}
    """
    return np.arange(n_slices)[::-1] / n_slices * TR

descending = _derived_func('descending', st_43210)


@_dec_stfunc
def st_02413(n_slices, TR):
    """Ascend alternate every second slice, starting at first slice

    Collect slice 0 first, slice 2 second up to top.  Then return to collect
    slice 1, slice 3 etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0. ,  0.6,  0.2,  0.8,  0.4])

    {pstr}
    """
    one_slice = TR / n_slices
    time_to_space = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
    space_to_time = np.argsort(time_to_space)
    return space_to_time * one_slice

asc_alt_2 = _derived_func('asc_alt_2', st_02413)


@_dec_stfunc
def st_13024(n_slices, TR):
    """Ascend alternate every second slice, starting at second slice

    Collect slice 1 first, slice 3 second up to top (highest numbered slice).
    Then return to collect slice 0, slice 2 etc.  This order is rare except on
    Siemens acquisitions with an even number of slices.  See
    :func:`st_odd0_even1` for this logic.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0.4,  0. ,  0.6,  0.2,  0.8])

    {pstr}
    """
    one_slice = TR / n_slices
    time_to_space = list(range(1, n_slices, 2)) + list(range(0, n_slices, 2))
    space_to_time = np.argsort(time_to_space)
    return space_to_time * one_slice

asc_alt_2_1 = _derived_func('asc_alt_2_1', st_13024)


@_dec_stfunc
def st_42031(n_slices, TR):
    """Descend alternate every second slice, starting at last slice

    Collect slice (`n_slices` - 1) first, slice (`nslices` - 3) second down to
    bottom (lowest numbered slice).  Then return to collect slice (`n_slices`
    -2), slice (`n_slices` - 4) etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0.4,  0.8,  0.2,  0.6,  0. ])

    {pstr}
    """
    return st_02413(n_slices, TR)[::-1]

desc_alt_2 = _derived_func('desc_alt_2', st_42031)


@_dec_stfunc
def st_odd0_even1(n_slices, TR):
    """Ascend alternate starting at slice 0 for odd, slice 1 for even `n_slices`

    Acquisitions with alternating ascending slices from Siemens scanners often
    seem to have this behavior as default - see:
        https://mri.radiology.uiowa.edu/fmri_images.html

    This means we use the :func:`st_02413` algorithm if `n_slices` is odd,
    and the :func:`st_13024` algorithm if `n_slices is even.

    For example, for 4 slices and a TR of 1:

    >>> {name}(4, 1.)
    array([ 0.5 ,  0.  ,  0.75,  0.25])

    5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0. ,  0.6,  0.2,  0.8,  0.4])

    {pstr}
    """
    if n_slices % 2 == 0:
        return st_13024(n_slices, TR)
    return st_02413(n_slices, TR)

asc_alt_siemens = _derived_func('asc_alt_siemens', st_odd0_even1)


@_dec_stfunc
def st_03142(n_slices, TR):
    """Ascend alternate, where alternation is by half the volume

    Collect slice 0 then slice ``ceil(n_slices / 2.)`` then slice 1 then slice
    ``ceil(nslices / 2.) + 1`` etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0. ,  0.4,  0.8,  0.2,  0.6])

    {pstr}
    """
    one_slice = TR / n_slices
    space_to_time = (list(range(0, n_slices, 2)) +
                     list(range(1, n_slices, 2)))
    return np.array(space_to_time) * one_slice

asc_alt_half = _derived_func('asc_alt_half', st_03142)


@_dec_stfunc
def st_41302(n_slices, TR):
    """Descend alternate, where alternation is by half the volume

    Collect slice (n_slices - 1) then slice ``floor(nslices / 2.) - 1`` then slice
    (n_slices - 2) then slice ``floor(nslices / 2.) - 2`` etc.

    For example, for 5 slices and a TR of 1:

    >>> {name}(5, 1.)
    array([ 0.6,  0.2,  0.8,  0.4,  0. ])

    {pstr}
    """
    return st_03142(n_slices, TR)[::-1]

desc_alt_half = _derived_func('desc_alt_half', st_41302)
