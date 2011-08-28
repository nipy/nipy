# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module defines a few common generators for slicing over arrays.

They are defined on ndarray, so they do not depend on Image.

* data_generator: return (item, data[item]) tuples from an iterable object
* slice_generator: return slices through an ndarray, possibly over many
  indices
* f_generator: return a generator that applies a function to the
  output of another generator

The above three generators return 2-tuples.

* write_data: write the output of a generator to an ndarray
* parcels: return binary array of the unique components of data
"""

import numpy as np


def parcels(data, labels=None, exclude=[]):
    """ Return a generator for ``[data == label for label in labels]``

    If labels is None, labels = numpy.unique(data).  Each label in labels can be
    a sequence, in which case the value returned for that label union::

        [numpy.equal(data, l) for l in label]

    Examples
    --------
    >>> for p in parcels([[1,1],[2,1]]):
    ...     print p
    ...
    [[ True  True]
     [False  True]]
    [[False False]
     [ True False]]
    >>> for p in parcels([[1,1],[2,3]], labels=[2,3]):
    ...     print p
    ...
    [[False False]
     [ True False]]
    [[False False]
     [False  True]]
    >>> for p in parcels([[1,1],[2,3]], labels=[(2,3),2]):
    ...     print p
    ...
    [[False False]
     [ True  True]]
    [[False False]
     [ True False]]
    """
    data = np.asarray(data)
    if labels is None:
        labels = np.unique(data)
    if exclude:
        labels = set(labels)
        for e in exclude:
            if e in labels:
                labels.remove(e)
    for label in labels:
        if type(label) not in [type(()), type([])]:
            yield np.equal(data, label)
        else:
            v = 0
            for l in label:
                v += np.equal(data, l)
            yield v.astype(bool)


def data_generator(data, iterable=None):
    """ Return generator for ``[(i, data[i]) for i in iterable]``

    If iterable is None, it defaults to range(data.shape[0])

    Examples
    --------
    >>> a = np.asarray([[True,False],[False,True]])
    >>> b = np.asarray([[False,False],[True,False]])

    >>> for i, d in data_generator(np.asarray([[1,2],[3,4]]), [a,b]):
    ...     print d
    ...
    [1 4]
    [3]
    """
    data = np.asarray(data)
    if iterable is None:
        iterable = range(data.shape[0])
    for index in iterable:
        yield index, data[index]


def write_data(output, iterable):
    """ Write (index, data) iterable to `output`

    Write some data to `output`. Iterable should return 2-tuples of the form
    index, data such that::

        output[index] = data

    makes sense.

    Examples
    --------
    >>> a=np.zeros((2,2))
    >>> write_data(a, data_generator(np.asarray([[1,2],[3,4]])))
    >>> a
    array([[ 1.,  2.],
           [ 3.,  4.]])
    """
    for index, data in iterable:
        try:
            print index.shape, 'index', data.shape
        except:
            pass
        output[index] = data


def slice_generator(data, axis=0):
    """ Return generator for yielding slices along `axis`

    Examples
    --------
    >>> for i,d in slice_generator([[1,2],[3,4]]):
    ...     print i, d
    ...
    (0,) [1 2]
    (1,) [3 4]
    >>> for i,d in slice_generator([[1,2],[3,4]], axis=1):
    ...     print i, d
    ...
    (slice(None, None, None), 0) [1 3]
    (slice(None, None, None), 1) [2 4]
    """
    data = np.asarray(data)
    if type(axis) is type(1):
        for j in range(data.shape[axis]):
            ij = (slice(None,None,None),)*axis + (j,)
            yield ij, data[(slice(None,None,None),)*axis + (j,)]
    elif type(axis) in [type(()),type([])]:
        data = np.asarray(data)

        # the total number of iterations to be made
        nmax = np.product(np.asarray(data.shape)[axis])

        # calculate the 'divmod' paramter which is used to work out
        # which index to use to use for each axis during iteration
        mods = np.cumprod(np.asarray(data.shape)[axis])
        divs = [1] + list(mods[:-1])

        # set up a full set of slices for the image, to be modified
        # at each iteration
        slices = [slice(0, s) for s in data.shape]

        n = 0
        while True:

            if n >= nmax:
                raise StopIteration
            for (a, div, mod) in zip(axis, divs, mods):
                x = n / div % mod
                slices[a] = x
            n += 1
            yield slices, data[slices]


def f_generator(f, iterable):
    """ Return a generator for ``[(i, f(x)) for i, x in iterable]``

    Examples
    --------
    >>> for i, d in f_generator(lambda x: x**2, data_generator([[1,2],[3,4]])):
    ...     print i, d
    ...
    0 [1 4]
    1 [ 9 16]
    """
    for i, x in iterable:
        yield i, np.asarray(f(x))


def slice_parcels(data, labels=None, axis=0):
    """
    A generator for slicing through parcels and slices of data...

    hmmm... a better description is needed

    >>> x=np.array([[0,0,0,1],[0,1,0,1],[2,2,0,1]])
    >>> for a in slice_parcels(x):
    ...     print a, x[a]
    ...
    ((0,), array([ True,  True,  True, False], dtype=bool)) [0 0 0]
    ((0,), array([False, False, False,  True], dtype=bool)) [1]
    ((1,), array([ True, False,  True, False], dtype=bool)) [0 0]
    ((1,), array([False,  True, False,  True], dtype=bool)) [1 1]
    ((2,), array([False, False,  True, False], dtype=bool)) [0]
    ((2,), array([False, False, False,  True], dtype=bool)) [1]
    ((2,), array([ True,  True, False, False], dtype=bool)) [2 2]
    >>> for a in slice_parcels(x, axis=1):
    ...     b, c = a
    ...     print a, x[b][c]
    ...
    ((slice(None, None, None), 0), array([ True,  True, False], dtype=bool)) [0 0]
    ((slice(None, None, None), 0), array([False, False,  True], dtype=bool)) [2]
    ((slice(None, None, None), 1), array([ True, False, False], dtype=bool)) [0]
    ((slice(None, None, None), 1), array([False,  True, False], dtype=bool)) [1]
    ((slice(None, None, None), 1), array([False, False,  True], dtype=bool)) [2]
    ((slice(None, None, None), 2), array([ True,  True,  True], dtype=bool)) [0 0 0]
    ((slice(None, None, None), 3), array([ True,  True,  True], dtype=bool)) [1 1 1]
    """
    for i, d in slice_generator(data, axis=axis):
        for p in parcels(d, labels=labels):
            yield (i, p)


def matrix_generator(img):
    """
    From a generator of items (i, r), return
    (i, rp) where rp is a 2d array with rp.shape = (r.shape[0], prod(r.shape[1:]))
    """
    for i, r in img:
        r.shape = (r.shape[0], np.product(r.shape[1:]))
        yield i, r


def shape_generator(img, shape):
    """
    From a generator of items (i, r), return
    (i, r.reshape(shape))
    """
    for i, r in img:
        r.shape = shape
        yield i, r
