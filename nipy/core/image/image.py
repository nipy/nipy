# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Define the Image class and functions to work with Image instances

* fromarray : create an Image instance from an ndarray (deprecated in favor of
  using the Image constructor)
* subsample : slice an Image instance (deprecated in favor of image slicing)
* rollimg : roll an image axis to given location
* synchronized_order : match coordinate systems between images
* iter_axis : make iterator to iterate over an image axis
* is_image : test for an object obeying the Image API
"""
import warnings
from copy import copy

import numpy as np

from nibabel.onetime import setattr_on_read

# These imports are used in the fromarray and subsample functions only, not in
# Image
from ..reference.coordinate_map import (AffineTransform, CoordinateSystem,
                                       input_axis_index)
from ..reference.array_coords import ArrayCoordMap


class Image(object):
    """ The `Image` class provides the core object type used in nipy.

    An `Image` represents a volumetric brain image and provides means
    for manipulating the image data.  Most functions in the image module
    operate on `Image` objects.

    Notes
    -----
    Images can be created through the module functions.  See nipy.io for
    image IO such as ``load`` and ``save``

    Examples
    --------
    Load an image from disk

    >>> from nipy.testing import anatfile
    >>> from nipy.io.api import load_image
    >>> img = load_image(anatfile)

    Make an image from an array.  We need to make a meaningful coordinate map
    for the image.

    >>> arr = np.zeros((21,64,64), dtype=np.int16)
    >>> cmap = AffineTransform('kji', 'zxy', np.eye(4))
    >>> img = Image(arr, cmap)
    """
    _doc = {}

    # Dictionary to store docs for attributes that are properties.  We
    # want these docs to conform with our documentation standard, but
    # they need to be passed into the property function.  Defining
    # them separately allows us to do this without a lot of clutter
    # in the property line.

    ###################################################################
    #
    # Attributes
    #
    ###################################################################

    metadata = {}
    _doc['metadata'] = "Dictionary containing additional information."

    coordmap = AffineTransform(CoordinateSystem('ijk'),
                               CoordinateSystem('xyz'),
                               np.diag([3,5,7,1]))
    _doc['coordmap'] = "Affine transform mapping from axes coordinates to reference coordinates."

    @setattr_on_read
    def shape(self):
        return self._data.shape
    _doc['shape'] = "Shape of data array."

    @setattr_on_read
    def ndim(self):
        return len(self._data.shape)
    _doc['ndim'] = "Number of data dimensions."

    @setattr_on_read
    def reference(self):
        return self.coordmap.function_range
    _doc['reference'] = "Reference coordinate system."

    @setattr_on_read
    def axes(self):
        return self.coordmap.function_domain
    _doc['axes'] = "Axes of image."

    @setattr_on_read
    def affine(self):
        if hasattr(self.coordmap, "affine"):
            return self.coordmap.affine
        raise AttributeError('Nonlinear transform does not have an affine.')
    _doc['affine'] = "Affine transformation if one exists."

    ###################################################################
    #
    # Properties
    #
    ###################################################################

    def _getheader(self):
        # data loaded from a file may have a header
        warnings.warn("Please don't use ``img.header``; use"
                      "``img.metadata['header'] instead",
                      DeprecationWarning,
                      stacklevel=2)
        hdr = self.metadata.get('header')
        if hdr is None:
            raise AttributeError('Image created from arrays '
                                 'may not have headers.')
        return hdr
    def _setheader(self, header):
        warnings.warn("Please don't use ``img.header``; use"
                      "``img.metadata['header'] instead",
                      DeprecationWarning,
                      stacklevel=2)
        self.metadata['header'] = header
    _doc['header'] = \
    """The file header structure for this image, if available.  This interface
    will soon go away - you should use ``img.metadata['header'] instead.
    """
    header = property(_getheader, _setheader, doc=_doc['header'])

    ###################################################################
    #
    # Constructor
    #
    ###################################################################

    def __init__(self, data, coordmap, metadata=None):
        """Create an `Image` object from array and `CoordinateMap` object.

        Images are often created through the ``load_image`` function in the nipy
        base namespace.

        Parameters
        ----------
        data : array-like
            object that as attribute ``shape`` and returns an array from
            ``np.asarray(data)``
        coordmap : `AffineTransform` object
            coordmap mapping the domain (input) voxel axes of the image to the
            range (reference, output) axes - usually mm in real world space
        metadata : dict, optional
            Freeform metadata for image.  Most common contents is ``header``
            from nifti etc loaded images.

        See Also
        --------
        load_image : load ``Image`` from a file
        save_image : save ``Image`` to a file
        """
        if metadata is None:
            metadata = {}
        else: # Shallow copy
            metadata = copy(metadata)
        ndim = len(data.shape)
        if not isinstance(coordmap, AffineTransform):
            raise ValueError('coordmap must be an AffineTransform')
        # self._data is an array-like object.  It must have a shape attribute
        # (see above) and return an array from np.array(data)
        self._data = data
        self.coordmap = coordmap
        if coordmap.function_domain.ndim != ndim:
            raise ValueError('the number of axes implied by the coordmap do '
                             'not match the number of axes of the data')
        self.metadata = metadata

    ###################################################################
    #
    # Methods
    #
    ###################################################################

    def reordered_reference(self, order=None):
        """ Return new Image with reordered output coordinates

        New Image coordmap has reordered output coordinates. This does
        not transpose the data.

        Parameters
        ----------
        order : None, sequence, optional
            sequence of int (giving indices) or str (giving names) - expressing
            new order of coordmap output coordinates.  None (the default)
            results in reversed ordering.

        Returns
        -------
        r_img : object
            Image of same class as `self`, with reordered output coordinates.

        Examples
        --------
        >>> cmap = AffineTransform.from_start_step(
        ...             'ijk', 'xyz', [1, 2, 3], [4, 5, 6], 'domain', 'range')
        >>> im = Image(np.empty((30,40,50)), cmap)
        >>> im_reordered = im.reordered_reference([2,0,1])
        >>> im_reordered.shape
        (30, 40, 50)
        >>> im_reordered.coordmap
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='domain', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('z', 'x', 'y'), name='range', coord_dtype=float64),
           affine=array([[ 0.,  0.,  6.,  3.],
                         [ 4.,  0.,  0.,  1.],
                         [ 0.,  5.,  0.,  2.],
                         [ 0.,  0.,  0.,  1.]])
        )
        """
        if order is None:
            order = range(self.ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.reference.index(s) for s in order]
        new_cmap = self.coordmap.reordered_range(order)
        return self.__class__.from_image(self, coordmap=new_cmap)

    def reordered_axes(self, order=None):
        """ Return a new Image with reordered input coordinates.

        This transposes the data as well.

        Parameters
        ----------
        order : None, sequence, optional
            Sequence of int (giving indices) or str (giving names) - expressing
            new order of coordmap output coordinates.  None (the default)
            results in reversed ordering.

        Returns
        -------
        r_img : object
            Image of same class as `self`, with reordered output coordinates.

        Examples
        --------
        >>> cmap = AffineTransform.from_start_step(
        ...             'ijk', 'xyz', [1, 2, 3], [4, 5, 6], 'domain', 'range')
        >>> cmap
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='domain', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='range', coord_dtype=float64),
           affine=array([[ 4.,  0.,  0.,  1.],
                         [ 0.,  5.,  0.,  2.],
                         [ 0.,  0.,  6.,  3.],
                         [ 0.,  0.,  0.,  1.]])
        )
        >>> im = Image(np.empty((30,40,50)), cmap)
        >>> im_reordered = im.reordered_axes([2,0,1])
        >>> im_reordered.shape
        (50, 30, 40)
        >>> im_reordered.coordmap
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='domain', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='range', coord_dtype=float64),
           affine=array([[ 0.,  4.,  0.,  1.],
                         [ 0.,  0.,  5.,  2.],
                         [ 6.,  0.,  0.,  3.],
                         [ 0.,  0.,  0.,  1.]])
        )
        """
        if order is None:
            order = range(self.ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.axes.index(s) for s in order]
        new_cmap = self.coordmap.reordered_domain(order)
        # Only transpose if we have to so as to avoid calling
        # self.get_data
        if order != range(self.ndim):
            new_data = np.transpose(self.get_data(), order)
        else:
            new_data = self._data
        return self.__class__.from_image(self,
                                         data=new_data,
                                         coordmap=new_cmap)

    def renamed_axes(self, **names_dict):
        """ Return a new image with input (domain) axes renamed

        Axes renamed according to the input dictionary.

        Parameters
        ----------
        \*\*names_dict : dict
            with keys being old names, and values being new names

        Returns
        -------
        newimg : Image
            An Image with the same data, having its axes renamed.

        Examples
        --------
        >>> data = np.random.standard_normal((11,9,4))
        >>> im = Image(data, AffineTransform.from_params('ijk', 'xyz', np.identity(4), 'domain', 'range'))
        >>> im_renamed = im.renamed_axes(i='slice')
        >>> print im_renamed.axes
        CoordinateSystem(coord_names=('slice', 'j', 'k'), name='domain', coord_dtype=float64)
        """
        new_cmap = self.coordmap.renamed_domain(names_dict)
        return self.__class__.from_image(self, coordmap=new_cmap)

    def renamed_reference(self, **names_dict):
        """ Return new image with renamed output (range) coordinates

        Coordinates renamed according to the dictionary

        Parameters
        ----------
        \*\*names_dict : dict
            with keys being old names, and values being new names

        Returns
        -------
        newimg : Image
            An Image with the same data, having its output coordinates renamed.

        Examples
        --------
        >>> data = np.random.standard_normal((11,9,4))
        >>> im = Image(data, AffineTransform.from_params('ijk', 'xyz', np.identity(4), 'domain', 'range'))
        >>> im_renamed_reference = im.renamed_reference(x='newx', y='newy')
        >>> print im_renamed_reference.reference
        CoordinateSystem(coord_names=('newx', 'newy', 'z'), name='range', coord_dtype=float64)
        """
        new_cmap = self.coordmap.renamed_range(names_dict)
        return self.__class__.from_image(self, coordmap=new_cmap)

    def __setitem__(self, index, value):
        """Setting values of an image, set values in the data array."""
        warnings.warn("Please don't use ``img[x] = y``; use "
                      "``img.get_data()[x]  = y`` instead",
                      DeprecationWarning,
                      stacklevel=2)
        self._data[index] = value

    def __array__(self):
        """Return data as a numpy array."""
        warnings.warn('Please use get_data instead - will be deprecated',
                      DeprecationWarning,
                      stacklevel=2)
        return self.get_data()

    def get_data(self):
        """Return data as a numpy array."""
        return np.asanyarray(self._data)

    def __getitem__(self, slice_object):
        """ Slicing an image returns an Image.

        Parameters
        ----------
        slice_object: int, slice or sequence of slice
            An object representing a numpy 'slice'.

        Returns
        -------
        img_subsampled: Image
            An Image with data self.get_data()[slice_object] and an
            appropriately corrected CoordinateMap.

        Examples
        --------
        >>> from nipy.io.api import load_image
        >>> from nipy.testing import funcfile
        >>> im = load_image(funcfile)
        >>> frame3 = im[:,:,:,3]
        >>> np.allclose(frame3.get_data(), im.get_data()[:,:,:,3])
        True
        """
        data = self.get_data()[slice_object]
        g = ArrayCoordMap(self.coordmap, self.shape)[slice_object]
        coordmap = g.coordmap
        if coordmap.function_domain.ndim > 0:
            return self.__class__.from_image(self,
                                             data=data,
                                             coordmap=coordmap)
        else:
            return data

    def __iter__(self):
        """ Images do not have default iteration

        This is because it's not obvious that axis 0 is the right axis to
        iterate over.  For example, we often want to iterate over the time or
        volume axis, and this is more likely to be axis 3
        """
        raise TypeError("Images do not have default iteration; "
                        "you can use ``iter_axis(img, axis)`` instead.")

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and np.all(self.get_data() == other.get_data())
                and np.all(self.affine == other.affine)
                and (self.axes.coord_names == other.axes.coord_names))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=2)
        representation = \
            'Image(\n  data=%s,\n  coordmap=%s)' % (
            '\n       '.join(repr(self._data).split('\n')),
            '\n         '.join(repr(self.coordmap).split('\n')))
        np.set_printoptions(**options)
        return representation

    @classmethod
    def from_image(klass, img, data=None, coordmap=None, metadata=None):
        """ Classmethod makes new instance of this `klass` from instance `img`

        Parameters
        ----------
        data : array-like
            object that as attribute ``shape`` and returns an array from
            ``np.asarray(data)``
        coordmap : `AffineTransform` object
            coordmap mapping the domain (input) voxel axes of the image to the
            range (reference, output) axes - usually mm in real world space
        metadata : dict, optional
            Freeform metadata for image.  Most common contents is ``header``
            from nifti etc loaded images.

        Returns
        -------
        img : `klass` instance
            New image with data from `data`, coordmap from `coordmap` maybe
            metadata from `metadata`

        Notes
        -----
        Subclasses of ``Image`` with different semantics for ``__init__`` will
        need to override this classmethod.

        Examples
        --------
        >>> from nipy import load_image
        >>> from nipy.core.api import Image
        >>> from nipy.testing import anatfile
        >>> aimg = load_image(anatfile)
        >>> arr = np.arange(24).reshape((2,3,4))
        >>> img = Image.from_image(aimg, data=arr)
        """
        if data is None:
            data = img._data
        if coordmap is None:
            coordmap = copy(img.coordmap)
        if metadata is None:
            metadata = copy(img.metadata)
        return klass(data, coordmap, metadata)


class SliceMaker(object):
    """ This class just creates slice objects for image resampling

    It only has a __getitem__ method that returns its argument.

    XXX Wouldn't need this if there was a way
    XXX to do this
    XXX subsample(img, [::2,::3,10:1:-1])
    XXX
    XXX Could be something like this Subsample(img)[::2,::3,10:1:-1]
    """
    def __getitem__(self, index):
        return index

slice_maker = SliceMaker()


def subsample(img, slice_object):
    """ Subsample an image

    Please don't use this function, but use direct image slicing instead.  That
    is, replace::

        frame3 = subsample(im, slice_maker[:,:,:,3])

    with::

        frame3 = im[:,:,:,3]

    Parameters
    ----------
    img : Image
    slice_object: int, slice or sequence of slice
       An object representing a numpy 'slice'.

    Returns
    -------
    img_subsampled: Image
         An Image with data img.get_data()[slice_object] and an appropriately
         corrected CoordinateMap.

    Examples
    --------
    >>> from nipy.io.api import load_image
    >>> from nipy.testing import funcfile
    >>> from nipy.core.api import subsample, slice_maker
    >>> im = load_image(funcfile)
    >>> frame3 = subsample(im, slice_maker[:,:,:,3])
    >>> np.allclose(frame3.get_data(), im.get_data()[:,:,:,3])
    True
    """
    warnings.warn('subsample is deprecated, please use image '
                  'slicing instead (e.g. img[:,:,1]',
                  DeprecationWarning,
                  stacklevel=2)
    return img.__getitem__(slice_object)


def fromarray(data, innames, outnames):
    """Create an image from array `data`, and input/output coordinate names

    The mapping between the input and output coordinate names is the identity
    matrix.

    Please don't use this routine, but instead prefer::

        from nipy.core.api import Image, AffineTransform
        img = Image(data, AffineTransform(innames, outnames, np.eye(4)))

    where ``4`` is ``len(innames) + 1``.

    Parameters
    ----------
    data : numpy array
        A numpy array of three dimensions.
    innames : sequence
       a list of input axis names
    innames : sequence
       a list of output axis names

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    save : function for saving images

    Examples
    --------
    >>> img = fromarray(np.zeros((2,3,4)), 'ijk', 'xyz')
    >>> img.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
       affine=array([[ 1.,  0.,  0.,  0.],
                     [ 0.,  1.,  0.,  0.],
                     [ 0.,  0.,  1.,  0.],
                     [ 0.,  0.,  0.,  1.]])
    )
    """
    warnings.warn('fromarray is deprecated, please use the Image '
                  'constructor instead',
                  DeprecationWarning,
                  stacklevel=2)
    ndim = len(data.shape)
    coordmap = AffineTransform.from_start_step(innames,
                                               outnames,
                                               (0.,)*ndim,
                                               (1.,)*ndim)
    return Image(data, coordmap)


@np.deprecate_with_doc('Please use rollimg instead')
def rollaxis(img, axis, inverse=False):
    """ Roll `axis` backwards, until it lies in the first position.

    It also reorders the reference coordinates by the same ordering.
    This is done to preserve a diagonal affine matrix if image.affine
    is diagonal. It also makes it possible to unambiguously specify
    an axis to roll along in terms of either a reference name (i.e. 'z')
    or an axis name (i.e. 'slice').

    This function is deprecated; please use ``rollimg`` instead.

    Parameters
    ----------
    img : Image
       Image whose axes and reference coordinates are to be reordered
       by rolling.
    axis : str or int
       Axis to be rolled, can be specified by name or as an integer.
    inverse : bool, optional
       If inverse is True, then axis must be an integer and the first axis is
       returned to the position axis.  This keyword is deprecated and we'll
       remove it in a future version of nipy.

    Returns
    -------
    newimg : Image
       Image with reordered axes and reference coordinates.

    Examples
    --------
    >>> data = np.zeros((30,40,50,5))
    >>> affine_transform = AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1]))
    >>> im = Image(data, affine_transform)
    >>> im.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='', coord_dtype=float64),
       affine=array([[ 1.,  0.,  0.,  0.,  0.],
                     [ 0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.,  0.],
                     [ 0.,  0.,  0.,  4.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> im_t_first = rollaxis(im, 't')
    >>> np.diag(im_t_first.affine)
    array([ 4.,  1.,  2.,  3.,  1.])
    >>> im_t_first.shape
    (5, 30, 40, 50)
    >>> im_t_first.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('t', 'x', 'y', 'z'), name='', coord_dtype=float64),
       affine=array([[ 4.,  0.,  0.,  0.,  0.],
                     [ 0.,  1.,  0.,  0.,  0.],
                     [ 0.,  0.,  2.,  0.,  0.],
                     [ 0.,  0.,  0.,  3.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    """
    if inverse not in (True, False):
        raise ValueError('Inverse should be True or False; did you mean to '
                         'use the ``rollimg` function instead?')
    if isinstance(axis, int) and axis < 0:
        axis = img.ndim + axis
    if inverse:
        if type(axis) != type(0):
            raise ValueError('If carrying out inverse rolling, '
                             'axis must be an integer')
        order = range(1, img.ndim)
        order.insert(axis, 0)
        return img.reordered_axes(order).reordered_reference(order)
    if axis not in (range(img.axes.ndim) +
                    list(img.axes.coord_names) +
                    list(img.reference.coord_names)):
        raise ValueError('axis must be an axis number,'
                         'an axis name or a reference name')
    # Find out which index axis corresonds to
    in_index = out_index = -1
    if type(axis) == type(''):
        try:
            in_index = img.axes.index(axis)
        except:
            pass
        try:
            out_index = img.reference.index(axis)
        except:
            pass
        if in_index > 0 and out_index > 0 and in_index != out_index:
            raise ValueError('ambiguous choice of axis -- it exists '
                             'both in as an axis name and a '
                             'reference name')
        if in_index >= 0:
            axis = in_index
        else:
            axis = out_index
    if axis == -1:
        axis += img.axes.ndim
    order = range(img.ndim)
    order.remove(axis)
    order.insert(0, axis)
    return img.reordered_axes(order).reordered_reference(order)


def rollimg(img, axis, start=0, fix0=True):
    """ Roll `axis` backwards in the inputs, until it lies before `start`

    Parameters
    ----------
    img : Image
        Image whose axes and reference coordinates are to be reordered by
        rollimg.
    axis : str or int
        Axis to be rolled, can be specified by name or as an integer.  If an
        integer, axis is an input axis.  If a name, can be name of input or
        output axis.  If an output axis, we search for the closest matching
        input axis, and raise an AxisError if this fails.
    start : str or int, optional
        position before which to roll axis `axis`.  Default to 0.  Can again be
        an integer (input axis) or name of input or output axis.
    fix0 : bool, optional
        Whether to allow for zero scaling when searching for an input axis
        matching an output axis.  Useful for images where time scaling is 0.

    Returns
    -------
    newimg : Image
       Image with reordered input axes and corresponding data.

    Examples
    --------
    >>> data = np.zeros((30,40,50,5))
    >>> affine_transform = AffineTransform('ijkl', 'xyzt', np.diag([1,2,3,4,1]))
    >>> im = Image(data, affine_transform)
    >>> im.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='', coord_dtype=float64),
       affine=array([[ 1.,  0.,  0.,  0.,  0.],
                     [ 0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.,  0.],
                     [ 0.,  0.,  0.,  4.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    >>> im_t_first = rollimg(im, 't')
    >>> im_t_first.shape
    (5, 30, 40, 50)
    >>> im_t_first.coordmap
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('l', 'i', 'j', 'k'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='', coord_dtype=float64),
       affine=array([[ 0.,  1.,  0.,  0.,  0.],
                     [ 0.,  0.,  2.,  0.,  0.],
                     [ 0.,  0.,  0.,  3.,  0.],
                     [ 4.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]])
    )
    """
    axis = input_axis_index(img.coordmap, axis, fix0)
    start = input_axis_index(img.coordmap, start, fix0)
    order = range(img.ndim)
    order.remove(axis)
    if axis < start:
        start -= 1
    order.insert(start, axis)
    return img.reordered_axes(order)


def iter_axis(img, axis, asarray=False):
    """ Return generator to slice an image `img` over `axis`

    Parameters
    ----------
    img : ``Image`` instance
    axis : int or str
        axis identifier, either name or axis number
    asarray : {False, True}, optional

    Returns
    -------
    g : generator
        such that list(g) returns a list of slices over `axis`.  If `asarray` is
        `False` the slices are images.  If `asarray` is True, slices are the
        data from the images.

    Examples
    --------
    >>> data = np.arange(24).reshape((4,3,2))
    >>> img = Image(data, AffineTransform('ijk', 'xyz', np.eye(4)))
    >>> slices = list(iter_axis(img, 'j'))
    >>> len(slices)
    3
    >>> slices[0].shape
    (4, 2)
    >>> slices = list(iter_axis(img, 'k', asarray=True))
    >>> slices[1].sum() == data[:,:,1].sum()
    True
    """
    rimg = rollimg(img, axis)
    for i in range(rimg.shape[0]):
        if asarray:
            yield rimg[i].get_data()
        else:
            yield rimg[i]


def synchronized_order(img, target_img,
                       axes=True,
                       reference=True):
    """ Reorder reference and axes of `img` to match target_img.

    Parameters
    ----------
    img : Image
    target_img : Image
    axes : bool, optional
        If True, synchronize the order of the axes.
    reference : bool, optional
        If True, synchronize the order of the reference coordinates.

    Returns
    -------
    newimg : Image
        An Image satisfying newimg.axes == target.axes (if axes == True),
        newimg.reference == target.reference (if reference == True).

    Examples
    --------
    >>> data = np.random.standard_normal((3,4,7,5))
    >>> im = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    >>> im_scrambled = im.reordered_axes('iljk').reordered_reference('txyz')
    >>> im == im_scrambled
    False
    >>> im_unscrambled = synchronized_order(im_scrambled, im)
    >>> im == im_unscrambled
    True

    The images don't have to be the same shape

    >>> data2 = np.random.standard_normal((3,11,9,4))
    >>> im2 = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    >>> im_scrambled2 = im2.reordered_axes('iljk').reordered_reference('xtyz')
    >>> im_unscrambled2 = synchronized_order(im_scrambled2, im)
    >>> im_unscrambled2.coordmap == im.coordmap
    True

    or have the same coordmap

    >>> data3 = np.random.standard_normal((3,11,9,4))
    >>> im3 = Image(data3, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,9,3,-2,1])))
    >>> im_scrambled3 = im3.reordered_axes('iljk').reordered_reference('xtyz')
    >>> im_unscrambled3 = synchronized_order(im_scrambled3, im)
    >>> im_unscrambled3.axes == im.axes
    True
    >>> im_unscrambled3.reference == im.reference
    True
    >>> im_unscrambled4 = synchronized_order(im_scrambled3, im, axes=False)
    >>> im_unscrambled4.axes == im.axes
    False
    >>> im_unscrambled4.axes == im_scrambled3.axes
    True
    >>> im_unscrambled4.reference == im.reference
    True
    """
    # Caution, we can't just use target_img.reference because other subclasses
    # of Image may not have all axes in the .reference attribute.
    target_axes = target_img.axes # = target_img.coordmap.function_domain
    # the below not necessarily == target_image.reference
    target_reference = target_img.coordmap.function_range
    if axes:
        img = img.reordered_axes(target_axes.coord_names)
    if reference:
        img = img.reordered_reference(target_reference.coord_names)
    return img


def is_image(obj):
    ''' Returns true if this object obeys the Image API

    This allows us to test for something that is duck-typing an image.

    For now an array must have a 'coordmap' attribute, and a callable
    'get_data' attribute.

    Parameters
    ----------
    obj : object
       object for which to test API

    Returns
    -------
    is_img : bool
       True if object obeys image API

    Examples
    --------
    >>> from nipy.testing import anatfile
    >>> from nipy.io.api import load_image
    >>> img = load_image(anatfile)
    >>> is_image(img)
    True
    >>> class C(object): pass
    >>> c = C()
    >>> is_image(c)
    False
    '''
    if not hasattr(obj, 'coordmap') or not hasattr(obj, 'metadata'):
        return False
    return callable(getattr(obj, 'get_data'))
