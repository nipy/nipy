"""
This module defines the Image class, as well as two functions that create Image instances.

fromarray : create an Image instance from an ndarray

"""
import numpy as np
import warnings

from nipy.utils.onetime import setattr_on_read

# These imports are used in the fromarray and subsample 
# functions only, not in Image

from nipy.core.reference.coordinate_map import AffineTransform, \
    CoordinateSystem, CoordinateMap

from nipy.core.reference.array_coords import ArrayCoordMap

__docformat__ = 'restructuredtext'
__all__ = ['fromarray', 'subsample']

class Image(object):
    """
    The `BaseImage` class provides the core object type used in nipy. An `BaseImage`
    represents a volumetric brain image and provides means for manipulating
    the image data.  Most functions in the image module operate on `BaseImage`
    objects.

    Notes
    -----
    Images are most often created through the module functions 
    load and fromarray.

    Examples
    --------

    >>> from nipy.core.image import image
    >>> from nipy.testing import anatfile
    >>> from nipy.io.api import load_image
    >>> img = load_image(anatfile)

    >>> import numpy as np
    >>> img = image.fromarray(np.zeros((21, 64, 64), dtype='int16'),
    ...                       'kji', 'zxy')

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
    _doc['coordmap'] = "Affine transform mapping from axes coordinates to world coordinates."

    @setattr_on_read
    def shape(self):
        return self._data.shape
    _doc['shape'] = "Shape of data array."

    @setattr_on_read
    def ndim(self):
        return self._data.ndim
    _doc['ndim'] = "Number of data dimensions."

    @setattr_on_read
    def world(self):
        return self.coordmap.function_range
    _doc['world'] = "World coordinate system."

    @setattr_on_read
    def axes(self):
        return self.coordmap.function_domain
    _doc['axes'] = "Axes of image."

    @setattr_on_read
    def affine(self):
        if hasattr(self.coordmap, "affine"):
            return self.coordmap.affine
        raise AttributeError, 'Nonlinear transform does not have an affine.'
    _doc['affine'] = "Affine transformation if one exists."
    
    ###################################################################
    #
    # Properties
    #
    ###################################################################


    def _getheader(self):
        # data loaded from a file should have a header
        warnings.warn('Image.header  may be deprecated if load_image returns an LPIImage')
        try:
            return self._header
        except AttributeError:
            raise AttributeError('Image created from arrays '
                                 'may not have headers.')
    def _setheader(self, header):
        warnings.warn('Image.header may be deprecated if load_image returns an LPIImage')
        self._header = header
    _doc['header'] = \
    """The file header dictionary for this image.  In order to update
    the header, you must first make a copy of the header, set the
    values you wish to change, then set the image header to the
    updated header.

    Example
    -------

    hdr = img.header
    hdr['slice_duration'] = 0.200
    hdr['descrip'] = 'My image registered with MNI152.'
    img.header = hdr
    
    """
    header = property(_getheader, _setheader, doc=_doc['header'])

    ###################################################################
    #
    # Constructor
    #
    ###################################################################

    def __init__(self, data, coordmap, metadata={}):
        """Create an `Image` object from array and `CoordinateMap` object.
        
        Images are most often created through the module functions load and
        fromarray.

        Parameters
        ----------
        data : A numpy.ndarray
        coordmap : An `AffineTransform` object
        metadata : dictionary
        
        See Also
        --------
        load : load `Image` from a file
        save : save `Image` to a file
        fromarray : create an `Image` from a numpy array

        """

        if data is None or coordmap is None:
            raise ValueError('expecting an array and CoordinateMap instance')

        if not isinstance(coordmap, AffineTransform):
            raise ValueError('coordmap must be an AffineTransform')

        # they don't inherit from each other anymore
        if isinstance(coordmap, CoordinateMap):
            raise ValueError('coordmap must be an AffineTransform')

        # self._data is an array-like object.  It must implement a subset of
        # array methods  (Need to specify these, for now implied in pyniftio)
        self._data = data

        self.coordmap = coordmap
        if self.axes.ndim != self._data.ndim:
            raise ValueError('the number of axes implied by the coordmap do not match the number of axes of the data')
        self.metadata = metadata

    ###################################################################
    #
    # Methods
    #
    ###################################################################


    def reordered_world(self, order=None):
        """
        Return a new Image with its coordmap
        having reordered output coordinates. This
        does not transpose the data.

        >>> cmap = AffineTransform.from_start_step('ijk', 'xyz', [1,2,3],[4,5,6])
        >>> im = Image(np.empty((30,40,50)), cmap)
        >>> im_reordered = im.reordered_world([2,0,1])
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

        >>> 

        """

        if order is None:
            order = range(self.ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.world.index(s) for s in order]

        new_cmap = self.coordmap.reordered_range(order)
        return self.__class__(self._data, new_cmap, metadata=self.metadata)

    def reordered_axes(self, order=None):
        """
        Return a new Image with its coordmap
        having reordered input coordinates. This
        transposes the data as well.

        >>> cmap = AffineTransform.from_start_step('ijk', 'xyz', [1,2,3],[4,5,6])
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
        >>> 

        """

        if order is None:
            order = range(self.ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.axes.index(s) for s in order]
        new_cmap = self.coordmap.reordered_domain(order)

        # Only transpose if we have to
        # so as to avoid calling self.get_data

        if order != range(self.ndim):
            new_data = np.transpose(self.get_data(), order)
        else:
            new_data = self._data
        return self.__class__(new_data, new_cmap,
                     metadata=self.metadata)

    def renamed_axes(self, **names_dict):
        """
        Return a new image with its axes renamed according
        to the dictionary.

        Parameters
        ----------

        img : Image

        names_dict : dictionary

        Returns
        -------

        newimg : Image
            An Image with the same data, having its axes renamed.

        >>> data = np.random.standard_normal((11,9,4))
        >>> im = Image(data, AffineTransform.from_params('ijk', 'xyz', np.identity(4)))
        >>> im_renamed = im.renamed_axes(i='slice')
        >>> print im_renamed.axes
        CoordinateSystem(coord_names=('slice', 'j', 'k'), name='domain', coord_dtype=float64)

        """

        newcmap = self.coordmap.renamed_domain(names_dict)
        return self.__class__(self._data, newcmap)

    def renamed_world(self, **names_dict):
        """
        Return a new image with its world coordinates renamed according
        to the dictionary.

        Parameters
        ----------

        img : Image

        names_dict : dictionary

        Returns
        -------

        newimg : Image
            An Image with the same data, having its world coordinates renamed.

        Examples
        --------

        >>> data = np.random.standard_normal((11,9,4))
        >>> im = Image(data, AffineTransform.from_params('ijk', 'xyz', np.identity(4)))
        >>> im_renamed_world = im.renamed_world(x='newx', y='newy')
        >>> print im_renamed_world.world
        CoordinateSystem(coord_names=('newx', 'newy', 'z'), name='range', coord_dtype=float64)


        """

        newcmap = self.coordmap.renamed_range(names_dict)
        return self.__class__(self._data, newcmap)


    def __setitem__(self, index, value):
        """Setting values of an image, set values in the data array."""
        self._data[index] = value

    def __array__(self):
        """Return data as a numpy array."""
        warnings.warn('may be deprecated, use get_data instead')
        return self.get_data()

    def get_data(self):
        """Return data as a numpy array."""
        return np.asarray(self._data)

    def __getitem__(self, slice_object):
        """
        Slicing an image returns an Image.
        Just calls the function subsample.
        """
        warnings.warn('slicing Images is deprecated, use subsample instead')
        return subsample(self, slice_object)

    def __eq__(self, other):
        return (    isinstance(other, self.__class__)
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


class SliceMaker(object):
    """
    This class just creates slice objects to be used
    in resampling images. It only has a __getitem__ method
    that returns its argument.

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
    """
    Subsample an image. 

    Parameters
    ----------

    img: Image

    slice_object: int, slice or [slice]
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
    >>> from nipy.testing import funcfile, assert_almost_equal
    >>> assert_almost_equal(frame3.get_data(), im.get_data()[:,:,:,3])

    """
    data = img.get_data()[slice_object]
    g = ArrayCoordMap(img.coordmap, img.shape)[slice_object]
    coordmap = g.coordmap
    if coordmap.function_domain.ndim > 0:
        return img.__class__(data, coordmap, metadata=img.metadata)
    else:
        return data

def fromarray(data, innames, outnames, coordmap=None):
    """Create an image from a numpy array.

    Parameters
    ----------
    data : numpy array
        A numpy array of three dimensions.
    names : a list of axis names
    coordmap : A `CoordinateMap`
        If not specified, an identity coordinate map is created.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    save : function for saving images

    """

    ndim = len(data.shape)
    if not coordmap:
        coordmap = AffineTransform.from_start_step(innames,
                                                   outnames,
                                                   (0.,)*ndim,
                                                   (1.,)*ndim)
                                          
    return Image(data, coordmap)

def rollaxis(img, axis, inverse=False):
    """
    Roll the specified axis backwards, until it lies in the first position.

    It also reorders the world coordinates by the same ordering.
    This is done to preserve a diagonal affine matrix if image.affine
    is diagonal. It also makes it possible to unambiguously specify
    an axis to roll along in terms of either a world name (i.e. 'z')
    or an axis name (i.e. 'slice').

    Parameters
    ----------

    img : Image
        Image whose axes and world coordinates are to be reordered
        by rolling.

    axis : str or int
        Axis to be rolled, can be specified by name or 
        as an integer.

    inverse : bool, optional
        If inverse is True, then axis must be an integer and the first axis
        is returned to the position axis.

    Returns
    -------

    newimg : An Image with reordered axes and world coordinates.

    >>> data = np.zeros((30,40,50,5))
    >>> affine_transform = AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1]))
    >>> im = Image(data, affine_transform)
    >>> im_t_first = rollaxis(im, 't')
    >>> np.diag(im_t_first.affine)
    array([ 4.,  1.,  2.,  3.,  1.])
    >>> im_t_first.shape
    (5, 30, 40, 50)

    """
    if axis not in [-1] + range(img.axes.ndim) + list(img.axes.coord_names) + list(img.world.coord_names):
        raise ValueError('axis must be an axis number, -1, an axis name or a world name')

    # Find out which index axis corresonds to

    if inverse and type(axis) != type(0):
        raise ValueError('if carrying out inverse rolling, axis must be an integer')

    in_index = out_index = -1
    if type(axis) == type(''):
        try:
            in_index = img.axes.index(axis)
        except:
            pass
        try:
            out_index = img.world.index(axis)
        except:
            pass

        if in_index > 0 and out_index > 0 and in_index != out_index:
            raise ValueError('ambiguous choice of axis -- it exists both in as an axis name and a world name')
        if in_index >= 0:
            axis = in_index
        else:
            axis = out_index

    if axis == -1:
        axis += img.axes.ndim

    if not inverse:
        order = range(img.ndim)
        order.remove(axis)
        order.insert(0, axis)
    else:
        order = range(img.ndim)
        order.remove(0)
        order.insert(axis, 0)

    return img.reordered_axes(order).reordered_world(order)

def synchronized_order(img, target_img,
                       axes=True,
                       world=True):
    """
    Take an Image, and reorder its world and
    axes to match target_img.

    Parameters
    ----------

    img : Image

    target_img : Image

    axes : bool, optional
        If True, synchronize the order of the axes.

    world : bool, optional
        If True, synchronize the order of the world coordinates.

    Returns
    -------

    newimg : Image
        An Image satisfying newimg.axes == target.axes (if axes == True), 
        newimg.world == target.world (if world == True).
    
    Examples
    --------

    >>> data = np.random.standard_normal((3,4,7,5))
    >>> im = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    >>> im_scrambled = im.reordered_axes('iljk').reordered_world('txyz')
    >>> im == im_scrambled
    False
    >>> im_unscrambled = synchronized_order(im_scrambled, im)
    >>> im == im_unscrambled
    True
    >>> 

    >>> # the images don't have to be the same shape
    >>> 
    >>> data2 = np.random.standard_normal((3,11,9,4))
    >>> im2 = Image(data, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1])))
    >>> 
    >>> im_scrambled2 = im2.reordered_axes('iljk').reordered_world('xtyz')
    >>> im_unscrambled2 = synchronized_order(im_scrambled2, im)
    >>> 
    >>> print im_unscrambled2.coordmap == im.coordmap
    True
    >>> 
    >>> # or have the same coordmap
    >>> 
    >>> data3 = np.random.standard_normal((3,11,9,4))
    >>> im3 = Image(data3, AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,9,3,-2,1])))
    >>> 
    >>> im_scrambled3 = im3.reordered_axes('iljk').reordered_world('xtyz')
    >>> im_unscrambled3 = synchronized_order(im_scrambled3, im)
    >>> 
    >>> print im_unscrambled3.axes == im.axes
    True
    >>> print im_unscrambled3.world == im.world
    True
    >>> im_unscrambled4 = synchronized_order(im_scrambled3, im, axes=False)
    >>> print im_unscrambled4.axes == im.axes
    False
    >>> print im_unscrambled4.axes == im_scrambled3.axes
    True
    >>> print im_unscrambled4.world == im.world
    True

    """
    # Caution, we can't just use
    # target_img.world because it's always 3-dimensional
    # if isinstance(target_img, LPIImage)

    target_axes = target_img.axes # = target_img.coordmap.function_domain
    target_world = target_img.coordmap.function_range # not always = target_image.world
    if axes:
        img = img.reordered_axes(target_axes.coord_names)
    if world:
        img = img.reordered_world(target_world.coord_names)
    return img

