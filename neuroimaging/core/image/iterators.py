"""
The image iterator module.

This module contains classes which allow for iteration over Image
objects in a number of different ways. Each iterator follows a common
interface defined by the Iterator class.

Iterators can be used in two different modes, read-only and read-write. In
read-only mode, iterating over the iterator returns the actual data from
the Image. In read-write mode iterating over the iterator returns an
IteratorItem object. This has a get() and set() method which can be used
to read and write values from and to the image. The iterator mode is controlled
by the keyword argument mode in the Iterator constructor.
"""

__docformat__ = 'restructuredtext'

import operator

import numpy as N

class Iterator(object):
    """ The base class for image iterators.

    This is an abstract class which requires the _next() method
    to be overridden for it to work.
    """
    
    def __init__(self, img, mode='r'):
        """
        Create an `Iterator` for an image

        :Parameters:
            img : `api.Image`
                The image to be iterated over
            mode : ``string            ``
                The mode to run the iterator in.
                    'r' - read-only (default)
                    'w' - read-write
        """
        self.set_img(img)
        self.mode = mode
        self.item = NotImplemented

    def __iter__(self):
        """        
        Use this `Iterator` as a python iterator.
        
        :Returns: ``self``
        """
        return self
    
    def next(self):
        """
        Return the next item from the iterator.

        If in read-only mode, this will be a slice of the image.
        If in read-write mode, this will be an `IteratorItem` object.
        """
        self.item = self._next()
        if self.mode == 'r':
            return self.item.get()
        else:
            return self.item
    
    def _next(self):
        """
        Do the hard work of generating the next item from the iterator.

        :Returns: `IteratorItem`

        :raises NotImplementedError: This method must be overriden by the
            subclasses of `Iterator`.
        """
        raise NotImplementedError

    def set_img(self, img):
        """
        Setup the iterator to have a given image.

        :Parameters:
            img : `api.Image`
                The new image for the iterator
        """
        self.img = img

    def copy(self, img):
        """
        Create a copy of this iterator for a new image.
        The new iterator starts from the beginning, it does not get
        initialised to the current position of the original iterator.

        :Parameters:
            img : `api.Image`
                The image to be used with the new iterator
        """
        return self.__class__(img, mode=self.mode)

class IteratorItem(object):
    """
    This class provides the interface for objects returned by the
    `Iterator`.`_next`() method. This is the class which actually
    interacts with the image data itself.
    """

    def __init__(self, img, slice_):
        """ Create the `IteratorItem` for a given item and slice

        :Parameters:
            img : `api.Image`
                The image being iterated over.
            slice_ : ``slice``
                TODO
        """
        self.img = img
        self.slice = slice_

    def get(self):
        """
        Return the slice of the image.
        """
        return self.img[self.slice]

    def set(self, value):
        """
        Set the value of the slice of the image.
        """
        self.img[self.slice] = value


class SliceIterator(Iterator):
    """
    This class provides an iterator for iterating over n axes
    of the image, returning ndim-n dimensional slices of the image at each
    iteration.
    """

    def __init__(self, img, axis=0, mode='r'):
        """
        :Parameters:
            img : `api.Image`
                The image being iterated over.
            axis : ``int`` or ``[int]``
                The index of the axis (or axes) to be iterated over. If a list
                is supplied, the axes are iterated over slowest to fastest.
            mode : ``string``
                The mode to run the iterator in.
                    'r' - read-only (default)
                    'w' - read-write
        """
        try:
            # we get given axes slowest changing to fastest, but we want
            # to store them the other way round.
            self.axis = list(axis)[::-1]
        except TypeError:
            self.axis = [axis]
        self.n = 0
        Iterator.__init__(self, img, mode)


    def set_img(self, img):
        """
        Setup the iterator to have a given image.

        :Parameters:
            img : `api.Image`
                The new image for the iterator
        """
        Iterator.set_img(self, img)
        if img is not None:
            self.shape = self.img.shape

            # the total number of iterations to be made
            self.max = N.product(N.asarray(self.shape)[self.axis])

            # calculate the 'divmod' paramter which is used to work out
            # which index to use to use for each axis during iteration
            mods = N.cumprod(N.asarray(self.shape)[self.axis])            
            divs = [1] + list(mods[:-1])
            self.divmod = zip(self.axis, divs, mods)

            # set up a full set of slices for the image, to be modified
            # at each iteration
            self.slices = [slice(0, shape) for shape in self.shape]


    def _next(self):
        """
        Do the hard work of generating the next item from the iterator.

        :Returns: `SliceIteratorItem`
        """
        if self.n >= self.max:
            raise StopIteration

        for (axis, div, mod) in self.divmod:
            x = self.n / div % mod
            self.slices[axis] = slice(x, x+1)

        self.n += 1
        return SliceIteratorItem(self.img, self.slices)


    def copy(self, img):
        """
        Create a copy of this iterator for a new image.
        The new iterator starts from the beginning, it does not get
        initialised to the current position of the original iterator.

        :Parameters:
            img : `api.Image`
                The image to be used with the new iterator
        """
        return self.__class__(img, axis=self.axis, mode=self.mode)



class SliceIteratorItem(IteratorItem):
    """
    A class for objects returned by L{SliceIterator}s
    """

    def get(self):
        """
        Return the slice of the image.

        This calls the squeeze method on the array before returning to remove
        any redundant dimensions.
        """
        return self.img[self.slice].squeeze()

    def set(self, value):
        """
        Set the value of the slice of the image.
        """
        if isinstance(value, N.ndarray):
            value = value.reshape(self.img[self.slice].shape)
        self.img[self.slice] = value


class ParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `ParcelIterator`\ s
    """

    def __init__(self, img, slice_, label):
        """
        :Parameters:
            img : `api.Image`
                The image being iterated over.
            slice_ : ``slice``
                TODO
            label : ``int`` or ``tuple`` of ``int``
                TODO
        """
        IteratorItem.__init__(self, img, slice_)
        self.label = label

    def get(self):
        """
        Return the slice of the image.
        """
        self.slice = self.slice.reshape(self.img.shape)
        return self.img[self.slice]

    def set(self, value):        
        """
        Set the value of the slice of the image.
        """
        self.slice = self.slice.reshape(self.img.shape)
        self.img[self.slice] = value



class ParcelIterator(Iterator):
    """
    This class is used to iterate over different regions of an image.
    A parcelmap is used to define the regions and a parcelseq is used
    to define the order in which the regions are iterated over. Each iteration
    returns a 1 dimensional array containing the values of the image in the
    specified region.

    Example
    -------

    >>> import numpy as N
    >>> from neuroimaging.core.api import ParcelIterator
    >>> img = N.arange(3*3)
    >>> img = img.reshape((3, 3))
    >>> parcelmap = [[1,2,1],
    ...       [3,4,3],
    ...       [1,2,1]]
    >>> parcelseq = [(1,), (2,), (3,), (4,), (1,4), (2,3,4)]
    >>> for x in ParcelIterator(img, parcelmap, parcelseq):
    ...     print x
    ... 
    [0 2 6 8]
    [1 7]
    [3 5]
    [4]
    [0 2 4 6 8]
    [1 3 4 5 7]
    >>>
    """

    iterator_item = ParcelIteratorItem
    
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        """
        :Parameters:
            image : `api.Image`
                The image to be iterated over
            parcelmap : ``[int]``
                This is an int array of the same shape as img.
                The different values of the array define different regions in
                the image. For example, all the 0s define a region, all the 1s
                define another region, etc.           
            parcelseq : ``[int]`` or ``[(int, int, ...)]``
                This is an array of integers or tuples of integers, which
                define the order to iterate over the regions. Each element of
                the array can consist of one or more different integers. The
                union of the regions defined in parcelmap by these values is
                the region taken at each iteration. If parcelseq is None then
                the iterator will go through one region for each number in
                parcelmap.
            mode : ``string``
                The mode to run the iterator in.
                    'r' - read-only (default)
                    'w' - read-write                
        """
        Iterator.__init__(self, img, mode)
        self.parcelmap = N.asarray(parcelmap)
        self._prep_seq(parcelseq)


    def _prep_seq(self, parcelseq):
        """
        This method does some preprocessing on the `parcelseq`. It can be
        overrided to suit the needs of sub classes.

        In this case it creates a parcelseq from the parcelmap if none
        was supplied. Otherwise it goes through and converts lone integers
        in the list to tuples, as this is what is needed in _get_wherelabel.
        """
        if parcelseq is None:
            parcelseq = N.unique(self.parcelmap.flat)
        self.parcelseq = list(parcelseq)
        for i, label in enumerate(self.parcelseq):
            try:
                len(label)
            except:
                label = (label,)
            self.parcelseq[i] = label


    def __iter__(self):
        """
        :Returns: ``self``
        """
        self._labeliter = iter(self.parcelseq)
        return self
    
    def _next(self):
        """
        Do the hard work of generating the next item from the iterator.
        """
        wherelabel, label = self._get_wherelabel()
        return self.iterator_item(self.img, wherelabel, label)

    def _get_wherelabel(self):
        label = self._labeliter.next()
        wherelabel = reduce(operator.or_,
          [N.equal(self.parcelmap, lbl) for lbl in label])
        return wherelabel, label


    def copy(self, img):
        """
        Create a copy of this iterator for a new image.
        The new iterator starts from the beginning, it does not get
        initialised to the current position of the original iterator.

        :Parameters:
            `img` : `api.Image`
                The image to be used with the new iterator

        :Returns: `self.__class__`
        """
        return self.__class__(img, self.parcelmap, self.parcelseq,
                              mode=self.mode)


class SliceParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `SliceParcelIterator`\ s
    """

    def __init__(self, img, slice_, label, i):
        """
        :Parameters:
            img : `api.Image`
                The image being iterated over.
            slice_ : ``slice``
                TODO
            label : ``int`` or ``tuple`` of ``int``
                TODO
            i : TODO
                TODO
        """    
        IteratorItem.__init__(self, img, slice_)
        self.label = label
        self.i = i

    def get(self):
        """
        Return the slice of the image.
        """
        return self.img[self.i, self.slice]

    def set(self, value):
        """
        Set the value of the slice of the image.
        """
        self.img[self.i, self.slice] = value


class SliceParcelIterator(ParcelIterator):
    """
    TODO
    """
    
    iterator_item = SliceParcelIteratorItem

    def __init__(self, img, parcelmap, parcelseq, mode='r'):
        """
        :Parameters:
            img : `api.Image`
                The image being iterated over.
            parcelmap : ``[int]``
                This is an int array of the same shape as img.
                The different values of the array define different regions in
                the image. For example, all the 0s define a region, all the 1s
                define another region, etc.           
            parcelseq : ``[int]`` or ``[(int, int, ...)]``
                This is an array of integers or tuples of integers, which
                define the order to iterate over the regions. Each element of
                the array can consist of one or more different integers. The
                union of the regions defined in parcelmap by these values is
                the region taken at each iteration. If parcelseq is None then
                the iterator will go through one region for each number in
                parcelmap.                
            mode : ``string``
                The mode to run the iterator in.
                    'r' - read-only (default)
                    'w' - read-write
        """
        ParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.i = 0
        self.max = len(self.parcelseq)

    def _prep_seq(self, parcelseq):
        if parcelseq is None:
            raise ValueError, "parcelseq cannot be None"
        ParcelIterator._prep_seq(self, parcelseq)

    def _next(self):
        """
        Do the hard work of generating the next item from the iterator.
        """
        if self.i >= self.max:            
            raise StopIteration
        wherelabel, label = self._get_wherelabel()

        ret = self.iterator_item(self.img, wherelabel[self.i], label,
                                      self.i)
        self.i += 1
        return ret
    


class ImageSequenceIterator(object):
    """
    Take a sequence of `image.Image`\ s, and create an iterator whose next method
    returns array with shapes (len(imgs),) + self.imgs[0].next().shape Very
    useful for voxel-based methods, i.e. regression, one-sample t.
    """
    def __init__(self, imgs):
        """
        :Parameters:
            imgs : ``[`image.Image`]``
                The sequence of images to iterate over
        """
        self.imgs = imgs
        self.iters = None
        iter(self)

    def __iter__(self): 
        """ Return self as an iterator. """
        self.iters = [img.slice_iterator() for img in self.imgs]
        return self

    def next(self):
        """ Return the next iterator value. """
        val = [it.next() for it in self.iters]
        return N.array(val, N.float64)



def test_suite(level=1):
    import doctest
    return doctest.DocTestSuite(eval(__name__))
