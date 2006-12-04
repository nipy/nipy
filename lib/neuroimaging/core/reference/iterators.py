"""
The image iterator module.

This module contains classes which allow for iteration over L{Image}
objects in a number of different ways. Each iterator follows a common
interface defined by the L{Iterator} class.

Iterators can be used in two different modes, read-only and read-write. In
read-only mode, iterating over the iterator returns the actual data from
the L{Image}. In read-write mode iterating over the iterator returns an
L{IteratorItem} object. This has a get() and set() method which can be used
to read and write values from and to the image. The iterator mode is controlled
by the keyword argument mode in the Iterator constructor.
"""

import operator

import numpy as N

class Iterator(object):
    """ The base class for image iterators.

    This is an abstract class which requires the _next() method
    to be overridden for it to work.
    """
    
    def __init__(self, img, mode='r'):
        """
        Create an Iterator for an image

        @param img: The image to be iterated over
        @oaram type: L{Image}
        @param mode: The mode to run the iterator in.
            'r' - read-only (default)
            'w' - read-write
        @param type: C{string}
        """
        self.set_img(img)
        self.mode = mode
        self.item = NotImplemented

    def __iter__(self):
        """        
        Use this L{Iterator} as a python iterator.
        """
        return self
    
    def next(self):
        """
        Return the next item from the iterator.

        If in read-only mode, this will be a slice of the image.
        If in read-write mode, this will be an L{IteratorItem} object.
        """
        self.item = self._next()
        if self.mode == 'r':
            return self.item.get()
        else:
            return self.item
    
    def _next(self):
        """
        Do the hard work of generating the next item from the iterator.

        This method must be overriden by the subclasses of Iterator.
        @rtype: L{IteratorItem}
        """
        raise NotImplementedError

    def set_img(self, img):
        """
        Setup the iterator to have a given image.

        @param img: The new image for the iterator
        @type img: L{Image}
        """
        self.img = img

    def copy(self, img):
        """
        Create a copy of this iterator for a new image.
        The new iterator starts from the beginning, it does not get
        initialised to the current position of the original iterator.

        @param img: The image to be used with the new iterator
        @type img: L{Image}
        """
        return self.__class__(img, mode=self.mode)

class IteratorItem(object):
    """
    This class provides the interface for objects returned by the
    L{Iterator._next()} method. This is the class which actually
    interacts with the image data itself.
    """

    def __init__(self, img, slice_):
        """ Create the IteratorItem for a given item and slice
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
        @param axis: The index of the axis (or axes) to be iterated over. If
            a list is supplied, the axes are iterated over slowest to fastest.
        @type axis: C{int} or C{list} of {int}.
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
            self.slices = [slice(0, shape, 1) for shape in self.shape]


    def _next(self):

        if self.n >= self.max:
            raise StopIteration

        for (ax, div, mod) in self.divmod:
            x = self.n / div % mod
            self.slices[ax] = slice(x, x+1, 1)

        self.n += 1
        return SliceIteratorItem(self.img, self.slices)


    def copy(self, img):
        """
        Create a copy of this iterator for a new image.
        The new iterator starts from the beginning, it does not get
        initialised to the current position of the original iterator.

        @param img: The image to be used with the new iterator
        @type img: L{Image}
        """
        return self.__class__(img, axis=self.axis, mode=self.mode)



class SliceIteratorItem(IteratorItem):

    def get(self):
        return self.img[self.slice].squeeze()

    def set(self, value):
        if type(value) == N.ndarray:
            value = value.reshape(self.img[self.slice].shape)
        self.img[self.slice] = value



class ParcelIterator(Iterator):
    """
    This class is used to iterate over different regions of an image.
    A C{parcelmap} is used to define the regions and a C{parcelseq} is used
    to define the order in which the regions are iterated over.

    >>> img = N.arange(3*3)
    >>> img = img.reshape((3, 3))
    >>> pm = [[1,2,1],
    ...       [3,4,3],
    ...       [1,2,1]]
    >>> ps = [(1,), (2,), (3,), (4,), (1,4), (2,3,4)]
    >>> for x in ParcelIterator(img, pm, ps):
    ...     print x
    ... 
    [0 2 6 8]
    [1 7]
    [3 5]
    [4]
    [0 2 4 6 8]
    [1 3 4 5 7]
    """
    
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        """

        @param parcelmap: This is an C{int} array of the same shape as C{img}.
           The different values of the array define different regions in the
           image. For example, all the 0s define a region, all the 1s define
           another region.
        @param parcelseq: This is an array of integers or tuples of integers,
           which define the order to iterate over the regions. Each tuple
           can consist of one or more different integers. The union of the
           regions defined in C{parcelmap} by these values is the region
           taken at each iteration.
        """
        Iterator.__init__(self, img, mode)
        self.parcelmap = N.asarray(parcelmap)
        self._prep_seq(parcelseq)

        self.iterator_item = ParcelIteratorItem

    def _prep_seq(self, parcelseq):
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
        self._labeliter = iter(self.parcelseq)
        return self
    
    def _next(self):
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

        @param img: The image to be used with the new iterator
        @type img: L{Image}
        """
        return self.__class__(img, self.parcelmap, self.parcelseq,
                              mode=self.mode)


class ParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice_, label):
        IteratorItem.__init__(self, img, slice_)
        self.label = label

    def get(self):
        self.slice = self.slice.reshape(self.img.shape)
        return self.img[self.slice]

    def set(self, value):        
        self.slice = self.slice.reshape(self.img.shape)
        self.img[self.slice] = value



class fMRIParcelIterator(ParcelIterator):
    """
    This class works in much the same way as the L{ParcelIterator} except
    that 
    """
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        ParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRIParcelIteratorItem
    

class fMRIParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice_, label):
        IteratorItem.__init__(self, img, slice_)
        self.label = label

    def get(self):
        self.slice = self.slice.reshape(self.img.shape[1:])
        return self.img[:,self.slice]

    def set(self, value):        
        self.slice = self.slice.reshape(self.img.shape[1:])
        self.img[:,self.slice] = value


class SliceParcelIterator(ParcelIterator):

    
    def __init__(self, img, parcelmap, parcelseq, mode='r'):
        ParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.i = 0
        self.max = len(self.parcelseq)
        self.iterator_item = SliceParcelIteratorItem

    def _prep_seq(self, parcelseq):
        if parcelseq is None:
            raise ValueError, "parcelseq cannot be None"
        ParcelIterator._prep_seq(self, parcelseq)

    def _next(self):
        if self.i >= self.max:            
            raise StopIteration
        wherelabel, label = self._get_wherelabel()

        ret = self.iterator_item(self.img, wherelabel[self.i], label,
                                      self.i)
        self.i += 1
        return ret
    


class SliceParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice_, label, i):
        IteratorItem.__init__(self, img, slice_)
        self.label = label
        self.i = i

    def get(self):
        return self.img[self.i,self.slice]

    def set(self, value):
        self.img[self.i,self.slice] = value


class fMRISliceParcelIterator(SliceParcelIterator):

    def __init__(self, img, parcelmap, parcelseq, mode='r'):
        SliceParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRISliceParcelIteratorItem
    

class fMRISliceParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice_, label, i):
        IteratorItem.__init__(self, img, slice_)
        self.label = label
        self.i = i

    def get(self):
        self.slice = self.slice.reshape(self.img.shape[2:])
        return self.img[:,self.i,self.slice]

    def set(self, value):
        self.slice = self.slice.reshape(self.img.shape[2:])
        self.img[:,self.i,self.slice] = value

