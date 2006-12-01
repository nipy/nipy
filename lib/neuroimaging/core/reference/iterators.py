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

#
# This is demonstration code to help thrash out ideas for the new iterator
# proposal. This will most likely change and shouldn't be used. See the
# end of the file for example usage.
#
# See http://projects.scipy.org/neuroimaging/ni/wiki/ImageIterators

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

        @param img: The image to be used with the new iterator
        @type img: L{Image}
        """
        iterator = self.__class__(img)
        self._copy_to(iterator)
        iterator.set_img(img)
        return iterator

    def _copy_to(self, iterator):
        """
        This method handles custom requirements of subclasses of Iterator.
        """
        iterator.mode = self.mode

class IteratorItem(object):

    def __init__(self, img, slice_):
        self.img = img
        self.slice = slice_

    def get(self):
        return self.img[self.slice]

    def set(self, value):
        self.img[self.slice] = value


class SliceIterator(Iterator):

    def __init__(self, img, mode='r', axis=0, step=1):
        try:
            self.axis = list(axis)[::-1]
        except TypeError:
            self.axis = [axis]
        self.n = 0
        self.step = step
        Iterator.__init__(self, img, mode)


    def set_img(self, img):
        Iterator.set_img(self, img)
        if img is not None:
            self.shape = self.img.shape
            self.max = N.product(N.asarray(self.shape)[self.axis])
            mods = N.cumprod(N.asarray(self.shape)[self.axis])            
            divs = [1] + list(mods[:-1])
            self.divmod = zip(divs, mods)
            self.slices = N.asarray([slice(0, shape, self.step) for shape in self.shape])

    def _next(self):
        if self.n >= self.max:
            raise StopIteration

        for ax, (div, mod) in zip(self.axis, self.divmod):
            self.slices[ax] = slice((self.n / div) % mod,
                                    ((self.n / div) % mod) + 1, 1)

        ret = SliceIteratorItem(self.img, list(self.slices))
        self.n += 1
        return ret


    def _copy_to(self, iterator):
        Iterator._copy_to(self, iterator)
        iterator.axis = self.axis
        iterator.step = self.step
        iterator.n = self.n



class SliceIteratorItem(IteratorItem):

    def get(self):
        return self.img[self.slice].squeeze()

    def set(self, value):
        if type(value) == N.ndarray:
            value = value.reshape(self.img[self.slice].shape)
        self.img[self.slice] = value



class ParcelIterator(Iterator):

    
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        Iterator.__init__(self, img, mode)
        self.parcelmap = N.asarray(parcelmap)
        self._prep_seq(parcelseq)

        self.iterator_item = ParcelIteratorItem

    def _prep_seq(self, parcelseq):
        if parcelseq is not None: 
            self.parcelseq = tuple(parcelseq)
        else:
            self.parcelseq = N.unique(self.parcelmap.flat)

    def __iter__(self):
        self._labeliter = iter(self.parcelseq)
        return self
    
    def _next(self):
        wherelabel, label = self._get_wherelabel()
        return self.iterator_item(self.img, wherelabel, label)

    def _get_wherelabel(self):
        label = self._labeliter.next()
        try:
            len(label)
        except:
            label = (label,)


        wherelabel = reduce(operator.or_,
          [N.equal(self.parcelmap, lbl) for lbl in label])
        return wherelabel, label

    def copy(self, img):
        iterator = self.__class__(img, self.parcelmap, self.parcelseq)
        self._copy_to(iterator)
        iterator.set_img(img)
        return iterator


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
        self.parcelseq = [tuple(ps) for ps in parcelseq]

    def _next(self):
        if self.i >= self.max:            
            raise StopIteration
        wherelabel, label = self._get_wherelabel()

        ret = self.iterator_item(self.img, wherelabel[self.i], label,
                                      self.i)
        self.i += 1
        return ret
    

    def _copy_to(self, iterator):
        ParcelIterator._copy_to(self, iterator)
        iterator.i = self.i


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



