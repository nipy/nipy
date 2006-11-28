#
# This is demonstration code to help thrash out ideas for the new iterator
# proposal. This will most likely change and shouldn't be used. See the
# end of the file for example usage.
#
# See http://projects.scipy.org/neuroimaging/ni/wiki/ImageIterators

import operator

import numpy as N

class Iterator(object):
    """ The base class for image iterators. """
    
    def __init__(self, img, mode='r'):
        self.set_img(img)
        self.mode = mode
        self.item = NotImplemented

    def __iter__(self):
        return self
    
    def next(self):
        self.item = self._next()
        if self.mode == 'r':
            return self.item.get()
        else:
            return self.item
    
    def _next(self):
        raise NotImplementedError

    def set_img(self, img):
        self.img = img

    def copy(self, img):
        it = self.__class__(img)
        self._copy_to(it)
        it.set_img(img)
        return it

    def _copy_to(self, it):
        it.mode = self.mode

class IteratorItem(object):

    def __init__(self, img, slice):
        self.img = img
        self.slice = slice

    def get(self):
        return self.img[self.slice]

    def set(self, value):
        self.img[self.slice] = value


class SliceIterator(Iterator):

    def __init__(self, img, mode='r', axis=0, step=1):
        self.axis = axis
        self.n = 0
        self.step = step
        Iterator.__init__(self, img, mode)

    def set_img(self, img):
        Iterator.set_img(self, img)
        if img is not None:
            self.shape = self.img.shape
            self.max = self.shape[self.axis]


    def _next(self):
        if self.n >= self.max:
            raise StopIteration

        slices = [slice(0, shape, self.step) for shape in self.shape]
        slices[self.axis] = slice(self.n, self.n+1, 1)
        ret = SliceIteratorItem(self.img, slices)
        self.n += 1
        return ret


    def _copy_to(self, it):
        Iterator._copy_to(self, it)
        it.axis = self.axis
        it.step = self.step
        it.n = self.n



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
        it = self.__class__(img, self.parcelmap, self.parcelseq)
        self._copy_to(it)
        it.set_img(img)
        return it


class ParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice, label):
        IteratorItem.__init__(self, img, slice)
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

    def __init__(self, img, slice, label):
        IteratorItem.__init__(self, img, slice)
        self.label = label

    def get(self):
        self.slice = self.slice.reshape(self.img.shape[1:])
        return self.img[:,self.slice]

    def set(self, value):        
        self.slice = self.slice.reshape(self.img.shape[1:])
        self.img[:,self.slice] = value


class SliceParcelIterator(ParcelIterator):

    
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        ParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.i = 0
        self.max = len(self.parcelseq)
        self.iterator_item = SliceParcelIteratorItem

    def _prep_seq(self, parcelseq):
        if parcelseq is not None: 
            self.parcelseq = [tuple(ps) for ps in parcelseq]
        else:
            self.parcelseq = [N.unique(pm.flat) for pm in self.parcelmap]


    def _next(self):
        if self.i >= self.max:            
            raise StopIteration
        wherelabel, label = self._get_wherelabel()

        ret = self.iterator_item(self.img, wherelabel[self.i], label,
                                      self.i)
        self.i += 1
        return ret
    

    def _copy_to(self, it):
        ParcelIterator._copy_to(self, it)
        it.i = self.i


class SliceParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice, label, i):
        IteratorItem.__init__(self, img, slice)
        self.label = label
        self.i = i

    def get(self):
        return self.img[self.i,self.slice]

    def set(self, value):
        self.img[self.i,self.slice] = value


class fMRISliceParcelIterator(SliceParcelIterator):

    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        SliceParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRISliceParcelIteratorItem
    

class fMRISliceParcelIteratorItem(IteratorItem):

    def __init__(self, img, slice, label, i):
        IteratorItem.__init__(self, img, slice)
        self.label = label
        self.i = i

    def get(self):
        self.slice = self.slice.reshape(self.img.shape[2:])
        return self.img[:,self.i,self.slice]

    def set(self, value):
        self.slice = self.slice.reshape(self.img.shape[2:])
        self.img[:,self.i,self.slice] = value



def _main():
    from neuroimaging.core.image.image import Image
    img = Image(N.zeros((3, 4, 5)))

    # Slice along the 0th axis (default)
    print "slicing along axis=0"
    for s in SliceIterator(img):
        print s, s.shape


    # Slice along the 1st axis
    print "slicing along axis=1"
    for s in SliceIterator(img, axis=1):
        print s, s.shape

    # Slice along the 2nd axis, writing the z index
    # to each point in the image
    print "Writing z index values"
    z = 0
    for s in SliceIterator(img, axis=2, mode='w'):
        s.set(z)
        z += 1

    print img[:]

    # Slice using the image method interface
    print "slicing with .slice() method"
    for s in img.slices():
        print s, s.shape

    print "...and along axis=1"
    for s in img.slices(axis=1):
        print s, s.shape

    print "...and writing along the y axis"
    y = 0
    for s in img.slices(mode='w', axis=1):
        s.set(y)
        y += 1

    print img[:]

    for s in img.iterate(SliceIterator(None, axis=2)):
        print s


    B = Image(N.zeros((3, 4, 5)))
    B.from_slices(SliceIterator(img))
    print B[:]

    B = Image(N.zeros((4, 3, 5)))
    B.from_slices(SliceIterator(img), axis=1)
    print B[:]

    B = Image(N.zeros((3, 5, 4)))
    B.from_slices(SliceIterator(img, axis=2), axis=1)
    print B[:]
    

    B.from_iterator(img.iterate(SliceIterator(None, axis=2)),
                    SliceIterator(None, axis=1))
    print B[:]
    

    x = 0
    for s in B.slices(mode='w'):
        s.set(x)
        x += 1

    print B[:]
    print "========"

    parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), 0, 2)
    i = SliceParcelIterator(B, parcelmap, parcelseq)
    for n in i:
        print n


    y = 0
    for s in B.slices(axis=1, mode='w'):
        s.set(y)
        y += 1

    parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), 0, 2)
    i = SliceParcelIterator(B, parcelmap, parcelseq)
    for n in i:
        print n


if __name__ == '__main__':
    _main()
