from neuroimaging.core.api import ParcelIterator, SliceParcelIterator, \
  IteratorItem

class fMRIParcelIterator(ParcelIterator):
    """
    This class works in much the same way as the `ParcelIterator` except
    that ...TODO
    """
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
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
        self.iterator_item = fMRIParcelIteratorItem
    

class fMRIParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `fMRIParcelIterator`\ s
    """

    def __init__(self, img, slice_, label):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
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
        self.slice = self.slice.reshape(self.img.shape[1:])
        return self.img[:, self.slice]

    def set(self, value):        
        """
        Set the value of the slice of the image.
        """
        self.slice = self.slice.reshape(self.img.shape[1:])
        self.img[:, self.slice] = value


class fMRISliceParcelIterator(SliceParcelIterator):
    """
    TODO
    """
    
    def __init__(self, img, parcelmap, parcelseq, mode='r'):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image to be iterated over
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
        SliceParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRISliceParcelIteratorItem
    

class fMRISliceParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `fMRISliceParcelIterator`\ s
    """

    def __init__(self, img, slice_, label, i):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
            slice_ : TODO
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
        self.slice = self.slice.reshape(self.img.shape[2:])
        return self.img[:, self.i, self.slice]

    def set(self, value):
        """
        Set the value of the slice of the image.
        """
        self.slice = self.slice.reshape(self.img.shape[2:])
        self.img[:, self.i, self.slice] = value


