from neuroimaging.core.api import ParcelIterator, SliceParcelIterator, \
  ParcelIteratorItem, SliceParcelIteratorItem


class fMRIParcelIteratorItem(ParcelIteratorItem):
    """
    A class for objects returned by `fMRIParcelIterator`\ s
    """

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


class fMRIParcelIterator(ParcelIterator):
    """
    This class works in much the same way as the `ParcelIterator` except
    that ...TODO
    """
    iterator_item = fMRIParcelIteratorItem


class fMRISliceParcelIteratorItem(SliceParcelIteratorItem):
    """
    A class for objects returned by `fMRISliceParcelIterator`\ s
    """

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


class fMRISliceParcelIterator(SliceParcelIterator):
    """
    TODO
    """
    iterator_item = fMRISliceParcelIteratorItem

