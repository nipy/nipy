from neuroimaging.core.api import ParcelIterator, SliceParcelIterator, \
  ParcelIteratorItem, SliceParcelIteratorItem


class FmriParcelIteratorItem(ParcelIteratorItem):
    """
    A class for objects returned by `FmriParcelIterator`\ s
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


class FmriParcelIterator(ParcelIterator):
    """
    This class works in much the same way as the `ParcelIterator` except
    that ...TODO
    """
    iterator_item = FmriParcelIteratorItem


class FmriSliceParcelIteratorItem(SliceParcelIteratorItem):
    """
    A class for objects returned by `FmriSliceParcelIterator`\ s
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


class FmriSliceParcelIterator(SliceParcelIterator):
    """
    TODO
    """
    iterator_item = FmriSliceParcelIteratorItem

