import numpy as np

class ImageList(object):
    ''' Class to contain ND image as list of (N-1)D images '''
    
    def __init__(self, images=None):
        """
        A lightweight implementation of a list of images.

        Parameters
        ----------
        images : iterable
           a iterable and sliceale object whose items are meant to be
           images, this is checked by asserting that each has a
           `coordmap` attribute

        >>> import numpy as np
        >>> from nipy.testing import funcfile
        >>> from nipy.core.api import BaseImage, ImageList
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList(funcim)
        >>> sublist = ilist[2:5]
        
        Slicing an ImageList returns a new ImageList

        >>> isinstance(sublist, ImageList)
        True

        Indexing an ImageList returns a new Image

        >>> newimg = ilist[2]
        >>> isinstance(newimg, BaseImage)
        True
        >>> isinstance(newimg, ImageList)
        False
        >>> np.asarray(sublist).shape
        (3, 2, 20, 20)
        >>> np.asarray(newimg).shape
        (2, 20, 20)

        """

        if images is None:
            self.list = []
            return
        if not hasattr(images, '__iter__'):
            raise ValueError('The images argument should be iterable')
        for im in images:
            if not hasattr(im, "get_transform"):
                raise ValueError("expecting each element of images "
                                 " to have a 'get_transform' method")
        self.list = images

    @classmethod
    def from_image(klass, image, axis=-1):
        if axis is None:
            raise ValueError('axis must be array axis no or -1')
        data = image.get_data()
        data = np.rollaxis(data, axis)
        imlist = [image.get_lookalike(dataslice) for dataslice in data]
        return klass(imlist)

    def get_data(self):
        """Return data in ndarray.  Called through numpy.array.
        
        Examples
        --------
        >>> import numpy as np
        >>> from nipy.testing import funcfile
        >>> from nipy.core.api import ImageList
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList(funcim)
        >>> np.asarray(ilist).shape
        (20, 2, 20, 20)

        """

        return np.asarray([np.asarray(im) for im in self.list])

    def __setitem__(self, index, value):
        """
        self.list[index] = value
        """
        
        self.list[index] = value

    def __getitem__(self, index):
        """
        self.list[index]
        """

        if type(index) is type(1):
            return self.list[index]
        else:
            return ImageList(images=self.list[index])

    def __getslice__(self, i, j):
        """
        Return another ImageList instance consisting with
        images self.list[i:j]
        """
        
        return ImageList(images=self.list[i:j])

    def __iter__(self):
        self._iter = iter(self.list)
        return self

    def next(self):
        return self._iter.next()

