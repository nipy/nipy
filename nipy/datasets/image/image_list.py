from copy import copy

import numpy as np

from nipy.core.image.image import Image


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
        >>> from nipy.core.api import Image, ImageList
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList(funcim)
        >>> sublist = ilist[2:5]
        
        Slicing an ImageList returns a new ImageList

        >>> isinstance(sublist, ImageList)
        True

        Indexing an ImageList returns a new Image

        >>> newimg = ilist[2]
        >>> isinstance(newimg, Image)
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
        for im in images:
            if not hasattr(im, "coordmap"):
                raise ValueError("expecting each element of images "
                                 " to have a 'coordmap' attribute")
        self.list = images

    @classmethod
    def from_image(klass, image, axis=-1):
        if axis is None:
            raise ValueError('axis must be array axis no or -1')
        imlist = []
        coordmap = image.coordmap
        data = np.asarray(image)
        data = np.rollaxis(data, axis)
        imlist = [Image(dataslice, copy(coordmap))
                  for dataslice in data]
        return klass(imlist)

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

    def __array__(self):
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

    def __iter__(self):
        self._iter = iter(self.list)
        return self

    def next(self):
        return self._iter.next()

