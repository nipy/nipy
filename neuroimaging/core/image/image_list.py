from numpy import asarray

class ImageList:

    def __init__(self, images=None):
        """
        
        A lightweight implementation of a list of images.

        Parameters
        ----------
        images: a sliceable object whose items are meant to be images,
                this is checked by asserting that each has a `coordmap` attribute

        >>> from numpy import asarray
        >>> from neuroimaging.testing import funcfile
        >>> from neuroimaging.core.api import Image, ImageList
        >>> from neuroimaging.io.api import load_image
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
        
        >>> asarray(sublist).shape
        (3, 20, 2, 20)

        >>> asarray(newimg).shape
        (20, 2, 20)

        """
        if images is not None:
            for im in images:
                if not hasattr(im, "coordmap"):
                    raise ValueError, "expecting each element of images to have a 'coordmap' attribute"
            self.list = images
        else:
            self.list = []

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
        >>> from numpy import asarray
        >>> from neuroimaging.testing import funcfile
        >>> from neuroimaging.core.api import ImageList
        >>> from neuroimaging.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList(funcim)
        >>> asarray(ilist).shape
        (20, 20, 2, 20)

        """

        return asarray([asarray(im) for im in self.list])

    def __iter__(self):
        self._iter = iter(self.list)
        return self

    def next(self):
        return self._iter.next()

