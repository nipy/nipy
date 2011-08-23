# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from copy import copy

import numpy as np

from .image import Image, rollaxis as img_rollaxis
from ..reference.coordinate_map import (CoordinateSystem,
                                       AffineTransform, compose)

class ImageList(object):
    ''' Class to contain ND image as list of (N-1)D images '''

    def __init__(self, images=None):
        """
        A lightweight implementation of a list of images.

        Parameters
        ----------
        images : iterable
           a iterable and sliceable object whose items are meant to be images;
           this is checked by asserting that each has a `coordmap` attribute

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
                                 "to have a 'coordmap' attribute")
        self.list = images

    @classmethod
    def from_image(klass, image, axis=0):
        """ Create an image list from an image

        Parameters
        ----------
        image : object
            object with ``coordmap`` attribute
        axis : int
            axis of `image` that should become the axis indexed by the image
            list

        Returns
        -------
        ilist : ``ImageList`` instance
        """
        # Now, reorder the axes and reference
        image = img_rollaxis(image, axis)

        imlist = []
        coordmap = image.coordmap

        # We drop the first output coordinate of image's coordmap
        drop1st = np.identity(coordmap.ndims[1]+1)[1:]
        drop1st_domain = image.reference
        drop1st_range = CoordinateSystem(image.reference.coord_names[1:],
                                 name=image.reference.name,
                                 coord_dtype=image.reference.coord_dtype)
        drop1st_coordmap = AffineTransform(drop1st_domain, drop1st_range,
                                           drop1st)
        # And arbitrarily add a 0 for the first axis
        add0 = np.vstack([np.zeros(image.axes.ndim),
                          np.identity(image.axes.ndim)])
        add0_domain = CoordinateSystem(image.axes.coord_names[1:],
                                 name=image.axes.name,
                                 coord_dtype=image.axes.coord_dtype)
        add0_range = image.axes
        add0_coordmap = AffineTransform(add0_domain, add0_range,
                                        add0)

        coordmap = compose(drop1st_coordmap, image.coordmap, add0_coordmap)

        data = np.asarray(image)
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

