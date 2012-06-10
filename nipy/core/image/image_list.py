# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings

import numpy as np

from .image import Image, iter_axis, is_image
from ..reference.coordinate_map import (drop_io_dim, io_axis_indices, AxisError)


class ImageList(object):
    ''' Class to contain ND image as list of (N-1)D images '''

    def __init__(self, images=None):
        """
        An implementation of a list of images.

        Parameters
        ----------
        images : iterable
           an iterable object whose items are meant to be images; this is
           checked by asserting that each has a `coordmap` attribute and a
           ``get_data`` method.  Note that Image objects are not iterable by
           default; use the ``from_image`` classmethod or ``iter_axis`` function
           to convert images to image lists - see examples below for the latter.

        Examples
        --------
        >>> from nipy.testing import funcfile
        >>> from nipy.core.api import Image, ImageList, iter_axis
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> iterable_img = iter_axis(funcim, 't')
        >>> ilist = ImageList(iterable_img)
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
        (3, 17, 21, 3)
        >>> newimg.get_data().shape
        (17, 21, 3)
        """
        if images is None:
            self.list = []
            return
        images = list(images)
        if not all(is_image(im) for im in images):
                raise ValueError("Expecting each element of images to have "
                                 "the Image API")
        self.list = images

    @classmethod
    def from_image(klass, image, axis=None, dropout=True):
        """ Create an image list from an `image` by slicing over `axis`

        Parameters
        ----------
        image : object
            object with ``coordmap`` attribute
        axis : str or int
            axis of `image` that should become the axis indexed by the image
            list.
        dropout : bool, optional
            When taking slices from an image, we will leave an output dimension
            to the coordmap that has no corresponding input dimension.  If
            `dropout` is True, drop this output dimension.

        Returns
        -------
        ilist : ``ImageList`` instance
        """
        if axis is None:
            raise ValueError('Must specify image axis')
        # Get corresponding input, output dimension indices
        in_ax, out_ax = io_axis_indices(image.coordmap, axis)
        if in_ax is None:
            raise AxisError('No correspnding input dimension for %s' % axis)
        dropout = dropout and not out_ax is None
        if dropout:
            out_ax_name = image.reference.coord_names[out_ax]
        imlist = []
        for img in iter_axis(image, in_ax):
            if dropout:
                cmap = drop_io_dim(img.coordmap, out_ax_name)
                img = Image(img.get_data(), cmap, img.metadata)
            imlist.append(img)
        return klass(imlist)

    def __setitem__(self, index, value):
        """
        self.list[index] = value
        """
        self.list[index] = value

    def __len__(self):
        """ Length of image list
        """
        return len(self.list)

    def __getitem__(self, index):
        """
        self.list[index]
        """
        # Integer slices return elements
        if type(index) is type(1):
            return self.list[index]
        # List etc slicing return new instances of self.__class__
        return self.__class__(images=self.list[index])

    def get_list_data(self, axis=None):
        """Return data in ndarray with list dimension at position `axis`

        Parameters
        ----------
        axis : int
            `axis` specifies which axis of the output will take the role of the
            list dimension. For example, 0 will put the list dimension in the
            first axis of the result.

        Returns
        -------
        data : ndarray
            data in image list as array, with data across elements of the list
            concetenated at dimension `axis` of the array.

        Examples
        --------
        >>> from nipy.testing import funcfile
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList.from_image(funcim, axis='t')
        >>> ilist.get_list_data(axis=0).shape
        (20, 17, 21, 3)
        """
        if axis is None:
            raise ValueError('Must specify which axis of the output will take '
                             'the role of the list dimension, eg 0 will put '
                             'the list dimension in the first axis of the '
                             'result')
        img_shape = self.list[0].shape
        ilen = len(self.list)
        out_dim = len(img_shape) + 1
        if axis >= out_dim or axis < -out_dim:
            raise ValueError('I have only %d axes position, but axis %d asked '
                             'for' % (out_dim -1, axis))
        # tmp_shape is the shape of the output if axis is 0
        tmp_shape = (ilen,) + img_shape
        v = np.empty(tmp_shape)
        # first put the data in an array, with list dimension in the first axis
        for i, im in enumerate(self.list):
            v[i] = im.get_data() # get_data method of an image has no axis
        # then roll (and rock?) the axis to have axis in the right place
        if axis < 0:
            axis += out_dim
        res = np.rollaxis(v, 0, axis + 1)
        # Check we got the expected shape
        target_shape = img_shape[0:axis] + (ilen,) + img_shape[axis:]
        if target_shape != res.shape:
            raise ValueError('We were not expecting this shape')
        return res

    def __array__(self):
        """Return data in ndarray.  Called through numpy.array.

        Examples
        --------
        >>> from nipy.testing import funcfile
        >>> from nipy.io.api import load_image
        >>> funcim = load_image(funcfile)
        >>> ilist = ImageList.from_image(funcim, axis='t')
        >>> np.asarray(ilist).shape
        (20, 17, 21, 3)
        """
        """Return data as a numpy array."""
        warnings.warn('Please use get_list_data() instead - default '
                      'conversion to array will be deprecated',
                      DeprecationWarning,
                      stacklevel=2)
        return self.get_list_data(axis=0)

    def __iter__(self):
        self._iter = iter(self.list)
        return self

    def next(self):
        return self._iter.next()
