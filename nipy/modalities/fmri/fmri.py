import warnings

from numpy import asarray, arange, empty

from nipy.core.image.image import rollaxis as image_rollaxis
from nipy.core.api import ImageList, Image, \
    CoordinateMap, AffineTransform, CoordinateSystem

class FmriImageList(ImageList):
    """
    Class to implement image list interface for FMRI time series

    Allows metadata such as volume and slice times
    """

    def __init__(self, images=None, volume_start_times=None, slice_times=None):

        """
        A lightweight implementation of an fMRI image as in ImageList
        
        Parameters
        ----------
        images: a sliceable object whose items are meant to be images,
                this is checked by asserting that each has a `coordmap` attribute
        volume_start_times: start time of each frame. It can be specified
                            either as an ndarray with len(images) elements
                            or as a single float, the TR. Defaults
                            to arange(len(images)).astype(np.float)

        slice_times: ndarray specifying offset for each slice of each frame

        See Also
        --------
        nipy.core.image_list.ImageList

        >>> from numpy import asarray
        >>> from nipy.testing import funcfile
        >>> from nipy.io.api import load_image
        >>> # fmrilist and ilist represent the same data
        >>> funcim = load_image(funcfile)
        >>> fmrilist = FmriImageList.from_image(funcim)
        >>> ilist = FmriImageList(funcim)
        >>> print asarray(ilist).shape
        (20, 2, 20, 20)
        >>> print asarray(ilist[4]).shape
        (2, 20, 20)

        """
        ImageList.__init__(self, images=images)
        if volume_start_times is None:
            volume_start_times = 1.

        v = asarray(volume_start_times)
        if v.shape == (len(self.list),):
            self.volume_start_times = volume_start_times
        else:
            v = float(volume_start_times)
            self.volume_start_times = arange(len(self.list)) * v

        self.slice_times = slice_times

    def __getitem__(self, index):
        """
        If index is an index, return self.list[index], an Image
        else return an FmriImageList with images=self.list[index].
        
        """
        if type(index) is type(1):
            return self.list[index]
        else:
            return FmriImageList(images=self.list[index], 
                                 volume_start_times=self.volume_start_times[index],
                             slice_times=self.slice_times)

    def __setitem__(self, index, value):
        self.list[index] = value
        
    def __array__(self):
        v = empty((len(self.list),) + self.list[0].shape)
        for i, im in enumerate(self.list):
            v[i] = asarray(im)
        return v

    @classmethod
    def from_image(klass, fourdimage, volume_start_times=None, slice_times=None, axis='t'):
        """Create an FmriImageList from a 4D Image by
        extracting 3d images along the 't' axis.

        Parameters
        ----------
        fourdimage: a 4D Image 
        volume_start_times: start time of each frame. It can be specified
                            either as an ndarray with len(images) elements
                            or as a single float, the TR. Defaults to
                            the diagonal entry of slowest moving dimension
                            of Affine transform
        slice_times: ndarray specifying offset for each slice of each frame

        """
        if fourdimage.ndim != 4:
            raise ValueError('expecting a 4-dimensional Image')
        image_list = ImageList.from_image(fourdimage, axis='t')
        return klass(images=image_list.list, 
                     volume_start_times=volume_start_times,
                     slice_times=slice_times)


def fmri_generator(data, iterable=None):
    """
    This function takes an iterable object and returns a generator that
    looks like:

    [numpy.asarray(data)[:,item] for item in iterator]

    This can be used to get time series out of a 4d fMRI image, if and
    only if time varies across axis 0.

    Parameters
    ----------
    data : array-like
       object such that ``arr = np.asarray(data)`` returns an array of
       at least 2 dimensions.
    iterable : None or sequence
       seqence of objects that can be used to index array ``arr``
       returned from data.  If None, default is
       ``range(data.shape[1])``, in which case the generator will
       return elements  ``[arr[:,0], arr[:,1] ... ]``

    Notes
    -----
    If data is an ``FmriImageList`` instance, there is more overhead
    involved in calling ``numpy.asarray(data)`` than if data is an Image
    instance or an array.
    """
    warnings.warn('generator _assumes_ time as first axis in array; '
                  'this may well not be true for Images')
    data = asarray(data)
    if iterable is None:
        iterable = range(data.shape[1])
    for item in iterable:
        yield item, data[:,item]

