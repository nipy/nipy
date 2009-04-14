from numpy import asarray, arange, empty

from nipy.core.api import ImageList, Image, \
    CoordinateMap, Affine, CoordinateSystem

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
        >>> from nipy.modalities.fmri.api import FmriImageList, fromimage
        >>> from nipy.io.api import load_image
        
        >>> # fmrilist and ilist represent the same data

        >>> funcim = load_image(funcfile)
        >>> fmrilist = fromimage(funcim)
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

def fmri_generator(data, iterable=None):
    """
    This function takes an iterable object and returns a generator for

    [numpy.asarray(data)[:,item] for item in iterator]

    This is used to get time series out of a 4d fMRI image.

    Note that if data is an FmriImageList instance, there is more 
    overhead involved in calling numpy.asarray(data) than if
    data is in Image instance.

    If iterables is None, it defaults to range(data.shape[0])
    """
    data = asarray(data)
    if iterable is None:
        iterable = range(data.shape[1])
    for item in iterable:
        yield item, data[:,item]


def fromimage(fourdimage, volume_start_times=None, slice_times=None):
    """Create an FmriImageList from a 4D Image.

    Load an image from the file specified by ``url`` and ``datasource``.

    Note this assumes that the 4d Affine mapping is such that it
    can be made into a list of 3d Affine mappings

    Parameters
    ----------
    fourdimage: a 4D Image 
    volume_start_times: start time of each frame. It can be specified
                            either as an ndarray with len(images) elements
                            or as a single float, the TR. Defaults to
                            the diagonal entry of slowest moving dimension
                            of Affine transform
    slice_times: ndarray specifying offset for each slice of each frame

    TODO: watch out for reordering the output coordinates to (x,y,z,t)

    """
    images = []
    if not isinstance(fourdimage.coordmap, Affine):
        raise ValueError, 'fourdimage must have an Affine mapping'
    
    for im in [fourdimage[i] for i in range(fourdimage.shape[0])]:
        cmap = im.coordmap
        oa = cmap.output_coords.coord_names[1:]
        oc = CoordinateSystem(oa, "world")
        t = im.coordmap.affine[1:]
        a = Affine(t, im.coordmap.input_coords, oc)
        images.append(Image(asarray(im), a))

    if volume_start_times is None:
        volume_start_times = fourdimage.coordmap.affine[0,0]
        
    return FmriImageList(images=images, 
                         volume_start_times=volume_start_times,
                         slice_times=slice_times)
