import numpy as N


from neuroimaging.core.api import Image, CoordinateSystem, SamplingGrid, \
     Mapping, Affine
from neuroimaging.core.api import load_image as _load_image
from neuroimaging.core.image.iterators import SliceIterator

from neuroimaging.data_io.datasource import DataSource

from neuroimaging.modalities.fmri.iterators import FmriParcelIterator, \
     FmriSliceParcelIterator

class FmriSamplingGrid(SamplingGrid):
    """
    TODO
    """

    def __init__(self, shape, mapping, input_coords, output_coords):
        """
        :Parameters:
            `shape` : tuple of ints
                The shape of the grid
            `mapping` : `mapping.Mapping`
                The mapping between input and output coordinates
            `input_coords` : `CoordinateSystem`
                The input coordinate system
            `output_coords` : `CoordinateSystem`
                The output coordinate system
        """
        SamplingGrid.__init__(self, shape, mapping, input_coords, output_coords)

    def isproduct(self, tol = 1.0e-07):
        """Determine whether the affine transformation is 'diagonal' in time.
        
        :Parameters:
            `tol` : float
                TODO
                
        :Returns: ``bool``
        """

        if not isinstance(self.mapping, Affine):
            return False
        ndim = self.ndim
        t = self.mapping.transform
        offdiag = N.add.reduce(t[1:ndim,0]**2) + N.add.reduce(t[0,1:ndim]**2)
        norm = N.add.reduce(N.add.reduce(t**2))
        return N.sqrt(offdiag / norm) < tol


    def subgrid(self, i):
        """
        Return a subgrid of FmriSamplingGrid. If the image's mapping is an
        Affine instance and is 'diagonal' in time, then it returns a new
        Affine instance. Otherwise, if the image's mapping is a list of
        mappings, it returns the i-th mapping.  Finally, if these two do not
        hold, it returns a generic, non-invertible map in the original output
        coordinate system.
        
        :Parameters:
            `i` : int
                The index of the subgrid to return

        :Returns:
            `SamplingGrid`
        """
        incoords = self.input_coords.sub_coords()
        outcoords = self.output_coords.sub_coords()

        if self.isproduct():
            t = self.mapping.transform
            t = t[1:,1:]
            W = Affine(t)

        else:
            def _map(x, fn=self.mapping.map, **keywords):
                if len(x.shape) > 1:
                    _x = N.zeros((x.shape[0]+1,) + x.shape[1:])
                else:
                    _x = N.zeros((x.shape[0]+1,))
                _x[0] = i
                return fn(_x)
            W = Mapping(_map)

        _grid = SamplingGrid(self.shape[1:], W, incoords, outcoords)
        return _grid

class FmriImage(Image):
    """
    TODO
    """

    def __init__(self, data, grid, frametimes=None, slicetimes=None):
        """
        :Parameters:
            `_image` : `FmriImage` or `Image` or ``string`` or ``array``
                The object to create this Image from. If an `Image` or ``array``
                are provided, their data is used. If a string is given it is treated
                as either a filename or url.
            `keywords` : dict
                Passed through as keyword arguments to `core.api.Image.__init__`
        """
        Image.__init__(self, data, grid)
        self.frametimes = frametimes
        self.slicetimes = slicetimes

        self._grid = FmriSamplingGrid(self.grid.shape, self.grid.mapping,
                                      self.grid.input_coords,
                                      self.grid.output_coords)
        if self.grid.isproduct() and self.frametimes is None:
            ndim = len(self.grid.shape)
            n = self.grid.input_coords.axisnames()[:ndim]
            try:
                d = n.index('time')
            except ValueError:
                raise ValueError, "FmriImage expecting a 'time' axis, got %s" % n
            transform = self.grid.mapping.transform[d, d]
            start = self.grid.mapping.transform[d, ndim]
            self.frametimes = start + N.arange(self.grid.shape[d]) * transform


    def frame(self, i):
        """
        TODO
        
        :Parameters:
            `i` : int
                TODO
            `clean` : bool
                If true then ``nan_to_num`` is called on the data before creating the `Image`
            `keywords` : dict
                Pass through as keyword arguments to `Image`
                
        :Returns: `Image`
        """
        data = N.squeeze(self[slice(i,i+1)])
        return Image(data, grid=self.grid.subgrid(i))

def slice_iterator(img, axis=1, mode='r'):
    """Return slice iterator for this FmriImage
    
    Parameters
    ----------
    img : An `FmriImage` object
    axis : ``int`` or ``[int]``
        The index of the axis (or axes) to be iterated over. If a list
        is supplied the axes are iterated over slowest to fastest.
    mode : ``string``
        The mode to run the iterator in.
        'r' - read-only (default)
        'w' - read-write

    Returns
    -------
    iterator : A `SliceIterator` object.
    
    Examples
    --------

    >>> import numpy as np
    >>> from neuroimaging.core.image import image
    >>> from neuroimaging.data import MNI_file
    >>> img = image.load(MNI_file)
    >>> for slice_ in image.slice_iterator(img):
    ...     y = np.mean(slice_)
    
    >>> imgiter = image.slice_iterator(img)
    >>> slice_ = imgiter.next()

    """
    
    return SliceIterator(img, axis=axis, mode=mode)


def parcel_iterator(img, parcelmap, parcelseq=None, mode='r'):
    """
    Parameters
    ----------
    parcelmap : ``[int]``
        This is an int array of the same shape as self.
        The different values of the array define different regions.
        For example, all the 0s define a region, all the 1s define
        another region, etc.           
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
    
    return FmriParcelIterator(img, parcelmap, parcelseq, mode=mode)

def slice_parcel_iterator(img, parcelmap, parcelseq=None, mode='r'):
    """
    Parameters
    ----------
    parcelmap : ``[int]``
        This is an int array of the same shape as self.
        The different values of the array define different regions.
        For example, all the 0s define a region, all the 1s define
        another region, etc.           
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
    
    return FmriSliceParcelIterator(img, parcelmap, parcelseq, mode=mode)

def fromarray(data, grid=None, names=['time', 'zspace', 'yspace', 'xspace']):
    """Create an image from a numpy array.

    Parameters
    ----------
    data : numpy array
        A numpy array of three dimensions.
    grid : A `SamplingGrid`
        If not specified, a uniform sampling grid is created.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    save : function for saving images

    """

    if not grid:
        grid = SamplingGrid.from_start_step(shape=data.shape, 
                                            start=(0,)*data.ndim, 
                                            step=(1,)*data.ndim,
                                            names=names)
    return Image(data, grid)

def load(url, datasource=DataSource(), format=None, **keywords):
    """Load an FmriImage from the given url.

    Load an image from the file specified by ``url`` and ``datasource``.

    Parameters
    ----------
    url : string
        Should resolve to a complete filename path, possibly with the provided
        datasource.
    datasource : A `DataSource` object
        A datasource for the image to load.
    format : A `Format` object
        The file format to use when opening the image file.  If ``None``, the
        default, all supported formats are tried.
    keywords : Keyword arguments passed to `Format` initialization call.

    Returns
    -------
    image : An `Image` object
        If successful, a new `Image` object is returned.

    See Also
    --------
    save : function for saving images
    fromarray : function for creating images from numpy arrays

    Notes
    -----
    The raising of an exception can be misleading. If for example, a bad url 
    is given, it will appear as if that file's format has not been implemented.

    Examples
    --------

    >>> from neuroimaging.core.image import image
    >>> from neuroimaging.data import MNI_file
    >>> img = image.load(MNI_file)
    >>> img.shape
    (91, 109, 91)

    """
    im = _load_image(url, datasource=datasource, format=format, **keywords)
    return FmriImage(im[:], im.grid)
