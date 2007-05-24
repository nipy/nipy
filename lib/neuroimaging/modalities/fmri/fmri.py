import numpy as N


from neuroimaging.core.api import Image, CoordinateSystem, SamplingGrid, \
     ParcelIterator, SliceIterator, SliceParcelIterator, IteratorItem, \
     Mapping, Affine


class fMRISamplingGrid(SamplingGrid):
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
        Return a subgrid of fMRISamplingGrid. If the image's mapping is an
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



class fMRIImage(Image):
    """
    TODO
    """

    def __init__(self, _image, **keywords):
        """
        :Parameters:
            `_image` : `fMRIImage` or `Image` or ``string`` or ``array``
                The object to create this Image from. If an `Image` or ``array``
                are provided, their data is used. If a string is given it is treated
                as either a filename or url.
            `keywords` : dict
                Passed through as keyword arguments to `core.api.Image.__init__`
        """
        Image.__init__(self, _image, **keywords)
        self.frametimes = keywords.get('frametimes', None)
        self.slicetimes = keywords.get('slicetimes', None)

        self.grid = fMRISamplingGrid(self.grid.shape, self.grid.mapping,
                                     self.grid.input_coords,
                                     self.grid.output_coords)
        if self.grid.isproduct():
            ndim = len(self.grid.shape)
            n = self.grid.input_coords.axisnames()[:ndim]
            try:
                d = n.index('time')
            except ValueError:
                raise ValueError, "fMRIImage expecting a 'time' axis, got %s" % n
            transform = self.grid.mapping.transform[d, d]
            start = self.grid.mapping.transform[d, ndim]
            self.frametimes = start + N.arange(self.grid.shape[d]) * transform


    def frame(self, i, clean=False, **keywords):
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
        if clean: 
            data = N.nan_to_num(data)
        return Image(data, grid=self.grid.subgrid(i), **keywords)


    def slice_iterator(self, mode='r', axis=1):
        ''' Return slice iterator for this image. By default we iterate
        over the ``axis=1`` instead of ``axis=0`` as for the `Image` class.

        :Parameters:
            `axis` : int or [int]
                The index of the axis (or axes) to be iterated over. If a list
                is supplied, the axes are iterated over slowest to fastest.
            `mode` : string
                The mode to run the iterator in.
                'r' - read-only (default)
                'w' - read-write

        :Returns: `SliceIterator`
        '''
        return SliceIterator(self, mode=mode, axis=axis)

    def from_slice_iterator(self, other, axis=1):
        """
        Take an existing `SliceIterator` and use it to set the values
        in this image. By default we iterate over the ``axis=1`` for this image
        instead of ``axis=0`` as for the `Image` class.

        :Parameters:
            `other` : `SliceIterator`
                The iterator from which to take the values
            `axis` : int or [int]
                The axis to iterate over for this image.
                
        :Returns: ``None``
        """
        it = iter(SliceIterator(self, mode='w', axis=axis))
        for s in other:
            it.next().set(s)

class fMRIParcelIterator(ParcelIterator):
    """
    This class works in much the same way as the `ParcelIterator` except
    that ...TODO
    """
    def __init__(self, img, parcelmap, parcelseq=None, mode='r'):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
            parcelmap : ``[int]``
                This is an int array of the same shape as img.
                The different values of the array define different regions in
                the image. For example, all the 0s define a region, all the 1s
                define another region, etc.           
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
        ParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRIParcelIteratorItem
    

class fMRIParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `fMRIParcelIterator`\ s
    """

    def __init__(self, img, slice_, label):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
            slice_ : ``slice``
                TODO
            label : ``int`` or ``tuple`` of ``int``
                TODO
        """
        IteratorItem.__init__(self, img, slice_)
        self.label = label

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


class fMRISliceParcelIterator(SliceParcelIterator):
    """
    TODO
    """
    
    def __init__(self, img, parcelmap, parcelseq, mode='r'):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image to be iterated over
            parcelmap : ``[int]``
                This is an int array of the same shape as img.
                The different values of the array define different regions in
                the image. For example, all the 0s define a region, all the 1s
                define another region, etc.           
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
        SliceParcelIterator.__init__(self, img, parcelmap, parcelseq, mode)
        self.iterator_item = fMRISliceParcelIteratorItem
    

class fMRISliceParcelIteratorItem(IteratorItem):
    """
    A class for objects returned by `fMRISliceParcelIterator`\ s
    """

    def __init__(self, img, slice_, label, i):
        """
        :Parameters:
            img : `modalities.fmri.fMRIImage`
                The fmri image being iterated over.
            slice_ : TODO
                TODO
            label : ``int`` or ``tuple`` of ``int``
                TODO
            i : TODO
                TODO
        """
        IteratorItem.__init__(self, img, slice_)
        self.label = label
        self.i = i

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


