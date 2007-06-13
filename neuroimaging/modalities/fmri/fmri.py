import numpy as N


from neuroimaging.core.api import Image, CoordinateSystem, SamplingGrid, \
     ParcelIterator, SliceIterator, SliceParcelIterator, IteratorItem, \
     Mapping, Affine


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

    def __init__(self, _image, **keywords):
        """
        :Parameters:
            `_image` : `FmriImage` or `Image` or ``string`` or ``array``
                The object to create this Image from. If an `Image` or ``array``
                are provided, their data is used. If a string is given it is treated
                as either a filename or url.
            `keywords` : dict
                Passed through as keyword arguments to `core.api.Image.__init__`
        """
        Image.__init__(self, _image, **keywords)
        self.frametimes = keywords.get('frametimes', None)
        self.slicetimes = keywords.get('slicetimes', None)

        self.grid = FmriSamplingGrid(self.grid.shape, self.grid.mapping,
                                     self.grid.input_coords,
                                     self.grid.output_coords)
        if self.grid.isproduct():
            ndim = len(self.grid.shape)
            n = self.grid.input_coords.axisnames()[:ndim]
            try:
                d = n.index('time')
            except ValueError:
                raise ValueError, "FmriImage expecting a 'time' axis, got %s" % n
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
