""" BaseImage class - wrapper for Image class to test changes to Image interface
"""

import os
import sys
import string, urllib, stat, ftplib, gzip, urllib2, copy
import types

import numpy as N
import enthought.traits as traits

import neuroimaging as ni
from neuroimaging.reference import grid, axis, mapping
from neuroimaging.reference.grid import IdentityGrid
from neuroimaging.image import formats
import neuroimaging.data
from neuroimaging.data import FileSystem
import neuroimaging.data.urlhandler as urlhandler

class Image(traits.HasTraits):

    shape = traits.ListInt()

    def __del__(self):
        if self.isfile:
            try:
                self.image.close()
            except:
                pass
        else:
            del(self.image)

    def open(self, mode='r'):
        if self.isfile:
            self.image.open(mode=mode)

    def close(self):
        if self.isfile:
            try:
                self.image.close()
            except:
                pass
        
    def __iter__(self):
        """
        Create an iterator over an image based on its grid's iterator.
        """
        iter(self.grid)

        if isinstance(self.grid.iterator, grid.ParcelIterator) or isinstance(self.grid.iterator, grid.SliceParcelIterator):
            if hasattr(self.image, 'memmap'):
                self.buffer = self.image.memmap
            elif isinstance(self.image.data, N.ndarray):
                self.buffer = self.image.data          
            self.buffer.shape = N.product(self.buffer.shape)
        return self

    def compress(self, where, axis=0):
        if hasattr(self, 'buffer'):
            return self.buffer.compress(where, axis=axis)
        else:
            raise ValueError, 'no buffer: compress not supported'

    def put(self, data, indices):
        if hasattr(self, 'buffer'):
            return self.put.compress(data, indices)
        else:
            raise ValueError, 'no buffer: put not supported'

    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data
        to this image without explicitly setting this image\'s grid to
        the original image\'s grid, i.e. to just take the value the
        original image\'s iterator returns and use it here.
        """
        if value is None:
            self.itervalue = self.grid.next()
            value = self.itervalue

        itertype = value.type

        if itertype is 'slice':
            if data is None:
                return_value = N.squeeze(self.getslice(value.slice))
                if hasattr(self, 'postread'):
                    return self.postread(return_value)
                else:
                    return return_value
            else:

                self.writeslice(value.slice, data)

        elif itertype is 'parcel':
            if data is None:
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                return_value = self.compress(value.where, axis=0)
                if hasattr(self, 'postread'):
                    return self.postread(return_value)
                else:
                    return return_value
            else:
                indices = N.nonzero(value.where)
                self.put(data, indices)

        elif itertype == 'slice/parcel':
            if data is None:
                tmp = self.getslice(value.slice)
                return_value = tmp.compress(value.where)
                if hasattr(self, 'postread'):
                    return self.postread(return_value)
                else:
                    return return_value
            else:
                indices = N.nonzero(value.where)
                self.buffer.put(data, indices)
                

    def getvoxel(self, voxel):
        if len(voxel) != self.ndim:
            raise ValueError, 'expecting a voxel coordinate'
        return self.getslice(voxel)

    def tofile(self, filename, array=True, **keywords):
        outimage = Image(filename, mode='w', grid=self.grid, **keywords)

        if array:
            tmp = self.toarray(**keywords)
            outimage.image.writeslice(slice(0,self.grid.shape[0],1), tmp.image.data)
        else:
            tmp = iter(self)
            outimage = iter(outimage)
            for dataslice in tmp:
                outimage.next(data=dataslice)
                
        outimage.close()
        
        return outimage

def image_factory(input_data, datasource=FileSystem()):
    '''
    Create a Image (volumetric image) object from either a file, an
    existing Image object, or an array.
    '''
    
    # from array
    if isinstance(input_data, N.ndarray):
        image = ArrayPipe(input_data, **keywords)
        image.isfile = False
        
    # from filename or url
    elif type(input_data) == types.StringType:
        image = URLPipe(input_data).getimage()
        image.isfile = True
            
    return image

class BaseImage(object):
    """ Base class for all image objects

    The base class contructor requires an array, or something with the
    same interface, such as a memmap array
    """
    
    def _get_shape(self):
        return self.grid.shape
    shape = property(_get_shape)
    
    def _get_ndim(self):
        return len(self.grid.shape)
    shape = property(_get_ndim)

    def _get_raw_array(self):
        return self._arr
    raw_array = property(_get_raw_array)

    def __init__(self, arr, grid=None):
        self._array = arr
        if grid is not None:
            grid = IdentityGrid(arr.shape)
        self.grid = grid
        
    def __getitem__(self, slices):
        return self.get_slice(slices)

    def __setitem__(self, slices, data):
        self.write_slice(slices, data)

    def to_grid_array(self): 
        '''
        Read an entire Image object, returning a numpy array. By
        default, it does not read 4d images. 
        '''
        # NB - this used to be the readall method of the Image class
        # We may need port old code from this usage in due course
        # CHECK THAT: all grid iterators should have allslice attribute
        slice_obj = self.grid.iterator.allslice
        return self.get_slice(slice_obj)

    def get_slice(self, slices):
        return self._arr[slices]

    def write_slice(self, slices, data):
        self._arr[slices] = data


class ImageSequenceIterator(traits.HasTraits):

    """
    Take a sequence of images, and an optional grid (which defaults to imgs[0].grid) and
    create an iterator whose next method returns array with shapes

    (len(imgs),) + self.imgs[0].next().shape

    Very useful for voxel-based methods, i.e. regression, one-sample t.

    """

    def __init__(self, imgs, grid=None, **keywords):
        self.imgs = imgs
        if grid is None:
            self.grid = iter(self.imgs[0].grid)
        else:
            self.grid = iter(grid)

    def __iter__(self):
        return self

    def next(self, value=None):
        if value is None:
            value = self.grid.next()
        v = []
        for i in range(len(self.imgs)):
            v.append(self.imgs[i].next(value=value))
        return N.array(v, N.Float)


""" Pipes """

'''This module contains the pipes used for the Image class to read and write data.'''

class Pipe(traits.HasTraits):
    shape = traits.ListInt()
    grid = traits.Any()

class URLPipe(Pipe, urlhandler.DataFetcher):
    """
    This class returns an object image from a file object.
    Plans to allow the file to be specified as a url with the goal
    being establishing a protocol to get data remotely.
    """

    mode = traits.Trait(['r', 'w', 'r+'])
    create = traits.false
    filebase = traits.Str()
    fileext = traits.Str()
    filename = traits.Str()
    filepath = traits.Str()
    clobber = traits.false

    def __init__(self, url, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.filename = self.url = url

    def _filename_changed(self):
        self.filebase, self.fileext = \
          os.path.splitext(string.strip(self.filename))
        if self.fileext in self.zipexts:
            self.filebase, self.fileext = \
                os.path.splitext(string.strip(self.filebase))
        #self.filepath, self.filebase = os.path.split(self.filebase)

    def _mode_changed(self): self.create = self.mode not in ('r+', 'r')

    def getimage(self, **keywords):
        if self.grid is None and self.mode == 'w':
            raise ValueError, 'must have a grid to create Image'

        creator = formats.get_creator(self.fileext)

        # determine appropriate data source
        if neuroimaging.data.isurl(self.filename):
            datasource = neuroimaging.data.Repository('')
        else:
            datasource = neuroimaging.data.FileSystem()

        # cache files locally
        #for ext in creator.extensions:
        #    url = self.filebase+ext
        #    cache.cache(url)

        _keywords = copy.copy(keywords)
        _keywords['filename'] = self.filename
        _keywords['datasource'] = datasource
        _keywords['mode'] = self.mode
        _keywords['clobber'] = self.clobber
        _keywords['grid'] = self.grid
        return creator(**_keywords)


class ArrayPipe(Pipe):
    '''A simple class to mimic an image file from an array.'''

    data = traits.Any()

    def __init__(self, data, **keywords):
        '''Create an ArrayPipe instance from an array, by default assumed to be 3d.

        >>> from numpy import *
        >>> from neuroimaging.image.pipes import ArrayPipe
        >>> z = ArrayPipe(zeros((10,20,20),Float))
        >>> print z.ndim
        3
        '''

        traits.HasTraits.__init__(self, **keywords)
        self.data = data.astype(N.Float)

        if self.grid is None and self.shape == []:
            raise ValueError, 'need grid or shape for ArrayPipe'

        if self.grid is None:
            self.grid = ni.reference.grid.IdentityGrid(self.shape)
        else:
            self.shape = self.grid.shape

    
""" Utils """
 
def fwhm2sigma(fwhm):
    return fwhm / N.sqrt(8 * N.log(2))

def sigma2fwhm(sigma):
    return sigma * N.sqrt(8 * N.log(2))

def writebrick(outfile, start, data, shape, offset=0, outtype=None, byteorder=sys.byteorder, return_tell = True):
    if return_tell:
        try:
            startpos = outfile.tell()
        except:
            outfile = file(outfile.name, 'rb+')
            startpos = 0
        
    if outtype:
        outdata = data.astype(outtype)
    else:
        outdata = data
        outtype = outdata.dtype
        
    if byteorder != sys.byteorder:
        outdata.byteswap()

    outdata.shape = (N.product(outdata.shape),)
    
    count = data.shape
    ndim = len(shape)

    # How many dimensions are "full" slices

    nslicedim = 0
    i = ndim - 1
    while count[i] is shape[i] and i >= 0:
        nslicedim = nslicedim + 1
        i = i - 1

    if nslicedim:
        nslice = N.product(shape[(ndim - nslicedim):])
    else:
        nslice = count[:-1]
        nslicedim = 1

    nloopdim = ndim - nslicedim

    test = N.product(N.less_equal(N.array(start) + N.array(count), N.array(shape)))
    if not test:
        raise ValueError, 'start+count not <= shape'

    nloop = N.product(count[nloopdim:])
    nskip = N.product(shape[nloopdim:])
    ntotal = N.product(count)

    elsize = outdata.dtype.itemsize

    shape_reverse = list(shape)
    shape_reverse.reverse()
    strides = [1] + list(N.multiply.accumulate(shape_reverse)[:-1])
    strides.reverse()

    strides = N.array(strides, N.Int64)
    strides = strides * elsize

    outfile.seek(offset + N.add.reduce(start * strides))

    index = 0
    while index < ntotal:
        outdata[index:(index+nloop)].tofile(outfile)
        outfile.seek((nskip - nloop) * elsize, 1)
        index = index + nloop
    if return_tell:
        outfile.seek(startpos, 0)

    outfile.flush()
    del(outdata)

    return 

