'''This module contains the pipes used for the Image class to read and write data.'''

import os, string, urllib, stat, ftplib, gzip, urllib2
import formats
import neuroimaging as ni
import numpy as N
import enthought.traits as traits
import neuroimaging.data.urlhandler as urlhandler

class Pipe(traits.HasTraits):

    shape = traits.ListInt()
    grid = traits.Any()

class URLPipe(Pipe,urlhandler.DataFetcher):

    """
    This class returns an object image from a file object.
    Plans to allow the file to be specified as a url with the goal
    being establishing a protocol to get data remotely.
    """

    repository = traits.Str('/home/jtaylo/.BrainSTAT/repository')
    mode = traits.Trait(['r', 'w', 'r+'])
    create = traits.false
    filebase = traits.Str()
    fileext = traits.Str()
    filename = traits.Str()
    filepath = traits.Str()
    
    clobber = traits.false

    def _filename_changed(self):
        self.filebase, self.fileext = os.path.splitext(string.strip(self.filename))
        self.filepath, self.filebase = os.path.split(self.filebase)

    def _mode_changed(self):
        if self.mode in ['r+', 'r']:
            self.create = False
        else:
            self.create = True

    def __init__(self, url, **keywords):

        traits.HasTraits.__init__(self, **keywords)
        self.filename = url
        self.otherexts = ['.hdr', '.mat']
        self.geturl(url)

    def getimage(self):

        if self.grid is None and self.mode == 'w':
            raise ValueError, 'must have a template to create BrainSTAT file'

        creator = None
        extensions = []
        for format in formats.valid:
            extensions += format.valid
            if self.fileext in format.valid:
                creator = format.creator
        if creator is None:
            raise NotImplementedError, 'file extension %(ext)s not recoginzed, %(exts)s files can be written at this time.' % {'ext':self.fileext, 'exts': extensions}

        image = creator(filename=self.url, mode=self.mode, clobber=self.clobber, grid=self.grid)
        return image


class ArrayPipe(Pipe):

    data = traits.Any()

    '''A simple class to mimic an image file from an array.'''
    def __init__(self, data, **keywords):
        '''Create an ArrayPipe instance from an array, by default assumed to be 3d.

        >>> from numpy import *
        >>> from BrainSTAT.Base.Pipes import ArrayPipe
        >>> z = ArrayPipe(zeros((10,20,20),Float))
        >>> print z.ndim
        3
        '''

        traits.HasTraits.__init__(self, **keywords)
        self.data = data.astype(N.Float)

        if self.grid is None and self.shape is []:
            raise ValueError, 'need grid or shape for ArrayPipe'

        if self.grid is None:
            self.grid = ni.reference.grid.IdentityGrid(self.shape)
        else:
            self.shape = self.grid.shape

    def read(self, start, count, **keywords):
        '''Read a hyperslab of data from an array. Start is a tuple containing the lower corner of the slab, and count is the dimension of the slab.

        >>> from numpy import *
        >>> from BrainSTAT.Base.Pipes import ArrayPipe
        >>> z = ArrayPipe(zeros((10,20,20),Float))
        >>> w = z.read((0,)*3, (10,)*3)
        >>> print w.shape
        (10, 10, 10)
        '''
        
        exec('value = self.data[%s]' % self._tuple2slice(start, count))
        return value

    def write(self, start, data, **keywords):
        '''Read a hyperslab of data from an array. Start is a tuple containing the lower corner of the slab, and count is the dimension of the slab.

        >>> from numpy import *
        >>> from BrainSTAT.Base.Pipes import ArrayPipe
        >>> z = ArrayPipe(zeros((10,20,20),Float))
        >>> z.write((0,)*3, ones((10,10,10)))
        >>> z.data[2,2,2]
        1.0
        '''

        count = data.shape
        exec('self.data[%s] = data' % _tuple2slice(start, count))

def _tuple2slice(start, count):
    end = tuple(N.array(start) + N.array(count))
    import string
    _slice = string.join(map(lambda x, y: '%d:%d' % (x,y), start, end), sep=',')
    return _slice
    
