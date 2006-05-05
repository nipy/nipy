'''This module contains the pipes used for the Image class to read and write data.'''

import os, string, urllib, stat, ftplib, gzip, urllib2, copy
import neuroimaging as ni
import numpy as N
import enthought.traits as traits
from neuroimaging.data import DataSource
import neuroimaging.data.urlhandler as urlhandler
from neuroimaging.image.formats import getreader


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

    def __init__(self, url, datasource=DataSource(), **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.datasource = datasource
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

        creator = getreader(self.fileext)
        _keywords = copy.copy(keywords)
        _keywords['filename'] = self.filename
        _keywords['datasource'] = self.datasource
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

    def getslice(self, _slice, **keywords):
        return self.data[_slice]

    def writeslice(self, _slice, data, **keywords):
        self.data[_slice] = data

    
