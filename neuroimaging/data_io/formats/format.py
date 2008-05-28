"""

"""
#__all__ = ['Format', 'getformats',]
__docformat__ = 'restructuredtext'

import numpy

from neuroimaging import import_from
from neuroimaging.data_io.datasource import DataSource
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path
from neuroimaging.utils.odict import odict

class Format:
    """
    Format is a class which is abstract enough for Images to talk with. It has:

     - grid and datasource
     - metadata in the form of a "header"
     - data get/set (though no specific means to find that data)
     - canonical fields, which is a sort of intersection of all specific
           formats' metadata, but in a known NI language
    """
        
    extensions = []
    """ Valid filename extensions for the file format. """ 

    def __init__(self, datasource=DataSource(), grid=None):
        """
        :Parameters:
            `datasource` : `DataSource`
                TODO
            `grid` : TODO
                TODO
        """
        self.grid = grid
        # Formats should concern themselves with datasources and grids
        self.datasource = datasource
        # Has metadata
        self.header = odict()
        # Possibly has extended data
        self.ext_header = odict()

##      Most of this information below is in the grid
##      "scaling" could be a property that returns the scaling and its inverse
##      "intent" is left as is for now
##      "datasize" can be obtained from self.dtype.itemsize        
##        
##         # Has general information (or should!)
##         self.canonical_fields = odict((
##             ('datasize', 0),
##             ('ndim', 0),
##             ('xdim', 0),
##             ('ydim', 0),
##             ('zdim', 0),
##             ('tdim', 0),
##             ('dx', 0),
##             ('dy', 0),
##             ('dz', 0),
##             ('dt', 0),
##             ('x0', 0),
##             ('y0', 0),
##             ('z0', 0),
##             ('t0', 0),
##             ('intent', ''),
##             ('scaling', 0),
##         ))

    def _getscalers(self):
        def f(x): return x
        return f, f
    scalers = property(_getscalers)

    def _getdatasize(self):
        return self.dtype.itemsize
    datasize = property(_getdatasize)

    def _getintent(self):
        if hasattr(self.header['intent_code']):
            return self.header['intent_code']
        raise AttributeError
    
    def dump_header(self):
        """
        :Returns: ``string``
        """
        return "\n".join(["%s\t%s"%(field,`self.header[field]`) \
                          for field in self.header.keys()])


    def __str__(self):
        """
        :Returns: ``string``
        """
        return self.dump_header()


    def __getitem__(self, slice_):
        """Data access

        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError


    def __setitem__(self, slice_, data):
        """Data access

        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError        


    @classmethod
    def valid(self, filename, verbose=False, mode='r'):
        """
        :Parameters:
            `filename` : string
                TODO
            `verbose` : bool
                TODO
            `mode` : string
                TODO
        
        :Returns: bool
        """
        # verbose not implemented
        try:
            ext = path(filename).splitext()[1]
            if ext not in self.extensions:
                return False
        except:
            return False
        return True
        
    def asfile(self):
        """
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError

format_modules = (
  "neuroimaging.data_io.formats.analyze",
  "neuroimaging.data_io.formats.nifti1",
  "neuroimaging.data_io.formats.ecat7",
  "neuroimaging.data_io.formats.afni",
  #"neuroimaging.data_io.formats.minc",
)

default_formats = [("neuroimaging.data_io.formats.nifti1", "Nifti1"),
                   ("neuroimaging.data_io.formats.analyze", "Analyze"),
                   ("neuroimaging.data_io.formats.ecat7", "Ecat7"),
                   ("neuroimaging.data_io.formats.afni", "AFNI"),
#                   ("neuroimaging.data_io.formats.minc", "MINC1"),
                  ]
                   

def getformat_exts():
    """Return a list of valid image file extensions.

    Returns
    -------
    exts : list of valid extensions

    """

    exts = []
    for modname, formatname in default_formats:
        format = import_from(modname, formatname)
        for fmtext in format.extensions:
            exts.append(fmtext)
    return exts

def getformats(filename):
    """Return the appropriate image format for the given file type.
    
    Parameters
    ----------
    filename : string
        The name of the file to be checked
    
    Returns
    -------
    formats : list
        A list of possible format readers for the filename.
    
    Notes
    -----
    Raises NotImplementedError: if no valid format can be found.

    """

    all_formats = []
    valid_formats = []
    for modname, formatname in default_formats:
        all_formats.append(formatname)
        format = import_from(modname, formatname)
        if format.valid(filename):
            valid_formats.append(format)
        
    if valid_formats:
        return valid_formats
    
    # if we made it this far, a format was not found

    extension = path(filename).splitext()[1]
    raise NotImplementedError, \
      "file extension %(ext)s not recognized, %(exts)s files can be created "\
      "at this time."% {'ext':extension, 'exts':all_formats}


def hasformat(filename):
    """
    Determine if there is an image format format registered for the given
    file type.
    
    :Parameters:
        `filename` : string
            The name of the file to be checked
    
    :Returns: ``bool``
    """
    try:
        getformats(filename)
        return True
    except NotImplementedError:
        return False

