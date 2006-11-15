"""
Format is a class which is abstract enough for Images to talk with. It has:

*grid and datasource
*metadata in the form of a "header"
*data get/set (though no specific means to find that data)
*canonical fields, which is a sort of intersection of all specific
formats' metadata, but in a known NI language

BinaryFormat(Format) is an object with binary header data and a "brick" of
binary data. The binary.py module has a lot of struct packing/unpacking
methods to make life nicer. BinaryFormats have:
*filenames and modes--in fact, BinaryFormats are always constructed with a
filename, this is not an option anymore (makes sense to me).
*header formats (in package struct language)
*means to read/write contiguous/deterministic-style headers

"""
__all__ = ['Format']

import numpy

from neuroimaging import import_from
from neuroimaging.data_io import DataSource
from neuroimaging.utils.path import path
from neuroimaging.utils.odict import odict
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.image.base_image import BaseImage

class Format(BaseImage):

    """ Valid filename extensions for the file format. """ 
    extensions = []

    def __init__(self, datasource=DataSource(), grid=None, **keywords):
        BaseImage.__init__(self, NotImplemented, grid, NotImplemented)
        # Formats should concern themselves with datasources and grids
        self.datasource = datasource
        # Has metadata
        self.header = odict()
        # Possibly has extended data
        self.ext_header = odict()
        # Has general information (or should!)
        self.canonical_fields = odict((
            ('datasize', 0),
            ('ndim', 0),
            ('xdim', 0),
            ('ydim', 0),
            ('zdim', 0),
            ('tdim', 0),
            ('dx', 0),
            ('dy', 0),
            ('dz', 0),
            ('dt', 0),
            ('x0', 0),
            ('y0', 0),
            ('z0', 0),
            ('t0', 0),
            ('intent', ''),
            ('scaling', 0),
        ))

        

    def dump_header(self):
        return "\n".join(["%s\t%s"%(field,`self.header[field]`) \
                          for field in self.header.keys()])


    def inform_canonical(self, fieldsDict=None):
        raise NotImplementedError


    def __str__(self):
        return self.dump_header()


    def __getitem__(self, slice_):
        """Data access"""
        raise NotImplementedError


    def __setitem__(self, slice_, data):
        """Data access"""
        raise NotImplementedError        


    @classmethod
    def valid(self, filename, verbose=False, mode='r'):
        # verbose not implemented
        try:
            ext = path(filename).splitext()[1]
            if ext not in self.extensions:
                return False
        except:
            return False
        return True
        

    #############################################
    ## The following header manipulations might
    ## not be implemented for a given format.
    ## They require specific knowledge of the
    ## header field names
    #############################################
    def add_header_field(self, name, type, value):
        """
        Add a field to the header. Type should be
        a format string interpretable by struct.pack.
        """
        raise NotImplementedError


    def remove_header_field(self, name):
        """
        Remove a field from the header. Will Definitely Not
        remove any protected fields.
        """
        raise NotImplementedError


    def set_header_field(self, name, value):
        """
        Set a field's value
        """ 
        raise NotImplementedError


    def get_header_field(self, name):
        """
        Get a field's value
        """
        raise NotImplementedError



format_modules = (
  "neuroimaging.data_io.formats.analyze",
  "neuroimaging.data_io.formats.nifti1",
  "neuroimaging.data_io.formats.ecat7",
  "neuroimaging.sandbox.formats.afni",
  #"neuroimaging.data_io.formats.afni",
  #"neuroimaging.data_io.formats.minc",
)

default_formats = [("neuroimaging.data_io.formats.nifti1", "Nifti1"),
                   ("neuroimaging.data_io.formats.analyze", "Analyze"),
                   ("neuroimaging.data_io.formats.ecat7", "Ecat7"),
                   ("neuroimaging.sandbox.formats.afni", "AFNI"),
                  ]
                   


def getformats(filename):
    "Return the appropriate image format for the given file type."
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
    """
    try:
        getformats(filename)
        return True
    except NotImplementedError:
        return False

