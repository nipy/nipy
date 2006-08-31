
import numpy

from neuroimaging.data_io import DataSource
from neuroimaging.utils.path import path
from neuroimaging.utils.odict import odict
from neuroimaging.core.reference.grid import SamplingGrid

##############################################################################
class Format (object):

    filename = ''
    filebase = ''

    mode = ''
    bmode = ''

    sctype = numpy.float64

    # Has metadata
    header = odict()

    # Possibly has extended data
    ext_header = odict()
    
    # Has general information (or should!)
    canonical_fields = odict((
        ('datasize', 0),
        ('ndim', 0),
        ('xdim', 0),
        ('ydim', 0),
        ('zdim', 0),
        ('tdim', 0),
        ('intent', ''),
        ('scaling', 0),
        #('grid', SamplingGrid()),
        #('orientation', Orienter()),
    ))

    #-------------------------------------------------------------------------
    def __init__(self, datasource=DataSource(), grid=None, **keywords):
        # Formats should concern themselves with datasources and grids
        self.datasource = datasource
        self.grid = grid
        
    #-------------------------------------------------------------------------
    def dumpHeader(self):
        return "\n".join(["%s\t%s"%(field,`header[field]`) \
                          for field in header.keys()])

    #-------------------------------------------------------------------------
    def inform_canonical(self, fieldsDict=None):
        raise NotImplementedError

    #-------------------------------------------------------------------------
    def __str__(self): self.dumpHeader()


    #-------------------------------------------------------------------------
    def __getitem__(self, slice):
        """Data access"""
        raise NotImplementedError

    #-------------------------------------------------------------------------
    def __setitem__(self, slice, data):
        """Data access"""
        raise NotImplementedError        


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

    #-------------------------------------------------------------------------
    def remove_header_field(self, name):
        """
        Remove a field from the header. Will Definitely Not
        remove any protected fields.
        """
        raise NotImplementedError

    #-------------------------------------------------------------------------
    def set_header_field(self, name, value):
        """
        Set a field's value
        """ 
        raise NotImplementedError

    #-------------------------------------------------------------------------
    def get_header_field(self, name):
        """
        Get a field's value
        """
        raise NotImplementedError

##############################################################################

format_modules = (
  "neuroimaging.data_io.formats.analyze",
  "neuroimaging.data_io.formats.nifti1",
  #"neuroimaging.data_io.formats.afni",
  #"neuroimaging.data_io.formats.minc",
)

default_formats = [("neuroimaging.data_io.formats.nifti1", "NIFTI1"),
                   ("neuroimaging.data_io.formats.analyze", "ANALYZE")]

#-----------------------------------------------------------------------------
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
    raise NotImplementedError,\
      "file extension %(ext)s not recognized, %(exts)s files can be created "\
      "at this time."% {'ext':extension, 'exts':all_formats}

#-----------------------------------------------------------------------------
def hasformat(filename):
    """
    Determine if there is an image format format registered for the given
    file type.
    """
    try:
        getformat(filename)
        return True
    except NotImplementedError:
        return False

