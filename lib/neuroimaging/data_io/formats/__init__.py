from numpy import sctypes as _sctypes

from neuroimaging import import_from, traits
from neuroimaging.data_io import DataSource
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path


sctypes = []
for key in ['float', 'complex', 'int', 'uint']:
    sctypes += [traits.Type(val) for val in _sctypes[key]]
sctypes = traits.Trait(sctypes)

class Format(traits.HasTraits):

    """ Valid file name extensions """
    extensions = []

    """ Character representation of scalar (numpy) type """
    sctype = traits.Trait(sctypes)

    """ Clobber, filename, filebase, mode """

    clobber = traits.false
    filename = traits.Str()
    filebase = traits.Str()

    mode = traits.Trait('r', 'w', 'r+')
    bmode = traits.Trait(['rb', 'wb', 'rb+'])

    """ Data source """
    datasource = traits.Instance(DataSource)

    """ Header definition """

    header = traits.List

    """ Sampling grid """
    grid = traits.Trait(SamplingGrid)

    def __init__(self, filename, datasource=DataSource(), **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.datasource = datasource
        self.filename = filename

    @classmethod
    def valid(self, filename, verbose=False, mode='r'):
        """
        Check if filename is valid. If verbose=True, actually try to open
        and read the file.
        """
        try:
            extension = path(filename).splitext()[1]
            if extension not in self.extensions:
                return False
            if verbose:
                # Try to actually instantiate the objects
                self(filename, mode)
        except:
            return False
        return True

    def __str__(self):
        value = ''
        for trait in self.header:
            value = value + '%s:%s=%s\n' % (self.filebase, trait[0], str(getattr(self, trait[0])))
        return value

    def add_header_attribute(self, name, definition, value):
        """
        Add an attribute to the header. Definition should be
        a format string interpretable by struct.pack. 
        """
        raise NotImplementedError

    def set_header_attribute(self, name, attribute, value):
        """
        Set an attribute to the header.
        """
        raise NotImplementedError

    def remove_header_attribute(self, name, attribute):
        """
        Remove an attribute from the header.
        """
        raise NotImplementedError

    def read_header(self, hdrfile=None):
        """
        Read header file.
        """
        raise NotImplementedError

    def write_header(self, hdrfile=None):
        """
        Write header file.
        """
        raise NotImplementedError

    def header_filename(self):
        """
        Get header file name.
        """
        return self.filename

    def __getitem__(self, slice):
        """Data access"""
        raise NotImplementedError

    def __setitem__(self, slice, data):
        """Data access"""
        raise NotImplementedError

format_modules = (
  "neuroimaging.data_io.formats.analyze",
  "neuroimaging.data_io.formats.nifti1",
  #"neuroimaging.data_io.formats.afni",
  #"neuroimaging.data_io.formats.minc",
)

default_formats = [("neuroimaging.data_io.formats.nifti1", "NIFTI1"),
                   ("neuroimaging.data_io.formats.analyze", "ANALYZE")]

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

