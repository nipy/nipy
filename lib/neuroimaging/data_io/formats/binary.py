__docformat__ = 'restructuredtext'

from numpy.core import memmap as memmap_type
from numpy import memmap
import numpy as N
import os

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.formats import Format, utils
from neuroimaging.data_io import DataSource, iszip, unzip, iswritemode


class BinaryFormatError(Exception):
    """
    Binary format error exception
    """    

class BinaryFormat(Format):
    """
    BinaryFormat(Format) is an object with binary header data and a "brick" of
    binary data. The binary.py module has a lot of struct packing/unpacking
    methods to make life nicer. BinaryFormats have:
    *filenames and modes--in fact, BinaryFormats are always constructed with a
    filename, this is not an option anymore (makes sense to me).
    *header formats (in package struct language)
    *means to read/write contiguous/deterministic-style headers
    """

    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        # keep BinaryFormats dealing with datasource and filename/mode
        Format.__init__(self, datasource, keywords.get('grid', None))        
        self.mode = mode
        self.filename = filename
        self.filebase = os.path.splitext(filename)[0]
        self.header_formats = odict()
        self.ext_header_formats = odict()
        self.clobber = keywords.get('clobber', False)

        self.header_file, self.data_file = self._get_filenames()

        if self.clobber and 'w' in self.mode:
            try:
                os.remove(self.datasource.filename(self.data_file))
            except IOError:
                # Ignore exceptions due to the file not existing, etc.
                pass

        self.dtype = NotImplemented
        self.byteorder = NotImplemented
        self.extendable = NotImplemented


    def read_header(self):
        """
        :Returns:
            `None`
        """
        # Populate header dictionary from a file
        values = utils.struct_unpack(self.datasource.open(self.header_file),
                               self.byteorder,
                               self.header_formats.values())
        for field, val in zip(self.header.keys(), values):
            self.header[field] = val


    def write_header(self, hdrfile=None, clobber=False):
        """
        :Returns:
            `None`
        """
        
        # If someone wants to write a headerfile somewhere specific,
        # handle that case immediately
        # Otherwise, try to write to the object's header file
        if hdrfile:
            fp = isinstance(hdrfile, str) and open(hdrfile,'wb+') or hdrfile
        elif self.datasource.exists(self.header_file):
            if not clobber:
                raise IOError('file exists, but not allowed to clobber it')
            fp = self.datasource.open(self.header_file, 'wb+')
        else:
            fp = open(self.datasource._fullpath(self.header_file), 'wb+')
        packed = utils.struct_pack(self.byteorder,
                             self.header_formats.values(),
                             self.header.values())
        fp.write(packed)

        if self.extendable and self.ext_header != {}:
            packed_ext = utils.struct_pack(self.byteorder,
                                     self.ext_header_formats.values(),
                                     self.ext_header.values())
            fp.write(packed_ext)

        # close it if we opened it
        if not hdrfile or type(hdrfile) is not type(fp):
            fp.close()


    def attach_data(self, offset=0):
        """
        :Returns:
            `None`

        :Raises IOError: If the file exists but we are not allowed to clobber
            it.
        """
        if iswritemode(self.mode):
            mode = 'readwrite'
        else:
            mode = 'readonly'
        if mode == 'readwrite':
            if not self.datasource.exists(self.data_file):
                utils.touch(self.datasource._fullpath(self.data_file))
            elif not self.clobber:
                raise IOError('file exists, but not allowed to clobber it')

        fname = iszip(self.datasource.filename(self.data_file)) and \
                unzip(self.datasource.filename(self.data_file)) or \
                self.datasource.filename(self.data_file)
            

        self.data = memmap(fname, dtype=self.dtype,
                           shape=tuple(self.grid.shape), mode=mode,
                           offset=offset)


    def prewrite(self, x):
        """
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError


    def postread(self, x):
        """
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError

    def __getitem__(self, slicer):
        """
        :Returns:
            `numpy.ndarray`
        """
        data = self.postread(self.data[slicer].newbyteorder(self.byteorder))
        return N.asarray(data)


    def __setitem__(self, slicer, data):
        """
        :Returns:
            `None
        """
        if not iswritemode(self.data._mode):
            print "Warning: memapped array is not writeable! Nothing done"
            return
        self.data[slicer] = \
            self.prewrite(data).astype(self.dtype)

    def __del__(self):
        """
        :Returns:
            `None`
        """
        if hasattr(self, 'memmap'):
            if isinstance(self.data, memmap_type):
                self.data.sync()
            del(self.data)

    #### These methods are extraneous, the header dictionaries are
    #### unprotected and can be looked at directly
    def add_header_field(self, field, format, value):
        """
        :Returns:
            `None`
        """
        if not self.extendable:
            raise NotImplementedError("%s header type not "\
                                      "extendable" % self.__class__)
        if field in (self.header.keys() + self.ext_header.keys()):
            raise ValueError("Field %s already exists in "
                             "the current header" % field)
        if not utils.sanevalues(format, value):
            raise ValueError("Field format and value(s) do not match up.")

        # if we got here, we're good
        self.ext_header_formats[field] = format
        self.ext_header[field] = value


    def remove_header_field(self, field):
        """
        :Returns:
            `None`
        """
        if field in self.ext_header.keys():
            self.ext_header.pop(field)
            self.ext_header_formats.pop(field)



    def set_header_field(self, field, value):
        """
        :Returns:
            `None`

        :Raises KeyError: if the given `field` is not valid.
        """
        try:
            if utils.sanevalues(self.header_formats[field], value):
                self.header[field] = value
        except KeyError:
            try:
                if utils.sanevalues(self.ext_header_formats[field], value):
                    self.ext_header[field] = value
            except KeyError:
                raise KeyError('Field does not exist')
    

    def get_header_field(self, field):
        """
        :Raises KeyError: if the given `field` is not valid.
        """
        try:
            return self.header[field]
        except KeyError:
            try:
                return self.ext_header[field]
            except KeyError:
                raise KeyError
    

    def _get_filenames(self):
        """
        Calculate header_file and data_file filenames

        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError
            
    def asfile(self):
        """
        :Returns:
            `string`
        """
        return self.datasource.filename(self._get_filenames()[1])

    def inform_canonical(self, fieldsDict=None):
        """
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError
