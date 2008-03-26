__docformat__ = 'restructuredtext'

from numpy.core import memmap as memmap_type
from numpy import memmap
import numpy as N
import os

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.formats import utils
from neuroimaging.data_io.formats.format import Format
from neuroimaging.data_io.datasource import DataSource, iszip, unzip, iswritemode


class BinaryFormatError(Exception):
    """
    Binary format error exception
    """    

class BinaryFormat(Format):
    """
    BinaryFormat(Format) is an object with binary header data and a "brick" of
    binary data. The binary.py module has a lot of struct packing/unpacking
    methods to make life nicer. BinaryFormats have:
     - filenames and modes--in fact, BinaryFormats are always constructed with a
       filename, this is not an option anymore (makes sense to me).
     - header formats (in package struct language)
     - means to read/write contiguous/deterministic-style headers
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
        self._check_filenames()

        if self.clobber and 'w' in self.mode:
            try:
                os.remove(self.datasource.filename(self.data_file))
            except IOError:
                # Ignore exceptions due to the file not existing, etc.
                pass

        self.dtype = NotImplemented
        self.byteorder = NotImplemented
        self.extendable = NotImplemented

    def _check_filenames(self):
        if (self.datasource.exists(self.header_file) or \
               self.datasource.exists(self.data_file)) and \
               not self.clobber and \
               'w' in self.mode:
            raise IOError('file exists, but not allowed to clobber it')

    def read_header(self):
        """
        :Returns: ``None``
        """
        # Populate header dictionary from a file
        values = utils.struct_unpack(self.datasource.open(self.header_file),
                               self.byteorder,
                               self.header_formats.values())
        for field, val in zip(self.header.keys(), values):
            self.header[field] = val
        return self.header

    def write_header(self, hdrfile=None, clobber=False):
        """
        :Returns: ``None``
        """
        
        # If someone wants to write a headerfile somewhere specific,
        # handle that case immediately
        # Otherwise, try to write to the object's header file
        if hdrfile:
            fp = isinstance(hdrfile, str) and open(hdrfile,'rb+') or hdrfile
        elif self.datasource.exists(self.header_file):
            fp = self.datasource.open(self.header_file, 'rb+')
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


    def attach_data(self, offset=0, use_memmap=True):
        """
        :Returns: ``None``

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
        fname = iszip(self.datasource.filename(self.data_file)) and \
                unzip(self.datasource.filename(self.data_file)) or \
                self.datasource.filename(self.data_file)
            

        self.data = memmap(fname, dtype=self.dtype,
                           shape=tuple(self.grid.shape), mode=mode,
                           offset=offset)

        self.use_memmap = True
        if not use_memmap:
            self.data = self[:]
            self.data.newbyteorder(self.byteorder)
        self.use_memmap = use_memmap


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
        :Returns: ``numpy.ndarray``
        """
        if self.use_memmap:
            data = self.postread(self.data[slicer].newbyteorder(self.byteorder))            
        else:
            data = self.postread(self.data[slicer])
        return N.asarray(data)


    def __setitem__(self, slicer, data):
        """
        :Returns: ``None``
        """
        if self.use_memmap and not iswritemode(self.data._mode):
            print "Warning: memapped array is not writeable! Nothing done"
            return
        self.data[slicer] = \
            self.prewrite(data).astype(self.dtype)

    def __del__(self):
        """
        :Returns: ``None``
        """
        if hasattr(self, 'memmap'):
            if isinstance(self.data, memmap_type):
                self.data.sync()
            del(self.data)

    def _get_filenames(self):
        """
        Calculate header_file and data_file filenames

        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError
            
    def asfile(self):
        """
        :Returns: ``string``
        """
        return self.datasource.filename(self._get_filenames()[1])

