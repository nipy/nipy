from types import StringType
from struct import calcsize, pack, unpack
from sys import byteorder
from os.path import exists

from neuroimaging import traits
from numpy import memmap, zeros

from neuroimaging.image.formats import Format
from neuroimaging.data import iszip, unzip

class BinaryFormatError(Exception):
    """
    Errors raised in intents.
    """

def is_tupled(packstr, value):
    return (packstr[-1] != 's' and len(tuple(value)) > 1)

def isseq(value):
    try:
        len(value)
        isseq = True
    except TypeError:
        isseq = False
    return type(value) != StringType and isseq

class BinaryHeaderValidator(traits.TraitHandler):
    """
    Trait handler used in definition of binary
    headers.
    """

    def __init__(self, packstr, value=None, seek=0, bytesign = '>', **keywords):
        if len(packstr) < 1: raise BinaryFormatError("packstr must be nonempty")
        for key, value in keywords.items(): setattr(self, key, value)
        self.seek = seek
        self.packstr = packstr
        self.bytesign = bytesign
        self.size = calcsize(self.packstr)

    def write(self, value, outfile=None):
        if isseq(value): valtup = tuple(value) 
        else: valtup = (value,)
        result = pack(self.bytesign+self.packstr, *valtup)
        if outfile is not None:
            outfile.seek(self.seek)
            outfile.write(result)
        return result

    def validate(self, object, name, value):
        try:
            result = self.write(value)
        except:
            self.error(object, name, value)

        _value = unpack(self.bytesign + self.packstr, result)
        if is_tupled(self.packstr, _value): return _value
        else: return _value[0]

    def info(self):
        return 'an object of type "%s", apply(struct.pack, "%s", object) must make sense' % (self.packstr, self.packstr)

    def read(self, hdrfile):
        value = unpack(self.bytesign + self.packstr,
                       hdrfile.read(self.size))
        if not is_tupled(self.packstr, value):
            value = value[0]
        return value

def BinaryHeaderAttribute(packstr, seek=0, value=None, **keywords):
    """
    Constructor for a binary header attribute for ANALYZE and NIFTI-1 files.
    """
    validator = BinaryHeaderValidator(packstr, value=value, seek=seek, **keywords)
    return traits.Trait(value, validator)

class BinaryHeader(traits.HasTraits):
    """
    A simple way to specify the format of a binary file in terms
    of strings that struct.{pack,unpack} can validate and a byte
    position in the file.

    This is used for both ANALYZE-7.5 and NIFTI-1 files in the neuroimaging package.
    """

    header_length = traits.Int(0)
    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    def __str__(self):
        value = ''
        for trait in self.header:
            value = value + '%s:%s=%s\n' % (self.filebase, trait[0], str(getattr(self, trait[0])))
        return value

class BinaryFormat(Format):

    """
    Format with a brick of data attached.
    """

    header_length = traits.Int(0)
    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    inited_flag = traits.ReadOnly(desc='Set once all predefined attributes are added.')

    def __init__(self, filename, **keywords):
        Format.__init__(self, filename, **keywords)
        traits.HasTraits.__init__(self, **keywords)
        if byteorder == 'little': self.bytesign = '<'
        else: self.bytesign = '>'

        for att in self.header:
            if len(att) == 3:
                name, definition, value = att
            else:
                name, definition, value, keywords = att
                
            self.add_header_attribute(name, definition, value, keywords=keywords, append=False)
        self.inited_flag = True

    def check_byteorder(self):
        """
        Determine byteorder of data on disk.
        """
        raise NotImplementedError

    def modifiable(self):
        if self.inited_flag is not True: return True
        else: return self.image_filename() != self.header_filename()

    def set_header_attribute(self, name, value):
        attnames = [att[0] for att in self.header]
        if name in attnames:
            setattr(self, name, value)
        else:
            raise BinaryFormatError, 'attribute "%s" not in header' % name

    def add_header_attribute(self, name, definition, value, keywords={}, append=True):

        if self.modifiable():
            seek = self.header_length
            self.add_trait(name, BinaryHeaderAttribute(definition, seek=seek, value=value, **keywords))
            self.header_length += calcsize(definition)
            if append:
                self.header.append((name, definition, value, keywords))
        else:
            raise BinaryFormatError, 'header can not be modified'

    def read_header(self, hdrfile=None):
        """
        Read header file.
        """

        self.check_byteorder()
        if hdrfile is None:
            hdrfile = self.datasource.open(self.header_filename())
        for att in self.header:
            name = att[0]
            trait = self.trait(name)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, name, value)
        self.get_dtype()
        hdrfile.close()

    def get_dtype(self):
        """
        Get numpy dtype of data.
        """
        return self.dtype


    def write_header(self, hdrfile=None):

        if hdrfile is None:
            hdrfilename = self.header_filename()
            if self.clobber or not exists(self.header_filename()):
                hdrfile = file(hdrfilename, 'wb')
            else:
                raise BinaryFormatError, 'error writing %s: clobber is False and hdrfile exists' % hdrfilename

        for att in self.header:
            name = att[0]
            trait = self.trait(name)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, name), outfile=hdrfile)

        hdrfile.close()

    def remove_header_attribute(self, name):
        if self.modifiable():

            attnames = [att[0] for att in self.header]
            if name in attnames:
                idx = attnames.index(name)
                trait = self.trait(name)
                if not trait.required:
                    self.remove_trait(name)
                self.header.pop(idx)
        else:
            raise BinaryFormatError, 'header can not be modified'

    def image_filename(self):
        raise NotImplementedError

    def emptyfile(self, offset=0):
        """
        Create an empty data file based on
        self.grid and self.dtype
        """
        
        if not exists(self.image_filename()):
            outfile = file(self.image_filename(), 'wb')
        else:
            outfile = file(self.image_filename(), 'rb+')

        if outfile.name == self.header_filename():
            offset += self.header_length
        outfile.seek(offset)
        
        outstr = zeros(self.grid.shape[1:], self.dtype).tostring()
        for i in range(self.grid.shape[0]):
            outfile.write(outstr)
        outfile.close()

    def __del__(self):
        if hasattr(self, "memmap"):
            self.memmap.sync()
            del(self.memmap)

    def postread(self, x):
        """
        Point transformation of data post reading.
        """
        return x

    def prewrite(self, x):
        """
        Point transformation of data pre writing.
        """
        return x

    def __getitem__(self, _slice):
        return self.postread(self.memmap[_slice])

    def getslice(self, _slice): return self[_slice]

    def __setitem__(self, _slice, data):
        self.memmap[_slice] = self.prewrite(data).astype(self.dtype)

    def writeslice(self, _slice, data): self[slice] = data

    def attach_data(self, offset=0):

        """
        Attach data to self.
        """

        imgpath = self.image_filename()
        if imgpath == self.header_filename():
            offset += self.header_length

        image_filename = self.datasource.filename(imgpath)
        if iszip(image_filename): image_filename = unzip(image_filename)
        mode = self.mode in ('r+', 'w') and "r+" or self.mode
        self.memmap = memmap(image_filename, dtype=self.dtype,
                             shape=tuple(self.grid.shape), mode=mode,
                             offset=offset)


## if __name__ == '__main__':

##     import numpy.random as R
##     import numpy as N
##     from neuroimaging.reference.grid import SamplingGrid

##     ofile = file('example.dat', 'wb')
##     shape = (40,5,6)
##     x = R.standard_normal(shape)
##     ofile.write(pack('3i', *shape))
##     ofile.write(x.tostring())
##     ofile.close()

##     class TestFormat(BinaryFormat):

##         extensions = ['dat']
##         def check_byteorder(self):
##             return 

##         header = traits.List([('dims', '3i', (10,10,10))])
        
##         def __init__(self, filename, dtype=float64, **keywords):
##             BinaryFormat.__init__(self, filename, **keywords)
##             self.dtype = N.dtype(dtype)
##             self.read_header()
##             self.grid = SamplingGrid.identity(self.dims)
##             self.attach_data()

##         def _dims_changed(self, old, new):
##             print 'here', self.dims, old, new
##             self.grid = SamplingGrid.identity(self.dims)
            
##         def image_filename(self):
##             return self.header_filename()

##     import os
    
##     a = TestFormat('example.dat', clobber=True)
##     print a.dims, a[:].max()
##     a.dims = (40,6,5)
##     a.write_header()
##     a.read_header()
##     print a.dims, a[:].max()
##     a.attach_data()
##     c = a[:] * 1.
##     print a.memmap.shape, c.shape, 'CCCCC'
##     os.system('ls -la example.dat')

##     class TestFormat2(TestFormat):

##         extensions = ['dat']
##         def check_byteorder(self):
##             return 

##         header = traits.List([('dims', '3i', (10,10,10))])
        
##         def __init__(self, filename, dtype=float64, **keywords):
##             BinaryFormat.__init__(self, filename, **keywords)
##             self.dtype = N.dtype(dtype)
##             self.read_header()
##             self.grid = SamplingGrid.identity(self.dims)
##             self.attach_data()

##         def image_filename(self):
##             return '%simg' % self.header_filename()[0:-3] 

##     ofile = file('example.img', 'w')
##     print c.shape, len(c.tostring())
##     ofile.write(c.tostring())
##     ofile.close()

##     ofile = file('example.dat', 'w')
##     ofile.write(pack('3i', *c.shape))
##     ofile.close()

##     a = TestFormat2('example.dat', clobber=True)

##     a.add_header_attribute('desc', '20s', 'test' + ' '*16)
##     a.write_header()
##     a.read_header()
##     print a.dims, a[:].max(), a.desc, a[:].shape, c.shape
##     os.system('ls -la example.dat')

##     N.testing.assert_almost_equal(c, a[:])

##     d = TestFormat2('example.dat')
##     N.testing.assert_almost_equal(c, d[:])
