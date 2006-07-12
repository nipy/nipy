from types import StringType
from struct import calcsize, pack, unpack
from sys import byteorder
from os.path import exists

from neuroimaging.data import DataSource
from neuroimaging import traits

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
        if len(packstr) < 1: raise ValueError("packstr must be nonempty")
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

def BinaryHeaderAtt(packstr, seek=0, value=None, **keywords):
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


    # file, mode, datatype

    clobber = traits.false
    filename = traits.Str()
    filebase = traits.Str()
    mode = traits.Trait('r', 'w', 'r+')
    _mode = traits.Trait(['rb', 'wb', 'rb+'])

    # datasource
    datasource = traits.Instance(DataSource)

    header = traits.List
    header_length = traits.Int(0)
    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if byteorder == 'little': self.bytesign = '<'
        else: self.bytesign = '>'

        for att in self.header:
            self.set_header_attribute(att, append=True)
            
    def __str__(self):
        value = ''
        for trait in self.header:
            value = value + '%s:%s=%s\n' % (self.filebase, trait[0], str(getattr(self, trait[0])))
        return value

    def readheader(self, hdrfile=None):
        self.check_byteorder()
        if hdrfile is None:
            hdrfile = self.datasource.open(self.hdrfilename())
        for att in self.header:
            name = att[0]
            trait = self.trait(name)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, name, value)
        self.getdtype()
        hdrfile.close()

    def getdtype(self):
        raise NotImplementedError

    def hdrfilename(self):
        raise NotImplementedError

    def check_byteorder(self):
        raise NotImplementedError
        
    def writeheader(self, hdrfile=None):

        if hdrfile is None:
            hdrfilename = self.hdrfilename()
            if self.clobber or not exists(self.hdrfilename()):
                hdrfile = file(hdrfilename, 'wb')
            else:
                raise ValueError, 'error writing %s: clobber is False and hdrfile exists' % hdrfilename

        for att in self.header:
            name = att[0]
            trait = self.trait(name)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, name), outfile=hdrfile)

        hdrfile.close()

    def set_header_attribute(self, headeratt, seek=0, append=True):
        """
        Add header attributes to a Header instance.
        """

        if append:
            seek = self.header_length
        if len(headeratt) == 3:
            name, definition, default = headeratt
            keywords = {}
        else:
            name, definition, default, keywords = headeratt
        if name in self.traits().keys():
            self.remove_trait(name)
        self.add_trait(name, BinaryHeaderAtt(definition, seek=seek, value=default, **keywords))
        if append:
            self.header_length += calcsize(definition)

