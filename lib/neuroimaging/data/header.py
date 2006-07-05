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
        hdrfile.seek(self.seek)
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

    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    # file, mode, datatype

    clobber = traits.false
    filename = traits.Str()
    filebase = traits.Str()
    mode = traits.Trait('r', 'w', 'r+')
    _mode = traits.Trait(['rb', 'wb', 'rb+'])

    # offset for the memmap'ed array

    offset = traits.Int(0)

    # datasource
    datasource = traits.Instance(DataSource)

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if byteorder == 'little': self.bytesign = '<'
        else: self.bytesign = '>'

    def __str__(self):
        value = ''
        for trait in self.headeratts:
            value = value + '%s:%s=%s\n' % (self.filebase, trait, str(getattr(self, trait)))
        return value

    def readheader(self, hdrfile=None):
        self.check_byteorder()
        if hdrfile is None:
            hdrfile = self.datasource.open(self.hdrfilename())
        for traitname in self.headeratts:
            trait = self.trait(traitname)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, traitname, value)
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

        for traitname in self.headeratts:
            trait = self.trait(traitname)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, traitname), outfile=hdrfile)

        hdrfile.close()

def add_headeratts(imageclass, headeratts):
    """
    Add header attributes to a BinaryHeader class.
    """
    imageclass.headeratts = []

    seek = 0
    for att in headeratts:
        if len(att) == 3:
            name, packstr, default = att
            extra = {}
        else:
            name, packstr, default, desc = att
            extra = {'desc':desc}
        imageclass.add_class_trait(name, BinaryHeaderAtt(packstr, seek, default, **extra))
        imageclass.headeratts.append(name)
        seek += calcsize(packstr)
