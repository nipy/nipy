from struct import pack, unpack, calcsize
from types import StringType
from sys import byteorder

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
    Trait handler used in definition of ANALYZE and NIFTI-1
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
        #if 1:
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

class BinaryFile(traits.HasTraits):
    """
    A simple way to specify the format of a binary file in terms
    of strings that struct.{pack,unpack} can validate and a byte
    position in the file.

    This is used for both ANALYZE-7.5 and NIFTI-1 files.
    """

    clobber = traits.false
    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if byteorder == 'little': self.bytesign = '<'
        else: self.bytesign = '>'

        self.hdrattnames = [name for name in self.trait_names() \
          if isinstance(self.trait(name).handler, BinaryHeaderValidator)]

    def readheader(self, hdrfile):
        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, traitname, value)

    def writeheader(self, hdrfile=None):

        if hdrfile is None:
            hdrfilename = self.hdrfilename()
            if self.clobber or not os.path.exists(self.hdrfilename()):
                hdrfile = file(hdrfilename, 'wb')
            else:
                raise ValueError, 'error writing %s: clobber is False and hdrfile exists' % hdrfilename

        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, traitname), outfile=hdrfile)

        hdrfile.close()
