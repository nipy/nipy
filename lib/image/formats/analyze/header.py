from BrainSTAT.Formats.validators import *
import struct

class header(HasTraits):

    sizeof_hdr = BinaryHeaderAtt('i', seek=0)
    data_type = BinaryHeaderAtt('10s', seek=4)
    db_name = BinaryHeaderAtt('18s', seek=14)
    extents = BinaryHeaderAtt('i', seek=32)
    session_error = BinaryHeaderAtt('h', seek=36)
    regular = BinaryHeaderAtt('s', seek=38)
    hkey_un0 = BinaryHeaderAtt('s', seek=39)
    dim = BinaryHeaderAtt('8h', seek=40)
    vox_units = BinaryHeaderAtt('4s', seek=56)
    calib_units = BinaryHeaderAtt('8s', seek=60)
    unused1 = BinaryHeaderAtt('h', seek=68)
    datatype = BinaryHeaderAtt('h', seek=70)
    bitpix = BinaryHeaderAtt('h', seek=72)
    dim_un0 = BinaryHeaderAtt('h', seek=74)
    pixdim = BinaryHeaderAtt('8f', seek=76)
    vox_offset = BinaryHeaderAtt('f', seek=108)
    funused1 = BinaryHeaderAtt('f', seek=112)
    funused2 = BinaryHeaderAtt('f', seek=116)
    funused3 = BinaryHeaderAtt('f', seek=120)
    calmax = BinaryHeaderAtt('f', seek=124)
    calmin = BinaryHeaderAtt('f', seek=128)
    compressed = BinaryHeaderAtt('i', seek=132)
    verified = BinaryHeaderAtt('i', seek=136)
    glmax = BinaryHeaderAtt('i', seek=140)
    glmin = BinaryHeaderAtt('i', seek=144)
    descrip = BinaryHeaderAtt('80s', seek=148)
    auxfile = BinaryHeaderAtt('24s', seek=228)
    orient = BinaryHeaderAtt('B', seek=252)
    origin = BinaryHeaderAtt('5H', seek=253)
    generated = BinaryHeaderAtt('10s', seek=263)
    scannum = BinaryHeaderAtt('10s', seek=273)
    patient_id = BinaryHeaderAtt('10s', seek=283)
    exp_date = BinaryHeaderAtt('10s', seek=293)
    exp_time = BinaryHeaderAtt('10s', seek=303)
    hist_un0 = BinaryHeaderAtt('3s', seek=313)
    views = BinaryHeaderAtt('i', seek=316)
    vols_added = BinaryHeaderAtt('i', seek=320)
    start_field = BinaryHeaderAtt('i', seek=324)
    field_skip = BinaryHeaderAtt('i', seek=328)
    omax = BinaryHeaderAtt('i', seek=332)
    omin = BinaryHeaderAtt('i', seek=336)
    smax = BinaryHeaderAtt('i', seek=340)
    smin = BinaryHeaderAtt('i', seek=344)

    def __init__(self, hdrfile=None, **keywords):

        if not hasattr(header, '_templatefile'):
            header._templatefile = '/usr/share/BrainSTAT/repository/avg152T1.hdr'
            header._template = header(hdrfile=header._templatefile)
            hdrfile = file(header._templatefile)
            print 'what'
        elif type(hdrfile) is str:
            hdrfile = file(hdrfile)

        self.byteorder, self.bytesign = guess_endianness(hdrfile)

        for traitname in self.trait_names():
            if traitname is not 'trait_added':
                trait = self.trait(traitname)
                trait.handler.bytesign = self.bytesign
                value = trait.handler.read(hdrfile)
                setattr(self, traitname, value)
            else:
                pass

def guess_endianness(hdrfile):
    for order, sign in {'big':'>', 'little':'<', 'net':'!'}.items():
        hdrfile.seek(40)
        test = struct.unpack(sign + 'h', hdrfile.read(2))[0]
        if test in range(1,8):
            print test, order, sign, hdrfile.name
            return order, sign
    raise ValueError, 'file format not recognized: endianness test failed'

        
x = header()

x.configure_traits(kind='modal')
