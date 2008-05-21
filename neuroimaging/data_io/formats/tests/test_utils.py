from StringIO import StringIO

from numpy import allclose

from neuroimaging.externals.scipy.testing import *

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.formats import utils

fmts = odict((
        ('uchar', 'B'),
        ('int', '4i'),
        ('short', 'h'),
        ('float', '3f'),
        ('string', '6s')
))
 
values = [255, 1, 2, 3, 4, 123, 1.1, 2.2, 3.3, 'foobar']

# NOTE: Test basic interface to our stuct packing/unpacking functions.
# Not meant to be a complete testing of the struct module.
def test_little_endian_struct():
    fp = StringIO()
    packed = utils.struct_pack(utils.LITTLE_ENDIAN, fmts.values(), values)
    fp.write(packed)
    fp.seek(0)
    unpacked = utils.struct_unpack(fp, utils.LITTLE_ENDIAN, fmts.values())
    mydict = {}
    for field, val in zip(fmts.keys(), unpacked):
        mydict[field] = val
    assert mydict['uchar'] == values[0]
    assert mydict['int'] == values[1:5]
    assert mydict['short'] == values[5]
    assert allclose(mydict['float'], values[6:9])
    assert mydict['string'] == values[9]
    fp.close()

def test_fail_wrong_endian():
    fp = StringIO()
    # Write as Little Endian
    packed = utils.struct_pack(utils.LITTLE_ENDIAN, fmts.values(), values)
    fp.write(packed)
    fp.seek(0)
    # Read as Big Endian
    unpacked = utils.struct_unpack(fp, utils.BIG_ENDIAN, fmts.values())
    mydict = {}
    for field, val in zip(fmts.keys(), unpacked):
        mydict[field] = val
    assert mydict['uchar'] == values[0]
    assert mydict['int'] != values[1:5]
    assert mydict['short'] != values[5]
    assert mydict['float'] != values[6:9]
    assert mydict['string'] == values[9]
    fp.close()

def test_big_endian_struct():
    fp = StringIO()
    packed = utils.struct_pack(utils.BIG_ENDIAN, fmts.values(), values)
    fp.write(packed)
    fp.seek(0)
    unpacked = utils.struct_unpack(fp, utils.BIG_ENDIAN, fmts.values())
    mydict = {}
    for field, val in zip(fmts.keys(), unpacked):
        mydict[field] = val
    assert mydict['uchar'] == values[0]
    assert mydict['int'] == values[1:5]
    assert mydict['short'] == values[5]
    assert allclose(mydict['float'], values[6:9])
    assert mydict['string'] == values[9]
    fp.close()

if __name__ == '__main__':
    nose.runmodule()
