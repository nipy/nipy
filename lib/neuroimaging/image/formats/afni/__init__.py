import os, string, re, sys, fpformat, types, tempfile, time, random, csv
import BrainSTAT.Base.Dimension as Dimension
import BrainSTAT.Base.Coordinates as Coordinates
from BrainSTAT.Base import Mapping
from BrainSTAT import Utils
from numpy import *
from _afniconstants import *

att_re = re.compile('type\s*=\s*(.*?)\n.*?name\s*=\s*(.*?)\n.*?count\s*=\s*(.*?)\n(.*)', re.DOTALL)

AFNI_float = 'float-attribute'
AFNI_integer = 'integer-attribute'
AFNI_string = 'string-attribute'

AFNI_UChar = 0
AFNI_Short = 1
AFNI_Float = 3
AFNI_Complex = 5

AFNI_byteorder = {'big':'MSB_FIRST', 'little':'LSB_FIRST'}

AFNI_brick_types = {AFNI_UChar:UInt8,
                   AFNI_Short:Int16,
                   AFNI_Float:Float32,
                   AFNI_Complex:Complex32
                   }

AFNI_exts = ['.HEAD', '.BRIK']
for end in ['HEAD', 'BRIK']:
    for space in ['tlrc', 'orig', 'acpc']:
        AFNI_exts.append(space + '.' + end)

numpy_type = type(array([3,4]))

class AFNI:
    '''A class to read and write AFNI format images. 

    >>> from BrainSTAT import *
    >>> test = VImage(testfile('anat+orig.HEAD'))
    >>> print test.shape, test.spatial_shape
    (124, 256, 256) (124, 256, 256)

    '''

    def __init__(self, filename, mode='r', datatype = AFNI_Float, clobber=False, nvector=-1, need_brick=True, **keywords):
        self.keywords = keywords
        self.mode = mode
        self.filebase, self.fileext = os.path.splitext(filename)
        if len(self.filebase.split('+')) > 1:
            self.filebase, self.filespace = self.filebase.split('+')
        else:
            self.filespace = 'orig'
        try:
            stat = os.stat(filename)
            self.clobber = clobber
        except:
            self.clobber = True

        if mode == 'r':
            self.hdrfile = file(self.filebase + '+' + self.filespace + '.HEAD', 'r')
            try:
                if keywords['no_brick']:
                    self.no_brick = True
                else:
                    self.brikfile = file(self.filebase + '+' + self.filespace +  + '.BRIK', 'rb')
            except:
                self.brikfile = file(self.filebase + '+' + self.filespace + '.BRIK', 'rb')
        elif mode == 'r+' and self.clobber:
            self.datatype = datatype
            ORIGIN = []
            DELTA = []
            DATASET_DIMENSIONS = []
            TAXIS_NUMS = None
            TAXIS_FLOATS = None

            self.dimensions = keywords['dimensions']

            for dim in self.dimensions:
                if dim.name != 'time': # == MItime
                    DELTA.append(float(dim.step))
                    ORIGIN.append(float(dim.start))
                    DATASET_DIMENSIONS.append(int(dim.length))
                else:
                    TAXIS_NUMS = [int(dim.length), 0, UNITS_SEC_TYPE]
                    TAXIS_FLOATS = [float(dim.start), float(dim.step), 0, 0]
                    for i in range(2):
                        if type(TAXIS_FLOATS[i]) not in [types.IntType, types.FloatType]:
                            TAXIS_FLOATS[i] = 0.0

            if TAXIS_NUMS is not None:
                DATASET_RANK = array((3, TAXIS_NUMS[0]) + (0,) * 6)
                if nvector > 0:
                    raise ValueError, 'nvector and TAXIS cannot be simultaneuosly defined'
            elif nvector > 0:
                DATASET_RANK = array((3, nvector) + (0,) * 6)
                self.nvector = nvector
            else:
                DATASET_RANK = array((3, 1) + (0,) * 6)
                self.nvector = 0

            self.hdrfile = file(self.filebase + '+' + self.filespace + '.HEAD', 'w')
            self.brikfile = file(self.filebase + '+' + self.filespace + '.BRIK', 'wb')
            self.DATASET_DIMENSIONS = DATASET_DIMENSIONS[::-1]
            if DATASET_RANK is None:
                DATASET_RANK = array((3, 1) + (0,) * 6)
            self.DATASET_RANK = DATASET_RANK
            self.ORIGIN = ORIGIN[::-1]
            self.DELTA = DELTA[::-1]
            for att in AFNI_defaults.keys():
                try:
                    setattr(self, att, keywords[att])
                except:
                    if AFNI_defaults[att] is not None:
                        if type(AFNI_defaults[att]) in [types.BuiltinFunctionType, types.FunctionType, types.MethodType]:
                            if type(AFNI_defaults[att]) is types.MethodType:
                                setattr(self, att, AFNI_defaults[att](self))
                            else:
                                setattr(self, att, AFNI_defaults[att]())
                        else:
                            setattr(self, att, AFNI_defaults[att])
            self.writeheader(self.hdrfile)
            self.hdrfile.close()
        elif self.clobber:
            self.brikfile = file(self.filebase + '+' + self.filespace + '.BRIK', 'rb+')
            self.hdrfile = file(self.filebase + '+' + self.filespace + '.HEAD', 'r+')
        else:
            raise ValueError, 'clobber does not agree with mode'
                
        self.readheader()
        try:
            self.need_scale = (add.reduce(equal(self.BRICK_FLOAT_FACS, 0)) == 0)
        except:
            self.need_scale = False

        self.shape = tuple(self.DATASET_DIMENSIONS)[0:3][::-1]
        self.step = tuple(self.DELTA)[::-1]
        self.start = tuple(self.ORIGIN)[::-1]
        self.size = multiply.reduce(array(list(self.shape)))
        
        if self.DATASET_RANK[1] > 1:
            self.shape = (self.DATASET_RANK[1],) + self.shape
            if hasattr(self, 'TAXIS_FLOATS'):
                self.step = (self.TAXIS_FLOATS[1],) + self.step
            else:
                self.step = (None,) + self.step
            if hasattr(self, 'TAXIS_FLOATS'):
                self.start = (self.TAXIS_FLOATS[0],) + self.start
            else:
                self.start = (None,) + self.start
        self.generate_dimensions()

    def generate_orient_specific(self):
        orient_specific = []
        for dim in self.dimensions:
            if dim.name != 'time':
                orient_specific.append(AFNI_orientations[(dim.name, (float(dim.step) > 0))])
        return array(orient_specific[::-1])

    def generate_dimensions(self):
        self.indim = ()
        self.outdim = ()
        signs = ()
        dimnames = ()
        for i in range(3):
            dimname, sign = AFNI_orientations_inv[self.ORIENT_SPECIFIC[::-1][i]]
            dimnames = dimnames + (dimname,)
            signs = signs + (sign * 2.0 - 1,)
        if self.ndim == 4:
            if self.nvector == 0:
                dimnames = ('time',) + dimnames
                signs = (1.0,) + signs
            else:
                dimnames = ('vector_dimension',) + dimnames
                signs = (1.0,) + signs
                self.step = (0.0,) + self.step[1:]
                self.start = (0.0,) + self.start[1:]
        for i in range(self.ndim):
            self.indim = self.indim + (Dimension.RegularDimension(name=dimnames[i], length=self.shape[i], start=0.0, step=1.0),)
            self.outdim = self.outdim + (Dimension.RegularDimension(name=dimnames[i], length=self.shape[i], start=self.start[i], step=abs(self.step[i]) * signs[i]),)
        
        # Setup affine transformation
                
        self.incoords = Coordinates.VoxelCoordinates('voxel', self.indim)
        self.outcoords = Coordinates.OrthogonalCoordinates('world', self.outdim)

        if self.keywords.has_key('xfmurl'):
            matrix = self._transform(url=self.keywords['xfmurl'])
        else:
            try:
                matrix = self._transform()
            except:
                matrix = self.incoords.transform()

        self.mapping = Mapping.Affine(self.incoords, self.outcoords, matrix)

    def _transform(self, url=None):
        """Tries to find a '.mat' file for an SPM type 4x4 transformation matrix."""
        if url is None:
            url = self.filebase + '.mat'
        return Mapping.fromurl(url, ndim=self.ndim)

    def generate_brick_labs(self, base='BRICK '):
        BRICK_LABS = []
        for i in range(self.DATASET_RANK[1]):
            BRICK_LABS.append(base + `i`)
        return BRICK_LABS

    def generate_brick_types(self):
        BRICK_TYPES = [self.datatype]*self.DATASET_RANK[1]
        return BRICK_TYPES

    def __setattr__(self, name, value):
        if not hasattr(self, 'header_atts'):
            self.__dict__['header_atts'] = {}
        if name in AFNI_header_atts.keys():
            attype, attlen = AFNI_header_atts[name]
            if attype is types.StringType:
                self.header_atts[name] = (AFNI_string, len(value))
            elif attlen is None:
                attlen = len(list(value))
            elif len(list(value)) != attlen:
                value = filter(lambda x: x not in  AFNI_missing, list(value))
                if len(list(value)) != attlen:
                    raise ValueError, 'attribute ' + name + ':' + `value` + ' has wrong length -- should be ' + `attlen`
            if attype is types.FloatType:
                self.header_atts[name] = (AFNI_float, attlen)
            elif attype is types.IntType:
                self.header_atts[name] = (AFNI_integer, attlen)
        self.__dict__[name] = value

    def close(self, header=True, brick=True):
        if header:
            self.hdrfile.close()
        if brick:
            self.brikfile.close()

    def open(self, mode='r', header=True, brick=True):
        if mode != 'r' and not self.clobber:
            raise ValueError, 'clobber does not agree with mode'

        if mode == 'r':
            mode = 'rb'
        if mode == 'w':
            mode = 'wb'
        elif mode == 'r+':
            mode = 'rb+'

        if header:
            try:
                self.hdrfile = file(self.hdrfile.name, mode=mode)
            except:
                pass
        if brick:
            try:
                self.brikfile = file(self.brikfile.name, mode=mode)
            except:
                pass

    def readheader(self):
        self.close(brick=False)
        self.open(mode='r', brick=False)
        self.header_atts = {}
        atts = iter(re.split('type', self.hdrfile.read().strip()))
        atts.next()
        try:
            while 1:
                att_str = 'type' + atts.next()
                att = att_re.search(att_str).groups()
                att_type = att[0]
                name = att[1]
                count = string.atoi(att[2])
                value = att[3].strip()
                if att_type == 'string-attribute':
                    value = value[1:-1]).strip().split('~')
                    if len(value) == 1:
                        value = value[0]
                elif att_type == 'integer-attribute':
                    att_array = zeros((count,), Int)
                    value = re.split('\s*', value)
                    for i in range(count):
                        att_array[i] = string.atoi(value[i])
                    value = att_array
                elif att_type == 'float-attribute':
                    att_array = zeros((count,), Float)
                    value = re.split('\s*', value)
                    for i in range(count):
                        att_array[i] = float(value[i])
                    value = att_array
                setattr(self, name, value)
        except StopIteration:
            pass
        
        self.ndim = 3 + (self.DATASET_RANK[1] > 1)
        if self.ndim == 4 and hasattr(self, 'TAXIS_NUMS'):
            self.nvector = 0
        else:
            self.nvector = self.DATASET_RANK[1]
        return
    

    def writeheader(self, headerfile, ndecimal = 2):
        for key, att in self.header_atts.items():
            att_type = att[0]
            count = att[1]
            try:
                reverse = att[2]
            except:
                reverse = False
            value = getattr(self, key)
            if reverse:
                value = value[::-1]
            if att_type == AFNI_integer:
                headerfile.write('type = ' + att_type + '\nname = ' + key + '\ncount = ' + `count` + '\n')
                for i in range(count):
                    headerfile.write(`value[i]` + ' ')
            elif att_type == AFNI_float:
                headerfile.write('type = ' + att_type + '\nname = ' + key + '\ncount = ' + `count` + '\n')
                for i in range(count):
                    headerfile.write(fpformat.fix(value[i], ndecimal) + ' ')
            elif att_type == AFNI_string:
                if type(value) is types.ListType:
                    cur_string = map(lambda x: x.__str__(), value).join('~')
                else:
                    cur_string = value
                count = len(cur_string) + 1
                headerfile.write('type = ' + att_type + '\nname = ' + key + '\ncount = ' + `count` + '\n\'' + cur_string + '~')
            headerfile.write('\n\n')
        return

    def write(self, start, data, offset = 0, **keywords):
        self.close()
        self.open(mode='r+',header=False)
        if self.BYTEORDER_STRING == 'LSB_FIRST':
            byteorder = 'little'
        else:
            byteorder = 'big'

        shape_d = len(data.shape)
        if shape_d != len(self.shape):
            data.shape = (1,) * (len(self.shape) - shape_d) + data.shape

        count = data.shape
        total_offset = offset

        if self.ndim > 3:
            
            for i in range(start[0]):
                type = AFNI_brick_types[self.BRICK_TYPES[i]]
                total_offset += self.size * type.bytes

            for i in range(count[0]):
                type = AFNI_brick_types[self.BRICK_TYPES[start[0] + i]]
                try:
                    fac = self.BRICK_FLOAT_FACS[start[0] + i]
                except:
                    fac = 0
                if fac:
                    outdata = data[i].astype(Float) / fac
                else:
                    outdata = data[i]
                Utils.brickutils.writebrick(self.brikfile, start[1:], data[i], self.shape[1:], byteorder = byteorder, outtype = type, offset = total_offset)
                total_offset += type.bytes * self.size
        else:
            type = AFNI_brick_types[self.BRICK_TYPES[0]]
            try:
                fac = self.BRICK_FLOAT_FACS[0]
            except:
                fac = 0
            if fac:
                outdata = data.astype(Float) / fac
            else:
                outdata = data
            Utils.brickutils.writebrick(self.brikfile, start, data, self.shape, byteorder = byteorder, outtype = type)
        return
        
    def read(self, start, count, offset = 0, **keywords):
        if hasattr(self, 'no_brick'):
            if self.no_brick:
                raise ValueError, 'no .BRIK file associated with this .HEAD file'
        if self.BYTEORDER_STRING == 'LSB_FIRST':
            byteorder = 'little'
        else:
            byteorder = 'big'

        total_offset = offset

        if len(start) > 3:
            for i in range(start[0]):
                type = AFNI_brick_types[self.BRICK_TYPES[i]]
                total_offset += self.size * type.bytes
            return_value = ()
            for i in range(start[0], start[0] + count[0]):
                type = AFNI_brick_types[self.BRICK_TYPES[i]]
                try:
                    fac = self.BRICK_FLOAT_FACS[i]
                except:
                    fac = 0
                value = Utils.brickutils.readbrick(self.brikfile, start[1:], count[1:], self.shape[1:], byteorder = byteorder, intype = type, offset = total_offset)
                if fac:
                    import sets
                    value = 1.0 * fac * value
                return_value = return_value + (value,)
                total_offset += type.bytes * self.size
            return_value = array(return_value)
        else:
            type = AFNI_brick_types[self.BRICK_TYPES[0]]
            try:
                fac = self.BRICK_FLOAT_FACS[0]
            except:
                fac = 0
            return_value = Utils.brickutils.readbrick(self.brikfile, start, count, self.shape, byteorder = byteorder, intype = type)
            if fac:
                return_value *= fac
        return return_value

AFNI_defaults = {'SCENE_DATA':array((AFNI_view['orig'], FUNC_BUCK_TYPE, AFNI_typestring['3DIM_HEAD_FUNC']) + (-999,)* 5),
                 'TYPESTRING':'3DIM_HEAD_FUNC',
                 'TAXIS_FLOATS':None,
                 'TAXIS_NUMS':None,
                 'TAXIS_OFFSETS':None,
                 'IDCODE_DATE':time.asctime,
                 'IDCODE_STRING':lambda : time.asctime() + '_%(number)03d' %{'number':10000*random.uniform(0,1)},
                 'HISTORY_NOTE':None,
                 'BRICK_TYPES':AFNI.generate_brick_types,
                 'BYTEORDER_STRING':AFNI_byteorder[sys.byteorder],
                 'BRICK_STATS':None,
                 'BRICK_FLOAT_FACS':None,
                 'BRICK_LABS':AFNI.generate_brick_labs,
                 'BRICK_STATAUX':None,
                 'STAT_AUX':None,
                 'ORIENT_SPECIFIC':AFNI.generate_orient_specific
                 }

def read1D(_1Dstring, try1D=True, delimiter=None):
    '''A little function to read 1D files a la AFNI notation.

    

    '''

    try:
        filename, columns = _1Dstring.split('[')
    except:
        filename = _1Dstring
        columns = '0]'

    data = []
    keep = []

    if delimiter is None:
        reader = csv.reader(file(filename))
        for row in reader:
            data.append(map(string.atof, string.split(string.strip(row[0]))))
    else:
        reader = csv.reader(file(filename), delimiter=delimiter)
        for row in reader:
            data.append(map(string.atof, row))
    data = numpy.array(data, Float)
    columns = '[' + columns
    try:
        columns = eval(columns)
        for i in columns:
            keep.append(list(data[:,i].flat))
    except:
        tmp = numpy.transpose(eval('data[:,' + columns[1:-1] + ']'))
        for i in range(tmp.shape[0]):
            keep.append(list(tmp[i].flat))
    if try1D and numpy.array(keep).shape[0] == 1:
        keep = keep[0]
    return numpy.array(keep)

"""
URLPipe class expects this.
"""

creator = AFNI
valid = AFNI_exts

