__docformat__ = 'restructuredtext'

import numpy as N, re
import sys, fpformat

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.datasource import DataSource, iswritemode
from neuroimaging.data_io.formats import utils, binary
from neuroimaging.data_io.formats._afniconstants import *
from neuroimaging.core.reference.axis import space, spacetime
from neuroimaging.core.reference.grid import SamplingGrid

class AFNIFormatError(Exception):
    """
    AFNI format exception
    """

## The header attributes can be organized in N parts:
## 1) Mandatory info
## 2) Time-dependent info (mandatory if time-axis is present)
## 3) "Almost mandatory" info
## 4) Notes info
## 5) Registration info
## 6) Misc info
## 7) Warping info
## 8) Talairach markers
## 9) User-defined tags
## 10) "Nearly Useless" info
##
## There seems to be no restriction on the order of the attributes in the
## header file. I'll make three attribute dictionaries: two mandatories,
## and one with everything else (even nearly useless info).

# The regexp parse string for any AFNI attribute
att_re = re.compile('type\s*=\s*(.*?)\n.*?name\s*=\s*(.*?)\n.*?count\s*=\s*(.*?)\n(.*)', re.DOTALL)

vol_header = odict((
    ('DATASET_RANK', (AFNI_integer, None)),
    ('DATASET_DIMENSIONS',(AFNI_integer, None)),
    ('TYPESTRING',(AFNI_string, None)),
    ('SCENE_DATA',(AFNI_integer, None)),
    ('ORIENT_SPECIFIC',(AFNI_integer, 3)),
    ('ORIGIN',(AFNI_float, 3)),
    ('DELTA',(AFNI_float, 3)),
))

time_header = odict((
    ('TAXIS_NUMS',(AFNI_integer, 3)),
    ('TAXIS_FLOATS',(AFNI_float, 5)),
    ('TAXIS_OFFSETS',(AFNI_float, None)),
))


extra_header = odict((
    ('IDCODE_STRING',(AFNI_string, None)),
    ('IDCODE_DATE',(AFNI_string, None)),
    ('BYTEORDER_STRING',(AFNI_string, None)),
    ('BRICK_STATS',(AFNI_float, None)),
    ('BRICK_TYPES',(AFNI_integer, None)),
    ('BRICK_FLOAT_FACS',(AFNI_float, None)),
    ('BRICK_LABS',(AFNI_string, None)),
    ('BRICK_STATAUX',(AFNI_float, None)),
    ('STAT_AUX',(AFNI_float, None)),
    ('HISTORY_NOTE',(AFNI_string, None)),
    ('NOTES_COUNT',(AFNI_integer, 1)),
    ('NOTES_NUMBER_001', (AFNI_string, None)),
    ('TAGALIGN_MATVEC',(AFNI_float, 12)),
    #('VOLREG_ROTCOM',(AFNI_string, None)),
    # name is actually VOLREG_ROTCOM_xxxxxx (xxxxxx = number of sub-brick)
    ('VOLREG_CENTER_OLD',(AFNI_string, None)),
    ('VOLREG_CENTER_BASE',(AFNI_string, None)),
    ('VOLREG_ROTPARENT_IDCODE',(AFNI_string, None)),
    ('VOLREG_ROTPARENT_NAME',(AFNI_string, None)),
    ('VOLREG_GRIDPARENT_IDCODE',(AFNI_string, None)),
    ('VOLREG_GRIDPARENT_NAME',(AFNI_string, None)),
    ('VOLREG_INPUT_IDCODE',(AFNI_string, None)),
    ('VOLREG_INPUT_NAME',(AFNI_string, None)),
    ('VOLREG_BASE_IDCODE',(AFNI_string, None)),
    ('VOLREG_BASE_NAME',(AFNI_string, None)),
    ('VOLREG_ROTCOM_NUM',(AFNI_integer, 1)),
    ('IDCODE_ANAT_PARENT',(AFNI_string, None)),
    ('TO3D_ZPAD',( AFNI_integer, 3)),
    ('IDCODE_WARP_PARENT',(AFNI_string, None)),
    ('WARP_TYPE',(AFNI_integer, 2)),
    ('WARP_DATA',(AFNI_float, None)),
    ('MARKS_XYZ',(AFNI_float, 30)),
    ('MARKS_LAB',(AFNI_string, None)),
    ('MARKS_FLAGS',(AFNI_integer, None)),
    ('MARKS_HELP', (AFNI_string, None)),
    ('TAGSET_NUM',(AFNI_integer, 2)),
    ('TAGSET_FLOATS',(AFNI_float, None)),
    ('TAGSET_LABELS',(AFNI_string, None)),
    ('LABEL_1',(AFNI_string, None)),
    ('LABEL_2',(AFNI_string, None)),
    ('DATASET_NAME',(AFNI_string, None)),
    ('DATASET_KEYWORD',(AFNI_string, None)),
    ('BRICK_KEYWORDS',(AFNI_string, None)),
))


##############################################################################
### AFNI should be considered a BinaryFormat. It's structure is:
###   A header file (*.HEAD), in plain-text format, with a minimum
###   set of attributes. Being plain-text, the header i/o architecture
###   will have to change a bit from the original BinaryFormat assumptions.
###
###   a brick-like data file (*.BRIK) of contiguous volume data in
###   x,y,z,t order (smallest -> largest stride)
##############################################################################

class AFNI(binary.BinaryFormat):
    """
    A class to read and write AFNI files
    """

    extensions = ('.HEAD', '.BRIK')

    def __init__(self, filename, mode='r', datasource=DataSource(), use_memmap=True, **kwds):

        # pick up these attributes:
        # mode, filename, filebase, clobber, header_file, data_file
        # header, ext_header, datasource (possibly), grid (possibly)
        binary.BinaryFormat.__init__(self, filename, mode, datasource, **kwds)
        if self.mode[0] is 'w':
            if not self.grid:
                raise AFNIFormatError("Can't create header info without a "\
                                      "grid object!")
            self.byteorder = utils.mybyteorders[sys.byteorder]
            self.dtype = self._fitdtype(N.dtype(kwds.get('dtype', N.float32)))
            self.dtype = self.dtype.newbyteorder(self.byteorder)
            self.fill_header()
            self.inform_canonical()
            self.write_header(clobber=self.clobber)
        else:
            self.read_header()
            # in the process get of inform_canonical get:
            #     dim
            #     scaling
            #     dtype
            #
            self.inform_canonical()
            # divine_byteorder() will inform self.dtype too
            self.divine_byteorder()

            if self.grid is None:
                # construct all in C-array ordering
                deltas = ( self.dt is not None and (self.dt,) or () ) + \
                         (self.dz, self.dy, self.dx)
                         
                origin = ( self.t0 is not None and (self.t0,) or () ) + \
                         (self.z0, self.y0,self.x0)
                         
                #origin = N.array(origin)
                shape = ( self.tdim is not None and (self.tdim,) or ()) + \
                        (self.zdim, self.ydim, self.xdim)
                        
                names = self.tdim and spacetime or space
                self.grid = SamplingGrid.from_start_step(names=names,
                                                         shape=shape,
                                                         start=N.array(origin),
                                                         step=deltas)
                # be satisfied with identity xform for now

        self.attach_data(use_memmap=use_memmap)
        #self.callback()

    def _fitdtype(self, dtype_request):
        # AFNI only does 4 data types, so try to find a match for
        # the requested dtype
        if dtype_request in N.sctypes['float']:
            return N.dtype(N.float32)
        # I guess if it's N.int8 cast up??
        if dtype_request in N.sctypes['int']:
            return N.dtype(N.int16)
        if dtype_request in N.sctypes['uint']:
            return N.dtype(N.uint8)
        if dtype_request in N.sctypes['complex']:
            return N.dtype(N.complex64)
        else:
            raise AFNIFormatError("not a valid dtype!")
    
    def _get_filenames(self):
        return self.filebase+'.HEAD', self.filebase+'.BRIK'

    def read_header(self):
        atype2dtype = {AFNI_float: N.float32,
                       AFNI_integer: N.int32}
        available_atts = dict()
        hdrfile = self.datasource.open(self.header_file, 'r')
        att_step = iter(re.split('type', hdrfile.read().strip()))
        att_step.next()
        try:
            while 1:
                att_str = 'type' + att_step.next()
                att = att_re.search(att_str).groups()
                atype = att[0]
                name = att[1]
                count = int(att[2])
                val = att[3].strip()
                if atype == AFNI_string:
                    val = val[1:-1].strip().split('~')
                    if len(val) == 1:
                        val = val[0]
                else:
                    val = N.array(re.split('\s*',val)).astype(atype2dtype[atype])
                available_atts[name] = val
        except StopIteration:
            pass

        # first get through mandatory attributes, raise exception if
        # something is missing
        for aname in vol_header.keys():
            try:
                val = available_atts.pop(aname)
            except KeyError:
                raise AFNIFormatError("mandatory attribute %s "\
                                      "was not found"%aname)
            self.header[aname] = val
            
        # if it seems to have a time-axis, go through mandatory taxis fields
        if available_atts.has_key('TAXIS_NUMS'):
            for aname in time_header.keys():
                # TAXIS_OFFSETS is only manditory if TAXIS_NUMS is set
                if aname == 'TAXIS_OFFSETS' and \
                       not self.header['TAXIS_NUMS'][1]:
                    continue
                try:
                    val = available_atts.pop(aname)
                except KeyError:
                    raise AFNIFormatError("mandatory taxis attribute %s "\
                                          "was not found, despite indication "\
                                          "of a time dimension"%aname)
                self.header[aname] = val
        # do the remaining available attributes
        for aname, val in available_atts.items():
            if extra_header.has_key(aname):
                self.header[aname] = val
            else:
                # IMPORTANT: could try VOLREG_ROTCOM_xxxxxx or something!
                print "Non-standard attribute %s skipped" % aname
        
    def inform_canonical(self):
        ## Get dims info
        ndim = self.canonical_fields['ndim'] = \
               self.header.has_key('TAXIS_NUMS')  and 4 or 3

        nbricks = self.nbricks = self.header['DATASET_RANK'][1]
        
        tdim = self.canonical_fields['tdim'] = ndim > 3 and nbricks or None

        (self.canonical_fields['xdim'],
         self.canonical_fields['ydim'],
         self.canonical_fields['zdim']) = self.header['DATASET_DIMENSIONS'][:3]

        ## get dim step and origin info
        (self.canonical_fields['dx'],
         self.canonical_fields['dy'],
         self.canonical_fields['dz']) =  tuple(self.header['DELTA'][:3])

        self.canonical_fields['dt'] = tdim and \
                                      self.header['TAXIS_FLOATS'][1] or None

        (self.canonical_fields['x0'],
         self.canonical_fields['y0'],
         self.canonical_fields['z0'],
         self.canonical_fields['t0']) = \
             tuple(self.header['ORIGIN'][:3]) + (tdim and (0,) or (None,))

        ## Get scaling info
        self.scales = self.header.get('BRICK_FLOAT_FACS', N.ones((nbricks,)))
        # when no scaling, BRICK_FLOAT_FACS is all 0
        N.putmask(self.scales, self.scales==0., 1.)
        self.canonical_fields['scaling'] = self.scales

        ## Get datatype info (which is NOT MANDATORY!!!!???)
        if self.header.has_key('BRICK_TYPES'):
            # there is a number for each brick--
            # if they're not all equal, we're not going to play
            types = self.header['BRICK_TYPES']
            if types.sum()/float(nbricks) != types[0]:
                raise AFNIFormatError("not all bricks are the same data type")
            else:
                self.dtype = N.dtype(AFNI_bricktype2dtype[types[0]])
        else:
            # try to guess?? damn!
            import os
            dsize = self.canonical_fields['zdim'] * \
                    self.canonical_fields['ydim'] * \
                    self.canonical_fields['xdim'] * nbricks
            fsize = os.stat(self.data_file).st_size
            sctype = {1:N.uint8, 2:N.int16,
                      4:N.float32, 8:N.complex64}[int(fsize/dsize)]
            self.dtype = N.dtype(sctype)
        self.canonical_fields['datasize'] = self.dtype.itemsize*8            

        # intent info is available, but I don't know what to do with it!

        ## since retrieving is awkward, map into self.__dict__ too
        for (k, v) in self.canonical_fields.items():
            self.__dict__[k] = v
        
    def divine_byteorder(self):
        # easiest case: it's in the header!
        if self.header.has_key('BYTEORDER_STRING'):
            self.byteorder = AFNI_att2byteorder[self.header['BYTEORDER_STRING']]
        # otherwise try to find out if max # in brick 1 matches expected?
        elif self.header.has_key('BRICK_STATS'):
            # min/max value for brick0 are in BRICK_STATS[0:1]
            scale_b0 = self.scales[0]
            maxval = self.header['BRICK_STATS'][1]
            readsize = self.zdim*self.ydim*self.xdim*self.dtype.itemsize
            a = N.reshape(
                N.fromstring(
                    self.datasource.open(self.data_file).read(readsize),
                    self.dtype),
                (self.zdim,self.ydim,self.xdim))
            brk_max = (a*scale_b0).max()
            try:
                N.testing.assert_almost_equal(brk_max, maxval)
                self.byteorder = utils.mybyteorders[sys.byteorder]
            except:
                self.byteorder = utils.mybyteorders['swapped']
                self.dtype.newbyteorder(self.byteorder)
        else:
            raise AFNIFormatError("no byteorder infomation in the header!")

    def fill_header(self):
        # let's do the basics for now
        grid = self.grid.python2matlab()
        self.ndim = ndim = grid.ndim
        shape = grid.shape
        self.nbricks = nbricks = ndim < 4 and 1 or shape[3]
        self.header['DATASET_RANK'] = (3, nbricks) 
        self.header['DATASET_DIMENSIONS'] = shape[:3]
        self.header['TYPESTRING'] = '3DIM_HEAD_FUNC' # ?? figure this out
        self.header['SCENE_DATA'] = (0, 0, 0)        # ?? too
        self.header['ORIENT_SPECIFIC'] = self.orientcode()
        # some problem with python2matlab changes the origin vals!
        self.header['ORIGIN'] = self.grid.mapping(N.array([0]*grid.ndim))[::-1][:3]
        self.header['DELTA'] = N.diag(grid.mapping.transform)[:3]
        if ndim > 3:
            self.header['TAXIS_NUMS'] = (nbricks, 0, UNITS_SEC_TYPE)
            self.header['TAXIS_FLOATS'] = (0, grid.mapping.transform[-1,-1],
                                           0, 0, 0)
        self.header['BYTEORDER_STRING'] = AFNI_byteorder2att[self.byteorder]
        self.header['BRICK_TYPES'] = N.array([AFNI_dtype2bricktype[self.dtype.type]]*nbricks)
        # put this to zero for now, it will get replaced in prewrite()
        # this will also set self.scales to N.ones() in inform_canonical()
        self.header['BRICK_FLOAT_FACS'] = N.zeros((nbricks,), N.float32)


    def orientcode(self):
        # not sure yet, here's some dummy info
        return (ORI_R2L_TYPE, ORI_P2A_TYPE, ORI_I2S_TYPE)

    def brickstats(self):
        # return minimum/maximum value per brick
        # take care to go through __getitem__ to get scaling kicked in
        nbricks, bricksize = (self.nbricks,
                              N.product(self.grid.shape)/self.nbricks)
        bstats = N.empty((nbricks*2,), N.float32)
        bstats[0::2] = N.reshape(self[:], (nbricks, bricksize)).min(axis=-1)
        bstats[1::2] = N.reshape(self[:], (nbricks, bricksize)).max(axis=-1)
        return bstats

    def write_header(self, hdrfile=None, clobber=False):
        if hdrfile:
            fp = isinstance(hdrfile, str) and open(hdrfile, 'w+') or hdrfile
        elif self.datasource.exists(self.header_file):
            if not clobber:
                raise IOError('file exists, but not allowed to clobber it')
            fp = self.datasource.open(self.header_file, 'w+')
        else:
            fp = open(self.datasource._fullpath(self.header_file), 'w+')


        for att, val in self.header.items():
            att_type, count = vol_header.has_key(att) and vol_header[att] or \
                              time_header.has_key(att) and time_header[att] or\
                              extra_header.has_key(att) and extra_header[att]
            
            ndecimal = 12
            if not count:
                count = len(val)
            if att_type == AFNI_integer:
                fp.write('type = ' + att_type + '\nname = ' + att + '\ncount = ' + `count` + '\n')
                for i in range(count):
                    fp.write(`val[i]` + ' ')
            elif att_type == AFNI_float:
                fp.write('type = ' + att_type + '\nname = ' + att + '\ncount = ' + `count` + '\n')
                for i in range(count):
                    fp.write(fpformat.fix(val[i], ndecimal) + ' ')
            elif att_type == AFNI_string:
                if isinstance(val, list):
                    cur_string = '~'.join([v.__str__() for v in val])
                else:
                    cur_string = val
                count = len(cur_string) + 1
                fp.write('type = ' + att_type + '\nname = ' + att + '\ncount = ' + `count` + '\n\'' + cur_string + '~')
            fp.write('\n\n')

        if not hdrfile or type(hdrfile) is not type(fp):
            fp.close()
        return


    def __getitem__(self, slicer):
        return N.asarray(self.postread(self.data[slicer].newbyteorder(self.byteorder), slicer))
        
    def __setitem__(self, slicer, data):
        if not iswritemode(self.data._mode):
            print "Warning: memapped array is not writeable! Nothing done"
            return
        self.data[slicer] = self.prewrite(data, slicer).astype(self.dtype)
        # assuming that at least BRICK_STATS has changed,
        # can write over header, wherever it is
        self.header['BRICK_STATS'] = self.brickstats()
        self.write_header(clobber=True)

    def prewrite(self, x, slicer):
        if not self.use_memmap:
             return x

        # try to cast in two cases:
        # 1 - we're replacing all the data of all bricks
        # 2 - if the maximum/maxima of the given slice of data exceeds
        #     the maximum/maxima of the brick(s) it slices into
        if isinstance(slicer, tuple):
            brick_slice = slicer[0]
        else:
            brick_slice = slicer
        # in case this is a simple index, fix to be a slice obj!
        if not isinstance(brick_slice, slice):
            brick_slice = slice(brick_slice, brick_slice+1)

        if self.ndim == 4:
            full_slice = isinstance(slicer, slice) and slice(None) or \
                         [isinstance(sl, slice) and slice(None) \
                          or N.newaxis for sl in slicer] + \
                          [slice(None)]*(self.ndim-len(slicer))
        else:
            full_slice = (None, slice(None))
            
        bmaxima = self.brickstats()[1::2][brick_slice]
        xmaxima = N.array([bx.max() for bx in x[full_slice]])
        if x.shape == self.data.shape or N.any(xmaxima > bmaxima):
            new_scales = self.scales.copy()
            for brk, dslice in enumerate(x[full_slice]):
                new_scales[brick_slice][brk] = \
                                utils.scale_data(N.squeeze(dslice), self.dtype,
                                                 self.scales[brick_slice][brk])
            if N.any(new_scales != self.scales):
                self.scales = new_scales.copy()
                # use new scales as a scratch array now for putmask
                N.putmask(new_scales, new_scales==1., 0.)
                self.header['BRICK_FLOAT_FACS'] = new_scales

        scale_slice = (brick_slice,) + (N.newaxis,)*(len(x.shape)-1)
        if self.dtype.type in N.sctypes['int']+N.sctypes['uint']:
            return N.round(x / self.scales[scale_slice])
        else:
            return x / self.scales[scale_slice]

    def postread(self, x, slicer):
        if not self.use_memmap:
             return x

        if self.scales is not None:
            if isinstance(slicer, tuple):
                brick_slice = slicer[0]
            else:
                brick_slice = slicer
            scale_slice = (brick_slice,) + (N.newaxis,)*(len(x.shape)-1)
            return x * self.scales[scale_slice]
        else:
            return x
            
