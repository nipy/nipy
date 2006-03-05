import os, types
import mincutils
from _mincconstants import *
import BrainSTAT.Base.Dimension as Dimension
import BrainSTAT.Base.Coordinates as Coordinates
import enthought.traits as traits
from BrainSTAT.Base import Warp

class MINCvar(traits.HasTraits):

    name = traits.Str()
    
    def __init__(self, **nc_atts):
        for nc_att, value in nc_atts.items():
            if nc_att != 'dimensions':
                self.setnc(nc_att, MINCatt(value))
            else:
                if value == (None, None):
                    self.dimensions = None
                else:
                    self.dimensions = value

    def setnc(self, name, value):
        if type(value) is type(MINCatt((1.0, NC_FLOAT))):
            if not hasattr(self, '__ncdict__'):
                self.__dict__['__ncdict__'] = {}
            self.__ncdict__[name] = value
        setattr(self, name, value.value)

class MINCdim(Dimension.RegularDimension, MINCvar):
    def __init__(self, name, step=None, start=None, length=0, dircos=None, **keywords):
        value = {}
        value['step'] = (step, NC_DOUBLE)
        value['start'] = (start, NC_DOUBLE)
        value['length'] = (length, NC_INT)
        if dircos:
            value['direction_cosines'] = (tuple(dircos), NC_DOUBLE)
        for key, att in keywords.items():
            if type(att) in [types.TupleType, types.ListType, mincutils.numpy.ArrayType]:
                test = 1
                for i in range(len(att)):
                    test *= (type(att[i]) is types.IntType)
                if test:
                    value[key] = (tuple(att), NC_INT)
                else:
                    value[key] = (tuple(att), NC_FLOAT)
            elif type(att) is types.StringType:
                value[key] = (att, NC_CHAR)
            else:
                raise TypeError, 'invalid attribute type for MINCdim'
        RegularDimension.__init__(self, name=name, length=length, step=step, start=start)

class MINCatt:
    def __init__(self, value):
        self.value = value[0]
        try:
            self.type = value[1]
        except:
            self.type = None

OPEN = 1
CLOSED = 0

class MINC:
    def __init__(self, filename, mode='r', nvector=-1, datatype=NC_SHORT, signtype=MI_UNSIGNED, modality='', frame_times=None, frame_widths=None, clobber=False, history='', mincid=MI_ERROR, create=False, special_atts={}, **keywords):
        
        if special_atts:
            for key, value in special_atts.items():
                if len(value) > 1:
                    special_atts[key] = value[0:2]
                else:
                    special_atts[key] = (NC_GLOBAL, value)
        try:
            stat = os.stat(filename)
            self.clobber = clobber
        except:
            self.clobber = True
        self.filename = filename
        if mode == 'r':
            self.open()
        else:
            dimensions = keywords['dimensions']
            if frame_times is None and frame_widths is None:
                dimnames = []
                for dimension in dimensions:
                    dimnames.append(dimension.name)

                if MItime in dimnames:
                    time_index = dimnames.index(MItime)
                    frame_times = dimensions[time_index].values()
                elif MItime_width in dimnames:
                    time_index = dimnames.index(MItime_width)
                    frame_widths = dimensions[time_index].values()
                    
            self.mincid = mincutils.minccreate(filename, dimensions, nvector=nvector, datatype=datatype, signtype=signtype, modality=modality, frame_times=frame_times, frame_widths=frame_widths, clobber=int(self.clobber), history=history, special_atts=special_atts)
            self.status = OPEN
        
        self.readheader()
        self.ndim = len(self.indim)
        return 

    def close(self, force=False):
        if self.status or force:
            self.status = CLOSED
            mincutils.mincclose(self.mincid)
        else:
            raise ValueError, 'file ' + self.filename + ' not open'

    def open(self, mode='r', force=False):
        self.mode = mode
        if not hasattr(self, 'mincid'):
            self.mincid = mincutils.mincopen(self.filename, mode=mode)
            self.readheader()
            self.status = OPEN
        else:
            if self.status:
                raise ValueError, 'file ' + self.filename + ' already open'
            else:
                self.mincid = mincutils.mincopen(self.filename, mode=mode)
                self.status = OPEN
                self.readheader()

    def readheader(self):
        header_data = mincutils.readheader(self.filename, mincid=self.mincid)

        for global_att in header_data['att'].keys():
            value = header_data['att'][global_att]
            setattr(self, global_att, MINCatt(value))
        
        ndim = len(header_data['dim'])

        self.indim = range(ndim)
        self.outdim = range(ndim)
        self.shape = range(ndim)
        self.step = range(ndim)
        self.start = range(ndim)

        dimlengths = {}
        dimnames = range(ndim)

        image_dims = list(header_data['var']['image']['dimensions'])

        # Make sure self.dimensions
        # has the same order as image->dimensions

        for dim in header_data['dim'].values():
            dimlengths[dim[0]] = dim[1]
            index = image_dims.index(dim[0])
            dimnames[index] = dim[0]
            if dim[0] == MIvector_dimension: # Vector dimension is
                                             # NOT a MINCvar so it does
                                             # not get set in the loop below
                setattr(self, dim[0], MINCdim(dim[0], step=0.0, length=dim[1], start=0.0))
                self.indim[index] = getattr(self, dim[0])
                self.outdim[index] = self.indim[index]
                self.shape[index] = dim[1]
                self.start[index] = 0.0
                self.step[index] = 0.0
        
        for var, value in header_data['var'].items():
            if var not in dimnames:
                setattr(self, var, MINCvar(name=var, **value))
            else:
                length = dimlengths[var]
                try:
                    step = value['step'][0]
                except:
                    step = 0.0
                try:
                    start = value['start'][0]
                except:
                    start = 0.0
                dircos = mincutils.getdircos(self.filename, var)
                if dircos:
                    setattr(self, var, MINCdim(var, step=step, length=length, start=start, dircos=dircos))
                else:
                    setattr(self, var, MINCdim(var, step=step, length=length, start=start))
                index = dimnames.index(var)

                self.outdim[index] = getattr(self, var)
                self.indim[index] = MINCdim(var, step=1.0, length=length, start=0.0)
            
                self.shape[index] = length
                self.step[index] = step
                self.start[index] = start
                
                if var == MItime:
                    try:
                        dimattr = getattr(self, MItime)
                        setattr(dimattr, 'value', mincutils.getvar(self.filename, MItime))
                    except:
                        dimattr = getattr(self, MItime_width)
                        setattr(dimattr, 'value', mincutils.getvar(self.filename, MItime_width))

        self.dimnames = tuple(dimnames)

        # Replace the 'dimensions' (string) attribute of each variable
        # with true MINCdims 

        for var in header_data['var'].keys():
            try: 
                dimensions = getattr(getattr(self, var), 'dimensions')
            except:
                dimensions = None
            if type(dimensions) in [types.ListType, types.TupleType]:
                new_dim = ()
                for i in range(len(dimensions)):
                    if dimensions[i] == MItime:
                        if hasattr(self, MItime):
                            dimname = MItime
                        else:
                            dimname = MItime_width
                    else:
                        dimname = dimensions[i]
                    new_dim = new_dim + (getattr(self, dimname),)
                setattr(getattr(self, var), 'dimensions', new_dim)

        self.shape = tuple(self.shape)

        # Setup affine transformation
                
        self.incoords = Coordinates.VoxelCoordinates('voxel', self.indim)
        self.outcoords = Coordinates.Coordinates('world', self.outdim)

        try:
            matrix = self._transform()
        except:
            matrix = self.incoords.transform()

        self.warp = Warp.Affine(self.incoords, self.outcoords, matrix)

    def _transform(self):
        """This method, (not yet implemented) determines the 4x4 (or larger) transformation matrix from the dircos attributes of the dimensions. """

        NA = mincutils.numpy
        ndim = self.outcoords.ndim
        transform = NA.zeros((ndim+1,)*2, NA.Float)
        for i in range(ndim):
            try:
                transform[i,i] = self.outcoords.dimensions[i].step
                transform[i,ndim] = self.outcoords.dimensions[i].start
            except:
                transform[i,i] = 1.0
                transform[i,ndim] = 0.0
        transform[ndim, ndim] = 1.0
        return transform

    def read(self, start, count, xdir = 'any', ydir = 'any', zdir = 'any', **keywords):
        return mincutils.mincextract(self.filename, start, count, xdir=xdir, ydir=ydir, zdir=zdir, mincid=self.mincid, **keywords)

    def write(self, start, data, set_minmax = True, offset = None, **keywords):
        return mincutils.mincwrite(self.filename, start, data, set_minmax=set_minmax, mincid=self.mincid, **keywords)

"""
URLPipe class expects this.
"""

creator = MINC
valid = ['.mnc']

