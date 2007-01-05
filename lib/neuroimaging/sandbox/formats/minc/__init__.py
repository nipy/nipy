import os, types

import numpy as N

from neuroimaging import traits

from neuroimaging.core.reference import mapping, axis, coordinate_system
from neuroimaging.data_io.formats import Format
from neuroimaging.data_io.formats.minc import _mincutils
from neuroimaging.data_io.formats.minc import _mincconstants as mc


class Dimension(axis.VoxelAxis):
    pass

class Variable(traits.HasTraits):
    """
    MINC variable class.
    """
    
    name = traits.Int(desc='MINC variable name')
    nctype = traits.Trait(mc.NC_TYPES, desc='NetCDF type of the variable')
    length = traits.Trait(mc.NC_TYPES, desc='Length of MINC variable')
    dimensions = traits.ListInstance(Dimension, desc='List of dimensions for variable.')
    value = traits.Any(desc='Value of variable')

class Attribute(traits.HasTraits):
    """
    MINC attribute class.
    """

    variable = traits.Instance(Variable, desc='Parent of attribute.')
    name = traits.Int(desc='MINC attribute name')
    nctype = traits.Trait(mc.NC_TYPES, desc='NetCDF type of the attribute')
    length = traits.Int(0, desc='Length of MINC variable')
    value = traits.Any(desc='Value of variable')

    def __init__(self, name, nctype, *value):
        self.name = name
        self.nctype = nctype
        self.value = value
        try:
            self.length = len(value)
        except:
            pass

class DimensionVariable(Variable, axis.VoxelAxis):
    pass

class MINC(Format):

    mincid = traits.Int(desc="NetCDF ID")

    def __init__(self, filename=None, datasource=DataSource(), grid=None,
                 sctype=N.float64, **keywords):
        
        self.filename = filename
        if mode == 'r':
            self.open()
        else:
            dimensions = keywords['dimensions']
            if frame_times is None and frame_widths is None:
                dimnames = []
                for dimension in dimensions:
                    dimnames.append(dimension.name)

                if mc.MItime in dimnames:
                    time_index = dimnames.index(mc.MItime)
                    frame_times = dimensions[time_index].values()
                elif mc.MItime_width in dimnames:
                    time_index = dimnames.index(mc.MItime_width)
                    frame_widths = dimensions[time_index].values()
                    
            self.mincid = minccreate(filename, dimensions,
                                               nvector=nvector,
                                               datatype=datatype,
                                               signtype=signtype,
                                               modality=modality,
                                               frame_times=frame_times,
                                               frame_widths=frame_widths,
                                               clobber=int(self.clobber),
                                               history=history,
                                               special_atts=special_atts)
        
        self.readheader()
        self.ndim = len(self.indim)

    def close(self, force=False):
        mincclose(self.mincid)

    def open(self, mode='r', force=False):
        self.mode = mode
        self.mincid = mincopen(self.filename, mode=mode)
        self.readheader()

    def readheader(self):
        header_data = readheader(self.filename, mincid=self.mincid)

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
            if dim[0] == mc.MIvector_dimension: # Vector dimension is
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
                dircos = getdircos(self.filename, var)
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
                
                if var == mc.MItime:
                    try:
                        dimattr = getattr(self, mc.MItime)
                        setattr(dimattr, 'value', getvar(self.filename, mc.MItime))
                    except:
                        dimattr = getattr(self, mc.MItime_width)
                        setattr(dimattr, 'value', getvar(self.filename, mc.MItime_width))

        self.dimnames = tuple(dimnames)

        # Replace the 'dimensions' (string) attribute of each variable
        # with true MINCdims 

        for var in header_data['var'].keys():
            try: 
                dimensions = getattr(getattr(self, var), 'dimensions')
            except:
                dimensions = None
            if isinstance(dimensions, list) or isinstance(dimensions tuple):
                new_dim = ()
                for d in dimensions:
                    if d == mc.MItime:
                        if hasattr(self, mc.MItime):
                            dimname = mc.MItime
                        else:
                            dimname = mc.MItime_width
                    else:
                        dimname = d
                    new_dim = new_dim + (getattr(self, dimname),)
                setattr(getattr(self, var), 'dimensions', new_dim)

        self.shape = tuple(self.shape)

        # Setup affine transformation
                
        self.incoords = coordinate_system.CoordinateSystem('voxel', self.indim)
        self.outcoords = coordinate_system.DiagonalCoordinateSystem('world', self.outdim)

        try:
            matrix = self._transform()
        except:
            matrix = self.outcoords.transform()

        self.mapping = mapping.Affine(self.incoords, self.outcoords, matrix)

    def _transform(self):
        """This method, (not yet implemented) determines the 4x4 (or larger) transformation matrix from the dircos attributes of the dimensions. """

        ndim = self.outcoords.ndim
        transform = N.zeros((ndim+1,)*2)
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
        return mincextract(self.filename, start, count, xdir=xdir, ydir=ydir, zdir=zdir, mincid=self.mincid, **keywords)

    def write(self, start, data, set_minmax = True, offset = None, **keywords):
        return mincwrite(self.filename, start, data, set_minmax=set_minmax, mincid=self.mincid, **keywords)


from neuroimaging.data_io.formats.minc import _mincutils

import numpy as N

# Load constants

INVALID_DATA = _mincutils._invalid_data()

import _mincconstants as mc

# Import C functions

def mincopen(filename, mode='r'):
    if mode in ['w', 'r+']:
        nc_mode = mc.NC_WRITE
    elif mode == 'r':
        nc_mode = mc.NC_NOWRITE
    else:
        raise ValueError, 'mode must be one of ' + `['r', 'r+', 'w']`
    w = _mincutils._mincopen(filename, mode=nc_mode)
    return _mincutils._mincopen(filename, mode=nc_mode)

def mincclose(mincid):
    return _mincutils._mincclose(mincid)

def mincextract(filename, start, count, xdir = 'any', ydir = 'any', zdir = 'any', mincid=mc.MI_ERROR, **keywords):
    if keywords.has_key('mincid'):
        mincid = keywords.pop('mincid')

    value = _mincutils._mincextract(filename, start, count, xdir = xdir, ydir = ydir, zdir = zdir)
    return value

def mincwrite(filename, start, data, set_minmax = False, mincid=mc.MI_ERROR, **keywords):
    count = data.shape
    data_max = ()
    data_min = ()
    step, shape, start_ignore, names = _mincutils._getinfo(filename)
    if mc.MIvector_dimension in names:
        ndims = len(start) - 1
    else:
        ndims = len(start)
    ndims_max = ndims - 2
    if data.typecode() != N.float64:
        data_float = data.astype(N.float64)
    else:
        data_float = data
    if set_minmax:
        if ndims_max == 1:
            for i in range(count[0]):
                cur_data = data_float[i].flat
                try:
                    data_max = data_max + (N.nanmax(cur_data),)
                    data_min = data_min + (N.nanmin(cur_data),)
                except:
                    data_max = data_max + (0.0,)
                    data_min = data_min + (0.0,)
        elif ndims_max == 2:
            for i in range(count[0]):
                for j in range(count[1]):
                    cur_data = data_float[i,j].flat
                    try:
                        data_max = data_max + (N.nanmax(cur_data),)
                        data_min = data_min + (N.nanmin(cur_data),)
                    except:
                        data_max = data_max + (0.0,)
                        data_min = data_min + (0.0,)
    else:
        data_max = None
        data_min = None

    value = _mincutils._mincwrite(filename, start, count, data_float, set_minmax=set_minmax, data_max=data_max, data_min=data_min)
    del(data_max)
    del(data_min)
    del(data_float)
    return value

## Format for atts: {attname=(varname, attvalue)} where varname can be NC_GLOBAL
def minccreate(filename, dimensions,
               nvector=-1,
               datatype=mc.NC_SHORT,
               signtype=mc.MI_UNSIGNED,
               modality='',
               frame_times=None,
               frame_widths=None,
               clobber=mc.FALSE,
               history='',
               special_atts={}):
    return _mincutils._minccreate(filename, dimensions,
                                  datatype=datatype,
                                  nvector=nvector,
                                  signtype=signtype,
                                  modality=modality,
                                  frame_widths=frame_widths,
                                  frame_times=frame_times,
                                  clobber=clobber,
                                  special_atts=special_atts,
                                  history=history)

def getinfo(filename, varname = mc.MIimage, mincid=mc.MI_ERROR):
    step, start, shape, names = _mincutils._getinfo(filename, varname=varname, mincid=mincid)
    return (step, start, shape, names)

def getvar(filename, varname, mincid=mc.MI_ERROR):
    step, start, shape, names = getinfo(filename, varname = varname, mincid=mincid)
    start = (0,) * len(shape)
    return _mincutils._getvar(filename, varname, start, shape)

def getdircos(filename, dimname, mincid=mc.MI_ERROR):
    return _mincutils._getdircos(filename, dimname, mincid=mincid)

def readheader(filename, mincid=mc.MI_ERROR):
    return _mincutils._readheader(filename, mincid)

## Code to implement conversion of origin to start -- not implemented yet

def convert_origin_to_start(origin, xdircos, ydircos, zdircos):
    axes = N.zeros((3, 3))
    axes[0] = xdircos
    axes[1] = ydircos
    axes[2] = zdircos

    lengths = N.zeros((3,))
    start = N.zeros((3,))

    for i in range(3):
        d1, d2 = (i, int(N.fmod(i+1, 3)))
        normal = cross(axes[d1], axes[d2])
        lengths[i] = N.add.reduce(normal**2)
        if lengths[i] == 0.0:
            raise ValueError, 'axes ' + `(d1, d2)` + ' are parallel'
        if length == 0.0:
            raise ValueError, 'axis ' + `i` + ' has zero length'
    for i in range(3):
        d1, d2 = (int(N.fmod(i+1, 3)), int(N.fmod(i+2, 3)))
        normal = N.cross(axes[d1], axes[d2])

        denom = N.innerproduct(axes[i], normal)
        numer = N.innerproduct(origin, normal)

        if denom != 0.0:
            start[i] = lengths[i] * numer / denom
        else:
            start[i] = 0.0
    return start

