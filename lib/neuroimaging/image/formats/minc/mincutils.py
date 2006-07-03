from neuroimaging.image.formats.minc import _mincutils

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
    axes = N.zeros((3, 3), N.float64)
    axes[0] = xdircos
    axes[1] = ydircos
    axes[2] = zdircos

    lengths = N.zeros((3,), N.float64)
    start = N.zeros((3,), N.float64)

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





