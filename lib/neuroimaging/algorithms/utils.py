import sys
import numpy as N


def fwhm2sigma(fwhm):
    """
    Convert a FWHM value to sigma in a Gaussian kernel.
    """
    return fwhm / N.sqrt(8 * N.log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value.
    """
    return sigma * N.sqrt(8 * N.log(2))


def readbrick(infile, start, count, shape, offset=0, intype=N.int32, outtype=N.float64, byteorder=sys.byteorder, return_tell = True, check_nan=True, fill=0.0):
 	
    if return_tell:
        try:
            startpos = infile.tell()
        except:
            infile = file(infile.name, 'rb+')
            startpos = 0

        ndim = len(shape)

        if not outtype:
            outtype = intype

        # How many dimensions are "full" slices

        nslicedim = 0
        i = ndim - 1
        while count[i] is shape[i] and i >= 0:
            nslicedim = nslicedim + 1
            i = i - 1

        if nslicedim:
            nslice = N.product(shape[(ndim - nslicedim):])
        else:
            nslice = count[:-1]
            nslicedim = 1

        nloopdim = ndim - nslicedim

        test = N.product(N.less_equal(N.array(start) + N.array(count), N.array(shape)))
        if not test:
            raise ValueError, 'start+count not <= shape'

        nloop = N.product(count[nloopdim:])
        nskip = N.product(shape[nloopdim:])
        ntotal = N.product(count)
        return_value = N.zeros((ntotal,), outtype)

        elsize = intype.bytes

        shape_reverse = list(shape)
        shape_reverse.reverse()
        strides = [1] + list(N.multiply.accumulate(shape_reverse)[:-1])
        strides.reverse()
        strides = N.array(strides)
        strides = strides * elsize

        infile.seek(offset + N.add.reduce(start * strides))

        index = 0
        while index < ntotal:
            indata = N.fromfile(infile, intype, shape=(nloop,))
            if byteorder != sys.byteorder:
                indata.byteswap()
            return_value[index:(index+nloop)] = indata.astype(outtype)
            infile.seek((nskip - nloop) * elsize, 1)
            index = index + nloop

            return_value.setshape(count)

        if return_tell:
            infile.seek(startpos, 0)
        return return_value


def writebrick(outfile, start, data, shape, offset=0, outtype=None, byteorder=sys.byteorder, return_tell = True):
    if return_tell:
        try:
            startpos = outfile.tell()
        except:
            outfile = file(outfile.name, 'rb+')
            startpos = 0
        
    if outtype:
        outdata = data.astype(outtype)
    else:
        outdata = data
        outtype = outdata.dtype
        
    if byteorder != sys.byteorder:
        outdata.byteswap()

    outdata.shape = outdata.size
    
    count = data.shape
    ndim = len(shape)

    # How many dimensions are "full" slices

    nslicedim = 0
    i = ndim - 1
    while count[i] is shape[i] and i >= 0:
        nslicedim += 1
        i -= 1

    if nslicedim:
        nslice = N.product(shape[(ndim - nslicedim):])
    else:
        nslice = count[:-1]
        nslicedim = 1

    nloopdim = ndim - nslicedim

    test = N.product(N.less_equal(N.array(start) + N.array(count), N.array(shape)))
    if not test:
        raise ValueError, 'start+count not <= shape'

    nloop = N.product(count[nloopdim:])
    nskip = N.product(shape[nloopdim:])
    ntotal = N.product(count)

    elsize = outdata.dtype.itemsize

    shape_reverse = list(shape)
    shape_reverse.reverse()
    strides = [1] + list(N.multiply.accumulate(shape_reverse)[:-1])
    strides.reverse()

    strides = N.array(strides, N.int64)
    strides = strides * elsize

    outfile.seek(offset + N.add.reduce(start * strides))

    index = 0
    while index < ntotal:
        outdata[index:(index+nloop)].tofile(outfile)
        outfile.seek((nskip - nloop) * elsize, 1)
        index = index + nloop
    if return_tell:
        outfile.seek(startpos, 0)

    outfile.flush()
    del(outdata)

    return 

