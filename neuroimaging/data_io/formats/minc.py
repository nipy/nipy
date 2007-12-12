__docformat__ = 'restructuredtext'

import os
from tempfile import mkstemp

import numpy as N

from scipy.io.netcdf import netcdf_file as netcdf

from neuroimaging.data_io.datasource import DataSource, Cache, iszip, ensuredirs, unzip
from neuroimaging.data_io.formats.format import Format

from neuroimaging.core.reference.grid import SamplingGrid

from ecat7 import CacheData

class MINC(Format):
    """
    A Class to read MINC format images. No write support yet
    	
    TODO: 
    * handle direction cosines (need some test cases!) 	
    * check that valid range is specified correctly for netcdf types

    """

    extensions = [".mnc"]

    def __init__(self, filename, mode="r", datasource=DataSource(), dtype=N.float, norm_range=None, **keywords):
        """
        Constructs a MINC format from a filename for reading.
        
        dtype = numpy data type to typecast memmap
        norm_range = [lower,upper] range used for truncating image data before
                     normalization
        """

        self.norm_range = norm_range
        filename = datasource.filename(filename)

        #Check if data is zipped
        if iszip(filename):
            fullfilename = unzip(filename)
        else:
            fullfilename = filename
            
	if mode == 'w':
	     raise NotImplementedError, "write support for MINC not available"

        self.filebase = os.path.splitext(fullfilename)[0]
        self.data_file = self.filebase+".mnc"
        self.header_file = self.data_file
	self._netcdf = netcdf(self.header_file, mode)

        ## grid for data

        im = self.get_variable('image')
        dims = [self.get_variable(dname) for dname in im.dimensions]
	step = [dim.step for dim in dims]
        start = [dim.start for dim in dims]
        shape = [self.get_dimension(dname) for dname in im.dimensions]

	self.grid = SamplingGrid.from_start_step(names=im.dimensions, 
	                                         shape=shape,
                                                 start=start,
                                                 step=step)
        # cache for data

        tmpimgfile = os.path.split(self.data_file)[1][:-4] + '.cache'
        cachepath = CacheData(self)
        cachefile = open(cachepath.filename(tmpimgfile),'w')

        # write frames to tmpimgfile
        
	d = N.asarray(self._netcdf.variables['image'][:].copy())
        d = self.normalize(d).astype(dtype)
	d.tofile(cachefile)
        cachefile.close()
        self.data = N.memmap(cachefile.name,
                             shape=tuple(self.grid.shape), mode='r',
                             dtype=dtype)

    def normalize(self, data):
        """
        MINC normalization:

        Otherwise, it uses "image-min" and "image-max" variables
        to map the data from the valid range of the NC_TYPE to the
        range specified by "image-min" and "image-max".

        If self.norm_range is not None, it is used in place of the
        builtin default valid ranges of the NC_TYPEs. If the NC_TYPE
        is NC_FLOAT or NC_DOUBLE, then the transformation is only done if 
        self.norm_range is not None, otherwise the data is untransformed.

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace", "yspace", "xspace"]
        and "image-max" has dimensions ["zspace"]. In this case, the
        normalization is defined by the following transformation:

        for i in range(d.shape[0]):
            d[i] = (clip((d - norm_range[i]).astype(float) / 
                         (norm_range[i] - norm_range[i]), 0, 1) * 
                         (image_max[i] - image_min[i]) + image_min[i])

        """

       	im = self.get_variable("image")

        if self.norm_range is None:
            if im.typecode() not in ['f', 'd']:

                struct_type = {('b','unsigned'):'B',
                               ('b','signed__'):'b',		
                               ('c','unsigned'):'c',
                               ('i','unsigned'):'I',
                               ('i','signed__'):'i',		
                               ('h','unsigned'):'h',
                               ('h','signed__'):'H',		
                               }[(im.typecode(), im.signtype)]

                vrange = {'B':[0,255],
                          'b':[-128,127],
                          'c':[0,255],
                          'i':[-2147483648,2147483647],
                          'I':[0,4294967295],
                          'h':[-32768,32767],
                          'H':[0,65535]}[struct_type]
            else:
                vrange = None
        else:
            vrange = self.norm_range

        #By default, do not normalize for float or double data

        if vrange is None:
            return data

        image_max = self.get_variable("image-max") 
        image_min = self.get_variable("image-min") 
        if image_max.dimensions != image_min.dimensions:
            raise ValueError, '"image-max" and "image-min" do not have the same dimensions'
        axes = [list(im.dimensions).index(d) for d in image_max.dimensions]   
        shape = [self.get_dimension(d) for d in image_max.dimensions]
        indices = N.indices(shape)
        indices.shape = (indices.shape[0], N.product(indices.shape[1:]))
        
        def __normalize(d, i, I, v, V):
            return N.clip((d - v).astype(N.float) / (V - v), 0, 1) * (I - i) + i

        for index in indices.T:
            slice_ = []
            aslice_ = []
            iaxis = 0
	    for idim, dim in enumerate(im.dimensions):
                if idim not in axes:
                    slice_.append(slice(0,self.get_dimension(dim),1))
                else:
                    slice_.append(slice(index[iaxis], index[iaxis]+1,1))
                    aslice_.append(slice(index[iaxis], index[iaxis]+1,1))
                    iaxis += 1
            data[slice_] = __normalize(data[slice_], image_min[aslice_], image_max[aslice_], vrange[0], vrange[1])
	return data

    def __getitem__(self, slice_):
        """
        Data access: return corresponding part of underlying memmap

        """
        return self.data[slice_]

    def get_dimension(self, name):
        """
        Get dimension of MINC file
        """
        return self._netcdf.dimensions[name]

    def get_variable(self, name):
        """
        Get variables of MINC file
        """
        return self._netcdf.variables[name]

    def get_attribute(self, name):
        """
        Get variables of MINC file
        """
        return self._netcdf.attributes[name]


    def get_header_field(self, name):
        """
        Get a field's value from the MINC file.

        It first checks in variables, then attributes
        and finally the dimensions of the MINC file.

        """
	if name in self._netcdf.variables.keys():
	    self.get_variable(name)
        elif name in self.netcdf.attributes.keys():
            return self.get_attribute(name)
        elif name in self.netcdf.dimensions.keys():
            return self.get_dimension(name)
	else:
            raise KeyError, '"%s" not found in variables, attributes or dimensions of MINC file'

    def prewrite(self, x):
        """
        Might transform the data before writing;
        at least confirm dtype.

	Currently, write support is unavaliable.

        """
	raise NotImplementedError 

    def postread(self, x):
        """
        Might transform the data after getting it from memmap
        """
        return x


