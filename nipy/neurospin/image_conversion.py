"""
Dummy utility to convert some image objects to the 'brifti'
class. This is mainly a compatibility fix for former fff users, and is
not intended to be robust, nor elegant at all.

Supported formats are 'aims' and 'pynifti' (as of spring 2009). 

Usage: 

im = asbrifti(nifti_im, 'nifti')
im = asbrifti(aims_im, 'aims')
"""

import numpy as np

from nipy.io.imageformats import Nifti1Image as Image 

def asbrifti(obj, iolib):

    # pyaims image
    if 'aims' in iolib:
        data = obj.__array__().squeeze()
        header = obj.header().get()
        voxsize = np.asarray(header['voxel_size'])[0:np.minimum(3, self._array.ndim)]
        affine = np.diag(np.concatenate((voxsize,[1])))
        if header.has_key('transformations'):
            world_transform = np.asarray(header['transformations'][0]).reshape(4,4)
            affine = np.dot(world_transform, affine) 
        
    # pynifti image
    elif 'nifti' in iolib:
        data = obj.data.T
        affine = obj.qform

    else:
        print 'Unknown input/output library.'
        return None

    return Image(data, affine)

