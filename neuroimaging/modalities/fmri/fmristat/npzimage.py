"""
A simple way of storing `neuroimaging.core.image.image.Image`'s in .npz files.

Also includes NPZBuffer that keeps data in an ndarray instance until flushed to an .npz file.

At the time, NifTi file writing is broken -- that's why NifTi files are not written.
"""

import numpy as np
import os

from neuroimaging.core.api import Image, Affine, CoordinateMap

def _fixext(filename):
    a, b = os.path.splitext(filename)
    return os.path.join(a + '.npz')

def save_npz(filename, image,
             clobber=False, **extra):
    """
    Save an .npz Image, which consists of at least three arrays:

    * data: the data array
    * dimnames: the dimension names of the corresponding grid
    * affine: the affine transformation of grid

    Image must have an affine transformation, which is the only
    part of the mapping that is saved.
    """
    d = os.path.dirname(os.path.abspath(filename))
    if d and not os.path.exists(d):
        os.makedirs(d)
    if clobber or not os.path.exists(filename):
        np.savez(filename, affine=image.affine, data=np.asarray(np.asarray(image)),
                 dimnames=image.coordmap.output_coords.axisnames(), **extra)
    else:
        raise IOError, 'file exists and clobber=False'
    
def create_npz(filename, grid, dtype=np.float32,
               clobber=False):
    """
    Create an .npz Image, which consists of at least three arrays:

    * data: the data array
    * dimnames: the dimension names of the corresponding grid
    * affine: the affine transformation of grid
    """
    
    tmp = Image(np.zeros(grid.shape, dtype), grid)
    save_npz(filename, tmp, clobber=clobber)
    del(tmp)
    return load_npz(filename)

def load_npz(filename):
    """
    Load an .npz Image, this .npz file must have at least two arrays

    * data: the data array
    * dimnames: the dimension names of the corresponding grid
    * affine: the affine transformation of grid

    The remaining arrays of .npz file are stored as the 'extra' attribute of the Image.
    """
    npzobj = np.load(filename)
    
    im = Image(npzobj['data'], CoordinateMap.from_affine(Affine(npzobj['affine']),
                                                        list(npzobj['dimnames']),
                                                        npzobj['data'].shape))
    im.extra = {}
    for f in npzobj.files:
        if f not in ['affine', 'dimnames', 'data']:
            im.extra[f] = npzobj[f]

    return im

class NPZBuffer:

    """
    A temporary image that is saved to an npz Image when its
    flush method is called.

    The attribute 'extra' should be a dictionary of arrays that will be included in the
    resuting .npz file when flush is called. This is a simple way to add extra information
    to Image files.
    
    """
    
    def __init__(self, filename, grid, clobber=False):
        self.filename = _fixext(filename)
        self.clobber = clobber
        self._im = Image(np.zeros(grid.shape), grid)
        self._flushed = False
        self.extra = {}

    def __getitem__(self, item):
        if not self._flushed:
            return self._im[item]
        else:
            raise ValueError, 'trying to read value from flushed NPZBuffer'

    def __setitem__(self, item, value):
        if not self._flushed:
            self._im[item] = value
        else:
            raise ValueError, 'trying to set value on flushed NPZBuffer'
        
    def flush(self):
        save_npz(self.filename, self._im, clobber=self.clobber, **self.extra)
        del(self._im)
        self._flushed = True
