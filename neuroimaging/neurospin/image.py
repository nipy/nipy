"""
Input/output functions. This 'high-level' module is not meant to stay
permanently in fff, but is supplied to fff users so that they can
process data for real until fff is properly integrated into nipy.
"""


import numpy as np


DEFAULT_IOLIB = 'pynifti'

class Image:

    def __init__(self, obj=None, affine=None, voxsize=None, iolib=DEFAULT_IOLIB):
        """
        Basic image class to be substituted with nipy's class when
        ready.
        
        affine: 4x4 transformation matrix from array to scanner
        coordinate system.
        
        IMPORTANT NOTE: the image class assumes that the scanner
        coordinate system is ALWAYS right-handed and such that x
        increases from left to right, y increases from posterior to
        anterior, and z increases in the inferior to superior
        direction. This transformation is thus exactly what the nifti
        norm calls the 'qform'.
        """
       
        self._iolib = iolib
        
        # Case: initialize from array
        if isinstance(obj, np.ndarray):
            self._array = obj
            # Default transform: origin maps to real coordinates zero
            if affine == None:
                if voxsize == None:
                    voxsize = np.ones(3)
                affine = np.diag(np.concatenate((voxsize,[1]),1))
                origin = .5*(np.asarray(obj.shape)-1) 
                affine[0:3,3] = -voxsize*origin
            self._affine = affine
            
            if self._iolib == 'pynifti':
                import nifti
                self._image = nifti.NiftiImage(obj.T)
                self._image.setQForm(affine)
            
        else:
            # Case: initialize from file
            if isinstance(obj, str):
                self._load(obj)
            elif isinstance(obj, unicode):
                self._load(str(obj))
            # Case: initialize from image
            elif isinstance(obj, image):
                self._clone(obj)
            # Default 
            else:
                self._array = None
                self._affine = None
                self._image = None
        
    def _clone(self, model):
        self._array = model._array
        self._affine = model._affine
        self._iolib = model._iolib
        self._image = model._image

    def _load(self, filename):
        """
        Read an image file, and make a python dictionary based on array
        data and header info. 
        """
        iolib = self._iolib 
        if iolib == 'nipy':
            from neuroimaging.core.api import Image
            self._image = Image.load(filename)
            self._array = self._image.buffer
            self._affine = self._image.grid.mapping.transform
        
        elif iolib == 'aims':
            from soma import aims
            reader = aims.Reader()
            self._image = reader.read(filename)
            self._array = self._image.__array__().squeeze()
            header = self._image.header().get()
            voxsize = np.asarray(header['voxel_size'])[0:np.minimum(3, self._array.ndim)]
            if header.has_key('transformations'):
                affine = np.asarray(header['transformations'][0]).reshape(4,4)
                self._affine = np.dot(affine, np.diag(np.concatenate((voxsize,[1]))))      
            
        elif iolib == 'pynifti':
            import nifti
            self._image = nifti.NiftiImage(filename)
            self._array = self._image.data.T
            ##voxsize = self._image.voxdim
            self._affine = self._image.qform

        else:
            print 'Unknown input/output library.'
        
                
    def save(self, filename):
        """
        Save an image as described by a dictionary with keys 'array' and voxsize'.
        """
        filename = str(filename)
        if self._image == None: 
            print('Dummy image: cannot save')
            return

        if self._iolib == 'nipy':
            ##imIt = Image(It, grid=imJ.grid)
            return
        
        elif self._iolib == 'aims':
            from soma import aims
            w = aims.Writer()
            # what if the two arrays are still the same object ?!
            self._image.__array__().squeeze()[:] = self._array.squeeze()[:]
            w.write(self._image, filename)

        elif self._iolib == 'pynifti':
            self._image.data[:] = self._array.T[:]
            self._image.save(filename)
            

    def get_data(self):
        return self._array

    def get_shape(self):
        return self._array.shape

    def get_affine(self):
        return self._affine

    def set_data(self, array):
        self._array = array


"""
AIMS
##image.header()[key] = header[key]

NIPY
##imIt.tofile(outfile+'.nii', clobber=True)
"""

