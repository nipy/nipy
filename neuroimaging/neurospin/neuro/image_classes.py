"""
Input/output functions. This 'high-level' module is not meant to stay
permanently in fff, but is supplied to fff users so that they can
process data for real until fff is properly integrated into nipy.
"""


import numpy as np

from neuroimaging.neurospin.utils import slice_time

DEFAULT_IOLIB = 'pynifti'

class image:
    def __init__(self, obj=None, transform=None, voxsize=None, iolib=DEFAULT_IOLIB):

        """
        Basic image class. 
        
        transform: 4x4 transformation matrix from array to scanner coordinate
        system.
        
        IMPORTANT NOTE: the image class assumes that the scanner
        coordinate system is ALWAYS right-handed and such that x
        increases from left to right, y increases from posterior to
        anterior, and z increases in the inferior to superior
        direction. This transformation is thus exactly what the nifti
        norm calls the 'qform'.
        """
        self.iolib = iolib

        # Case: initialize from array
        if isinstance(obj, np.ndarray):
            self.array = obj
            # Default transform: origin maps to real coordinates zero
            if transform == None:
                if voxsize == None:
                    voxsize = np.ones(3)
                transform = np.diag(np.concatenate((voxsize,[1]),1))
                origin = .5*(np.asarray(obj.shape)-1) 
                transform[0:3,3] = -voxsize*origin
            self.transform = transform
            
            if self.iolib == 'pynifti':
                import nifti
                self._image = nifti.NiftiImage(obj.T)
                self._image.setQForm(transform)
            
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
                self.array = None
                self.transform = None
                self._image = None
        
    def _clone(self, model):
        self.array = model.array
        self.transform = model.transform
        self.iolib = model.iolib
        self._image = model._image

    def _load(self, filename):
        """
        Read an image file, and make a python dictionary based on array
        data and header info. 
        """
        iolib = self.iolib 
        if iolib == 'nipy':
            from neuroimaging.core.api import Image
            self._image = Image.load(filename)
            self.array = self._image.buffer
            self.transform = self._image.grid.mapping.transform
        
        elif iolib == 'aims':
            from soma import aims
            reader = aims.Reader()
            self._image = reader.read(filename)
            self.array = self._image.__array__().squeeze()
            header = self._image.header().get()
            voxsize = np.asarray(header['voxel_size'])[0:np.minimum(3, self.array.ndim)]
            if header.has_key('transformations'):
                transform = np.asarray(header['transformations'][0]).reshape(4,4)
                self.transform = np.dot(transform, np.diag(np.concatenate((voxsize,[1]))))      
            
        elif iolib == 'pynifti':
            import nifti
            self._image = nifti.NiftiImage(filename)
            self.array = self._image.data.T
            ##voxsize = self._image.voxdim
            self.transform = self._image.qform

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

        if self.iolib == 'nipy':
            ##imIt = Image(It, grid=imJ.grid)
            return
        
        elif self.iolib == 'aims':
            from soma import aims
            w = aims.Writer()
            # what if the two arrays are still the same object ?!
            self._image.__array__().squeeze()[:] = self.array.squeeze()[:]
            w.write(self._image, filename)

        elif self.iolib == 'pynifti':
            self._image.data[:] = self.array.T[:]
            self._image.save(filename)
            

    def set_array(self, array):
        self.array = array

"""
AIMS
##image.header()[key] = header[key]

NIPY
##imIt.tofile(outfile+'.nii', clobber=True)
"""

class fmri_image(image):

    def __init__(self, img, tr=1.0, tr_slices=None, start=0.0, \
                 slice_axis=2, slice_order='ascending', interleaved=False):
        self.array = img.array
        self.transform = img.transform
        self.iolib = img.iolib
        self._image = img._image
        self._set_timing(tr, tr_slices, start, slice_axis, slice_order, interleaved)

    def _set_timing(self, tr, tr_slices, start, slice_axis, slice_order, interleaved):
        """Configure fMRI acquisition time parameters.
        
        tr  : inter-scan repetition time, i.e. the time elapsed between two consecutive scans
        tr_slices : inter-slice repetition time, same as tr for slices
        start   : starting acquisition time respective to the implicit time origin
        slice_order : string or array 
        """
        # Number of slices
        nslices = self.array.shape[slice_axis]

        # Default slice repetition time (no silence)
        if tr_slices == None:
            tr_slices = tr/float(nslices)

        # Set slice order
        if isinstance(slice_order, str): 
            if not interleaved:
                aux = range(nslices)
            else:
                p = nslices/2
                aux = []
                for i in range(p):
                    aux.extend([i,p+i])
                if nslices%2:
                    aux.append(nslices-1)
            if slice_order == 'descending':
                aux.reverse()
            slice_order = aux
            
        # Set timing values
        self.nslices = nslices
        self.tr = float(tr)
        self.tr_slices = float(tr_slices)
        self.start = float(start)
        self.slice_order = np.asarray(slice_order)

        # Check whether 3th array index z increases from the bottom to
        # the top of the head, or the other way round
        # FIXME: what if transform involves a non-transversal rotation? 
        if self.transform[2,2] > 0:
            self.reversed_slices = False
        else:
            self.reversed_slices = True


    def z_to_slice(self, z):
        """
        Account for the fact that slices may be stored in reverse
        order wrt the scanner coordinate system convention (slice 0 ==
        bottom of the head)
        """
        if self.reversed_slices:
            return self.nslices - 1 - z
        else:
            return z

    def time_transform(self, z, t):
        return(self.start + self.tr*t + slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))


    def inverse_time_transform(self, z, tt):
        return((tt - self.start - slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))/self.tr)
