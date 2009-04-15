"""The Nipy interface to the PyNifti library.

Use PyNifti to open files and extract necessary data to generate a Nipy Image.

"""

import numpy as np

import nifti

NIFTI_UNKNOWN = 0
# Nifti orientation codes from nifti1_io.h
NIFTI_L2R = 1    # Left to Right
NIFTI_R2L = 2    # Right to Left
NIFTI_P2A = 3    # Posterior to Anterior
NIFTI_A2P = 4    # Anterior to Posterior
NIFTI_I2S = 5    # Inferior to Superior
NIFTI_S2I = 6    # Superior to Inferior

# TODO: Move to using the nifti orientations instead of our names/spaces
orientation_to_names = {NIFTI_L2R : 'xspace',
                        NIFTI_R2L : 'xspace',
                        NIFTI_P2A : 'yspace',
                        NIFTI_A2P : 'yspace',
                        NIFTI_I2S : 'zspace',
                        NIFTI_S2I : 'zspace'
}

"""
In [75]: img.getSOrientation(as_string=True)
Out[75]: ['Right-to-Left', 'Posterior-to-Anterior', 'Inferior-to-Superior']

In [76]: img.getQOrientation(as_string=True)
Out[76]: ['Right-to-Left', 'Posterior-to-Anterior', 'Inferior-to-Superior']

In [77]: funcimg.getSOrientation(as_string=True)
Out[77]: ['Unknown', 'Unknown', 'Unknown']

In [78]: funcimg.getQOrientation(as_string=True)
Out[78]: ['Left-to-Right', 'Posterior-to-Anterior', 'Inferior-to-Superior']
"""

def _dtype_maxval(dtype):
    """Return maximum value for the dtype."""
    if type(dtype) in np.sctypes['float']:
        # float32, float64
        return np.finfo(dtype).max
    elif type(dtype) in np.sctypes['complex']:
        # complex128
        raise NotImplementedError, 'BUG: Not handling complex128 types yet!'
    else:
        # uint8, int8, uint16, int16, uint32, int32, uint64, int64
        return np.iinfo(dtype).max

class PyNiftiIO(object):
    """Wrapper around the PyNifit image class.
    """

    def __init__(self, data, mode='r', dtype=None, header={}):
        """Create a PyNiftiIO object.

        Parameters
        ----------
        data : {array_like, filename}
            Data should be either a filename (string), a numpy array
            or an object that implements the __array__ interface from
            which we get an array.
        mode : {'r', 'w'}, optional
            File access mode.  Read-only or read-write mode.
        dtype : numpy.dtype
            The dtype to save the data array as.  An exception is
            raised if the requested dtype is not a valid nifti data
            type.

        Returns
        -------
        pyniftiio : A ``PyNiftiIO`` object
        
        """
        
        # If data is not an ndarray and not a filename (str)
        if not hasattr(data, 'dtype') and not hasattr(data, 'endswith'):
            # convert data to ndarray
            
            # Note: Let's say `data` is an Image that was loaded from a
            # file and it's dtype is uint8 with a non-zero slope.
            # Getting the data as an array will caused the data to
            # scale into our native dtype... float32/float64.
            # data = np.asarray(data)

            if dtype is None:
                # Get native dtype for this machine
                dtype = np.array([1.0]).dtype
            else:
                # Validate the requested dtype is a nifti type
                try:
                    nifti.utils.N2nifti_dtype_map[dtype]
                except KeyError:
                    msg = "The requested dtype '%s' is not a valid nifti type"\
                          % dtype
                    raise KeyError, msg
            
            # FIXME: Until we implement scaling, don't use the dtype,
            # casting like this will loose information.
            #data = np.asarray(data).astype(dtype)

            # np.asarray will return the array in the native type
            # (float32 or float64)
            data = np.asarray(data)

        # Create NiftiImage
        self._nim = nifti.NiftiImage(data, header=header)
        self.mode = mode

        #print self._nim.data.dtype
        #print 'slope:', self._nim.slope
        #print 'intercept:', self._nim.intercept

    # image attributes
    # ----------------
    def _get_affine(self):
        # build affine transform from header info
        return getaffine(self._nim)
    affine = property(_get_affine)

    def _get_orientation(self):
        # Try the S and Q orientations and get one that works
        ornt = self._nim.getSOrientation()
        if ornt[0] == NIFTI_UNKNOWN:
            ornt = self._nim.getQOrientation()
        return ornt
    orientation = property(_get_orientation)

    def _get_header(self):
        return self._nim.header
    def _set_header(self, header):
        self._nim.updateFromDict(header)
    header = property(_get_header, _set_header,
                      doc='Get or set the pynifti header.')

    def _getdata(self, index):
        """Apply slicing and return data."""
        # Note: We need to return the data as a copy of the array,
        # otherwise it may be freed in the C Extension code while
        # we're holding a reference to it.  This will cause a
        # Segmentation Fault.

        # Only apply if scl_slope is nonzero
        if self._nim.slope != 0.0:
            return self._nim.data[index].copy() * self._nim.slope \
                   + self._nim.intercept
        else:
            return self._nim.data[index].copy()

    # array-like interface
    # --------------------
    def _get_ndim(self):
        return self._nim.data.ndim
    ndim = property(_get_ndim)

    def _get_shape(self):
        return self._nim.data.shape
    shape = property(_get_shape)

    def __getitem__(self, index):
        if type(index) not in [type(()), type([])]:
            index = (index,)
        else:
            index = tuple(index)
        
        msg = 'when slicing images, index must be a list of integers or slices'
        for i in index:
            if type(i) not in [type(1), type(slice(0,4,1))]:
                raise ValueError, msg

        data = self._getdata(index)
        return data
    
    def __setitem__(self, index, data):
        if self.mode is not 'r':
            self._nim.data[index] = data
        else:
            raise IOError, \
                "File %s is open for read only!" % self._nim.filename

    def __array__(self):
        # Generate slice index to match 'data[:]'
        index = (slice(None, None, None),)
        data = self._getdata(index)
        return np.asarray(data)

    # file interface
    # --------------
    def _get_filename(self):
        return self._nim.getFilename()
    def _set_filename(self, filename):
        self._nim.setFilename(filename)
    filename = property(fget=_get_filename, fset=_set_filename)

    def save(self, affine, pixdim, diminfo, filename=None):
        if filename is not None:
            # update filename
            self.filename = filename
        # We are setting both the sform and qform to the same affine
        # transform.  And setting the sform_code and qform_codes to be
        # aligned to another file, NIFTI_XFORM_ALIGNED_ANAT in the
        # Nifti standard.  This writing method matches that of SPM5.
        self.pixdim = pixdim
        self.diminfo = diminfo
        self._nim.setSForm(affine, code='aligned')
        self._nim.setQForm(affine, code='aligned')
        self._nim.save()

    # These two properties of the header are important for saving
    # Images with the correct dimension info

    def _getpixdim(self):
        return self._nim.getPixDims()
    def _setpixdim(self, pixdim):
        self._nim.setPixDims(pixdim)
    pixdim = property(_getpixdim, _setpixdim)

    # FIXME: setting the diminfo is not 
    # working quite right 

    def _getdiminfo(self):
        return self._nim.header['dim_info']
    def _setdiminfo(self, diminfo):
        self._nim.updateFromDict({'dim_info': diminfo})
    diminfo = property(_getdiminfo, _setdiminfo)
        
def getaffine(img):
    """Get affine transform from a NiftiImage.

    Parameters
    ----------
    img : NiftiImage
        image opened with PyNifti

    Returns
    -------
    affine : array
        The 4x4 affine transform as a numpy array

    """

    try:
        sform_code = img.header['sform_code']
        qform_code = img.header['qform_code']
    except KeyError:
        raise IOError, 'Invalid header! Unable to get sform or qform codes.'

    if sform_code > 0:
        # Use sform when there's a valid sform_code.  Transform is mapping
        # to a standard space. Method 3 in Nifti1 spec.
        transform = img.getSForm()
    elif qform_code > 0:
        # Use qform for mapping, generally to scanner space.
        # Method 2 in Nifti1 spec.
        transform = img.getQForm()
    else:
        # getPixDims only returns the last 7 pixdims, does not include qfac
        pdims = img.getPixDims()
        qfac = img.qfac
        pixdims = [qfac]
        # unpack pdims tuple into pixdims list
        [pixdims.append(i) for i in pdims]
        transform = np.identity(4)
        transform[:-1, :-1] = np.diag(pixdims[1:4])
        
    # deal with 4D+ dimensions
    ndim = img.header['dim'][0]
    if ndim > 3:
        # create identity with steps based on pixdim
        affine = np.eye(img.header['dim'][0]+1)
        step = np.array(img.header['pixdim'][1:(ndim+1)])
        affine[:-1,:-1] = np.diag(step)
        affine[:3,:3] = transform[:3,:3]
    else:
        affine = transform

    return affine
