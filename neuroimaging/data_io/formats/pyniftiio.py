"""The Nipy interface to the PyNifti library.

Use PyNifti to open files and extract necessary data to generate a Nipy Image.

"""

import numpy as np

from neuroimaging.externals.pynifti import nifti

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


class PyNiftiIO:
    """Wrapper around the PyNifit image class.
    """

    def __init__(self, data, mode='r'):
        self._nim = nifti.NiftiImage(data)
        self.mode = mode

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
    header = property(_get_header)

    def _getdata(self, index):
        """Apply slicing and return data."""
        # Only apply if scl_slope is nonzero
        if self._nim.slope != 0.0:
            return self._nim.data[index] * self._nim.slope + self._nim.intercept
        else:
            return self._nim.data[index]

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
        transform = np.diag(pixdims[1:5])

    """
    generate transforms to flip data from 
    matlabish (nifti header default fortran ordered)
    to nipyish (c ordered)
    to correctly correspond with c-ordered image data
    """
    trans = np.fliplr(np.eye(4, k=1))
    trans[3,3] = 1
    trans2 = trans.copy()
    trans2[:,3] = 1  # Why do we add a pixdim step to our translation?
    baseaffine = np.dot(np.dot(trans, transform), trans2)
    # deal with 4D+ dimensions
    ndim = img.header['dim'][0]
    if ndim > 3:
        # create identity with steps based on pixdim
        affine = np.eye(img.header['dim'][0])
        step = np.array(img.header['pixdim'][1:(ndim+1)])
        affine = affine * step[::-1]
        affine[-4:,-4:] = baseaffine
    else:
        affine = baseaffine

    return affine
