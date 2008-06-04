"""The Nipy interface to the PyNifti library.

Use PyNifti to open files and extract necessary data to generate a Nipy Image.

"""

import numpy as np

import nifti

#from neuroimaging.core.reference.mapping import Affine
#from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.data_io.formats.nifti1_ext import quatern2mat
from neuroimaging.data_io.formats.nifti1 import scale_data

NIFTI_UNKNOWN = 0
# Nifti orientation codes from nifti1_io.h
NIFTI_L2R = 1    # Left to Right
NIFTI_R2L = 2    # Right to Left
NIFTI_P2A = 3    # Posterior to Anterior
NIFTI_A2P = 4    # Anterior to Posterior
NIFTI_I2S = 5    # Inferior to Superior
NIFTI_S2I = 6    # Superior to Inferior

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
    def __init__(self, data):
        self.img = nifti.NiftiImage(data)

    def _get_affine(self):
        return getaffine(self.img)
    affine = property(_get_affine)

    def _get_data(self):
        return self.img.asarray(copy=True)
    data = property(_get_data, doc='Return data as array copy')

    def _get_scale_func(self):
        return scale_data
    scale_func = property(_get_scale_func)

    def _get_scale_factor(self):
        return self.img.slope
    scale_factor = property(_get_scale_factor)
    
    def _get_scale_inter(self):
        return self.img.intercept
    scale_inter = property(_get_scale_inter)

    def _get_orientation(self):
        # Try the S and Q orientations and get one that works
        ornt = self.img.getSOrientation()
        if ornt[0] == NIFTI_UNKNOWN:
            ornt = self.img.getQOrientation()
        return ornt
    orientation = property(_get_orientation)

    def _get_header(self):
        # Not sure this is needed in the long run... here for current 
        # compatibility of nipy
        return self.img.header
    header = property(_get_header)

def getaffine(img):
    """Get affine transform from a NiftiImage.

    Parameters
    ----------
    img : NiftiImage
        image opened with PyNifti

    Returns
    -------
    affine : array
        The 4x4 affine transform in a numpy array

    """

    sformcode = int(img.header['sform_code'])
    
    if sformcode > 0:
        # Method(3) to map into a standard space use srow_x, srow_y, srow_z

        """
        #Method 3
        METHOD 3 (used when sform_code > 0) 
        The (x,y,z) coordinates are given by a general affine transformation
        x = srow_x[0] * i + srow_x[1] * j + srow_x[2] * k + srow_x[3]
        y = srow_y[0] * i + srow_y[1] * j + srow_y[2] * k + srow_y[3] 
        z = srow_z[0] * i + srow_z[1] * j + srow_z[2] * k + srow_z[3]   
        """
        transform = img.getQForm()
    elif sformcode < 0:
        quaternion = img.getQuaternion()
        qoffset = img.getQOffset()
        pixdims = img.getPixDims()
        transform = _getaffine_method2(quaternion, qoffset,pixdims)
    else:
        qoffset = img.getQOffset()
        pixdims = img.getPixDims()
        transform = _getaffine_method1(qoffset,pixdims)
        
    """
    generate transforms to flip data from 
    matlabish (nifti header default fortran ordered)
    to nipyish (c ordered)
    to correctly correspond with c-ordered image data
    """
    trans = np.zeros((4,4))
    trans[0:3,0:3] = np.fliplr(np.eye(3))
    trans[3,3] = 1
    trans2 = trans.copy()
    trans2[:,3] = 1
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

    """
    spaces = ['vector','time','zspace','yspace','xspace']
    space = tuple(spaces[-ndim:])
    shape = tuple(img.header['dim'][1:ndim+1])
    grid = SamplingGrid.from_affine(Affine(affine),space,shape)
    return grid        
    """
    return affine
    
def _getaffine_method1(qoffset, pixdims):
    """
    Method to get image orientation location based on Method1 in nifti.h
        
    METHOD 1 (the "old" way, used only when qform_code = 0)
    The coordinate mapping from (i,j,k) to (x,y,z) is the ANALYZE
    7.5 way.  This is a simple scaling relationship:
    
    x = pixdim[1] * i
    y = pixdim[2] * j
    z = pixdim[3] * k
    
    No particular spatial orientation is attached to these (x,y,z)
    coordinates.  (NIFTI-1 does not have the ANALYZE 7.5 orient field,
    which is not general and is often not set properly.)  This method
    is not recommended, and is present mainly for compatibility with
    ANALYZE 7.5 files.
    
    Returns
    -------
    transmatrix : numpy.array
    4x4 affine transformation matrix
    
    """
    
    origin = qoffset
    step = np.ones(4)
    step[0:4] = pixdims[1:5]
    transmatrix = np.eye(4) * step
    transmatrix[:3,3] = origin
    return transmatrix

def _getaffine_method2(quaternion, qoffset, pixdims):
    """Method to get image orientation location based on Method2 in nifti.h
    
    METHOD 2 (used when qform_code > 0, which should be the "normal" case)
    The (x,y,z) coordinates are given by the pixdim[] scales, a rotation
    matrix, and a shift.  This method is intended to represent
    "scanner-anatomical" coordinates, which are often embedded in the
    image header (e.g., DICOM fields (0020,0032), (0020,0037), (0028,0030),
    and (0018,0050)), and represent the nominal orientation and location of
    the data.  This method can also be used to represent "aligned"
    coordinates, which would typically result from some post-acquisition
    alignment of the volume to a standard orientation (e.g., the same
    subject on another day, or a rigid rotation to true anatomical
    orientation from the tilted position of the subject in the scanner).
    The formula for (x,y,z) in terms of header parameters and (i,j,k) is:
    
    [ x ]   [ R11 R12 R13 ] [        pixdim[1] * i ]   [ qoffset_x ]
    [ y ] = [ R21 R22 R23 ] [        pixdim[2] * j ] + [ qoffset_y ]
    [ z ]   [ R31 R32 R33 ] [ qfac * pixdim[3] * k ]   [ qoffset_z ]
    
    The qoffset_* shifts are in the NIFTI-1 header.  Note that the center
    of the (i,j,k)=(0,0,0) voxel (first value in the dataset array) is
    just (x,y,z)=(qoffset_x,qoffset_y,qoffset_z).
    
    The rotation matrix R is calculated from the quatern_* parameters.
    This calculation is described below.
    
    The scaling factor qfac is either 1 or -1.  The rotation matrix R
    defined by the quaternion parameters is "proper" (has determinant 1).
    This may not fit the needs of the data; for example, if the image
    grid is
    i increases from Left-to-Right
    j increases from Anterior-to-Posterior
    k increases from Inferior-to-Superior
    Then (i,j,k) is a left-handed triple.  In this example, if qfac=1,
    the R matrix would have to be
    
    [  1   0   0 ]
    [  0  -1   0 ]  which is "improper" (determinant = -1).
    [  0   0   1 ]
    
    If we set qfac=-1, then the R matrix would be
    
    [  1   0   0 ]
    [  0  -1   0 ]  which is proper.
    [  0   0  -1 ]
    
    This R matrix is represented by quaternion [a,b,c,d] = [0,1,0,0]
    (which encodes a 180 degree rotation about the x-axis).
    
    
    Returns
    -------
    transmatrix : numpy.array
    4x4 affine transformation matrix
    
    """

    # check qfac
    qfac = float(pixdims[0])
    if qfac not in [-1.0, 1.0]:
        if qfac == 0.0:
            # According to Nifti Spec, if pixdim[0]=0.0, take qfac=1
            print 'qfac of nifti header is invalid: setting to 1.0'
            print 'check your original file to validate orientation'
            qfac = 1.0;
        else:
            raise Nifti1FormatError('invalid qfac: orientation unknown')

    transmatrix = quatern2mat(b=quaternion[0],
                              c=quaternion[1],
                              d=quaternion[2],
                              qx=qoffset[0],
                              qy=qoffset[1],
                              qz=qoffset[2],
                              dx=pixdims[1],
                              dy=pixdims[2],
                              dz=pixdims[3],
                              qfac=qfac)
    return transmatrix
