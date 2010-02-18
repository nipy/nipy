# -*- Mode: Python -*-  

"""
Fast registration routines module: joint histogram computation,
similarity measures, cubic spline resampling, affine transformation
parameterization.

Author: Alexis Roche, 2008.
"""

__version__ = '0.2'


# Includes
include "numpy.pxi"

# Externals
cdef extern from "math.h":
 
   double log(double)


cdef extern from "iconic.h":

    void iconic_import_array()
    void histogram(double* H, unsigned int clamp, flatiter iter)
    void local_histogram(double* H, unsigned int clamp, 
                         flatiter iter, unsigned int* size)
    void drange(double* h, unsigned int size, double* res)
    void L2_moments(double* h, unsigned int size, double* res)
    void L1_moments(double * h, unsigned int size, double *res)
    double entropy(double* h, unsigned int size, double* n)
    void joint_histogram(double* H, unsigned int clampI, unsigned int clampJ,  
                         flatiter iterI, ndarray imJ_padded, 
                         double* Tvox, int affine, int interp)
    double correlation_coefficient(double* H, unsigned int clampI, unsigned int clampJ, double* n)
    double correlation_ratio(double* H, unsigned int clampI, unsigned int clampJ, double* n) 
    double correlation_ratio_L1(double* H, double* hI, unsigned int clampI, unsigned int clampJ, double* n) 
    double joint_entropy(double* H, unsigned int clampI, unsigned int clampJ)
    double conditional_entropy(double* H, double* hJ, unsigned int clampI, unsigned int clampJ) 
    double mutual_information(double* H, 
                              double* hI, unsigned int clampI, 
                              double* hJ, unsigned int clampJ,
                              double* n)
    double normalized_mutual_information(double* H, 
                                         double* hI, unsigned int clampI, 
                                         double* hJ, unsigned int clampJ, 
                                         double* n) 
    double supervised_mutual_information(double* H, double* F, 
                                         double* fI, unsigned int clampI, 
                                         double* fJ, unsigned int clampJ,
                                         double* n) 
    void cubic_spline_resample(ndarray im_resampled, ndarray im, double* Tvox, int cast_integer)



# Initialize numpy
iconic_import_array()
import_array()
import numpy as np
#cimport numpy as np

# Enumerate texture measures
cdef enum texture_measure: 
    MIN, 
    MAX, 
    DRANGE, 
    MEAN, 
    VARIANCE, 
    MEDIAN, 
    L1DEV, 
    ENTROPY, 
    CUSTOM_TEXTURE

# Corresponding Python dictionary 
builtin_textures = {
    'min': MIN, 
    'max': MAX,
    'drange': DRANGE,
    'mean': MEAN, 
    'variance': VARIANCE, 
    'median': MEDIAN, 
    'l1dev': L1DEV, 
    'entropy': ENTROPY, 
    'custom': CUSTOM_TEXTURE}

# Enumerate similarity measures
cdef enum similarity_measure:
    CORRELATION_COEFFICIENT,
    CORRELATION_RATIO,
    CORRELATION_RATIO_L1,
    JOINT_ENTROPY,
    CONDITIONAL_ENTROPY,
    MUTUAL_INFORMATION,
    NORMALIZED_MUTUAL_INFORMATION,
    SUPERVISED_MUTUAL_INFORMATION, 
    LLR_CORRELATION_COEFFICIENT,
    LLR_CORRELATION_RATIO,
    LLR_CORRELATION_RATIO_L1,
    LLR_MUTUAL_INFORMATION,
    LLR_SUPERVISED_MUTUAL_INFORMATION, 
    CUSTOM_SIMILARITY

# Corresponding Python dictionary 
builtin_similarities = {
    'cc': CORRELATION_COEFFICIENT,
    'cr': CORRELATION_RATIO,
    'crl1': CORRELATION_RATIO_L1, 
    'mi': MUTUAL_INFORMATION, 
    'je': JOINT_ENTROPY,
    'ce': CONDITIONAL_ENTROPY,
    'nmi': NORMALIZED_MUTUAL_INFORMATION,
    'smi': SUPERVISED_MUTUAL_INFORMATION,
    'llr_cc': LLR_CORRELATION_COEFFICIENT,
    'llr_cr': LLR_CORRELATION_RATIO,
    'llr_crl1': LLR_CORRELATION_RATIO_L1,
    'llr_mi': LLR_MUTUAL_INFORMATION,
    'llr_smi': LLR_SUPERVISED_MUTUAL_INFORMATION,  
    'custom': CUSTOM_SIMILARITY}


def _texture(ndarray im, ndarray H, Size, int texture, method=None): 

    cdef double *res, *h
    cdef double moments[5]
    cdef unsigned int clamp
    cdef unsigned int coords[3], size[3]
    cdef broadcast multi
    cdef flatiter im_iter

    # Views
    clamp = <unsigned int>H.dimensions[0]
    h = <double*>H.data
    
    # Copy size parameters
    size[0] = <unsigned int>Size[0]
    size[1] = <unsigned int>Size[1]
    size[2] = <unsigned int>Size[2]

    # Allocate output 
    imtext = np.zeros(im.shape, dtype='double')

    # Loop over input and output images
    multi = PyArray_MultiIterNew(2, <void*>imtext, <void*>im)
    while(multi.index < multi.size):
        res = <double*>PyArray_MultiIter_DATA(multi, 0)
        im_iter = <flatiter>multi.iters[1]
        # Compute local image histogram
        local_histogram(h, clamp, im_iter, size)
        # Switch 
        if texture == MIN:
            drange(h, clamp, moments)
            res[0] = moments[0]
        elif texture == MAX:
            drange(h, clamp, moments)
            res[0] = moments[1]
        elif texture == DRANGE:
            drange(h, clamp, moments)
            res[0] = moments[1]-moments[0]
        elif texture == MEAN: 
            L2_moments(h, clamp, moments)
            res[0] = moments[1]
        elif texture == MEAN: 
            L2_moments(h, clamp, moments)
            res[0] = moments[2]
        elif texture == MEDIAN:
            L1_moments(h, clamp, moments)
            res[0] = moments[1] 
        elif texture == L1DEV: 
            L1_moments(h, clamp, moments)
            res[0] = moments[2] 
        elif texture == ENTROPY: 
            res[0] = entropy(h, clamp, moments)
        else: # CUSTOM
            res[0] = method(H)
        # Next voxel please
        PyArray_MultiIter_NEXT(multi)
   
    return imtext


def _histogram(ndarray H, flatiter iter):
    """
    _joint_histogram(H, iterI)
    Comments to follow.
    """
    cdef double *h
    cdef unsigned int clamp

    # Views
    clamp = <unsigned int>H.dimensions[0]
    h = <double*>H.data

    # Compute image histogram 
    histogram(h, clamp, iter)

    return 


def _joint_histogram(ndarray H, flatiter iterI, ndarray imJ, ndarray Tvox, int affine, int interp):
    """
    _joint_histogram(H, iterI, imJ, Tvox, interp)
    Comments to follow.
    """
    cdef double *h, *tvox
    cdef unsigned int clampI, clampJ

    # Views
    clampI = <unsigned int>H.dimensions[0]
    clampJ = <unsigned int>H.dimensions[1]    
    h = <double*>H.data
    tvox = <double*>Tvox.data

    # Compute joint histogram 
    joint_histogram(h, clampI, clampJ, iterI, imJ, tvox, affine, interp)

    return 


cdef cc2llr(double x, double n):
    cdef double y = 1-x
    if y < 0.0:
        y = 0.0 
    return -.5 * n * log(y)


def _similarity(ndarray H, ndarray HI, ndarray HJ, int simitype, 
                ndarray F=None, method=None):
    """
    _similarity(H, hI, hJ, simitype, ndarray F=None)
    Comments to follow
    """
    cdef int isF = 0
    cdef double *h, *hI, *hJ, *f=NULL
    cdef double simi=0.0, n
    cdef unsigned int clampI, clampJ

    # Array views
    clampI = <unsigned int>H.dimensions[0]
    clampJ = <unsigned int>H.dimensions[1]
    h = <double*>H.data
    hI = <double*>HI.data
    hJ = <double*>HJ.data
    if F != None:
        f = <double*>F.data
        isF = 1

    # Switch 
    if simitype == CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ, &n)
    elif simitype == CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ, &n) 
    elif simitype == CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ, &n) 
    elif simitype == MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == JOINT_ENTROPY:
        simi = joint_entropy(h, clampI, clampJ) 
    elif simitype == CONDITIONAL_ENTROPY:
        simi = conditional_entropy(h, hJ, clampI, clampJ) 
    elif simitype == NORMALIZED_MUTUAL_INFORMATION:
        simi = normalized_mutual_information(h, hI, clampI, hJ, clampJ, &n) 
    elif simitype == SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ, &n)
    elif simitype == LLR_CORRELATION_COEFFICIENT:
        simi = correlation_coefficient(h, clampI, clampJ, &n)
        simi = cc2llr(simi, n)
    elif simitype == LLR_CORRELATION_RATIO: 
        simi = correlation_ratio(h, clampI, clampJ, &n) 
        simi = cc2llr(simi, n)
    elif simitype == LLR_CORRELATION_RATIO_L1:
        simi = correlation_ratio_L1(h, hI, clampI, clampJ, &n) 
        simi = cc2llr(simi, n)
    elif simitype == LLR_MUTUAL_INFORMATION: 
        simi = mutual_information(h, hI, clampI, hJ, clampJ, &n) 
        simi = n*simi
    elif simitype == LLR_SUPERVISED_MUTUAL_INFORMATION:
        simi = supervised_mutual_information(h, f, hI, clampI, hJ, clampJ, &n)
        simi = n*simi
    else: # CUSTOM 
        simi = method(H)
        
    return simi




# Enumerate transformation types
cdef enum transformation_type:
    RIGID2D=0, SIMILARITY2D=1, AFFINE2D=2,
    RIGID3D=3, SIMILARITY3D=4, AFFINE3D=5

# Corresponding Python constants 
affines = ['rigid', 'similarity', 'affine']

_rigid2d = [0,1,5]
_similarity2d = [0,1,5,6,7]
_affine2d = [0,1,5,6,7,11]
_rigid3d = range(6)
_similarity3d = range(9)
_affine3d = range(12)

_affines = [_rigid2d, _similarity2d[0:4], _affine2d, 
            _rigid3d, _similarity3d[0:7], _affine3d]



def rotation_vec2mat(r):
    """
    R = rotation_vec2mat(r)

    The rotation matrix is given by the Rodrigues formula:
    
    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2  
    
    with:
    
           0  -nz  ny
    Sn =   nz   0 -nx
          -ny  nx   0
    
    where n = r / ||r||
    
    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:
    
    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2
    
    leading to:
    
    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2
    """
    cdef double theta, theta2
    theta = <double> np.linalg.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn,Sn)
    else:
        Sr = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        theta2 = theta*theta
        R = np.eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*np.dot(Sr,Sr)
    return R




def param_to_vector12(ndarray param, ndarray t0, ndarray precond, int stamp=AFFINE3D):
    """
    t = param_to_vector12(p, t, precond, stamp=AFFINE3D).

    In-place modification of t. 

    p is a 1d-array of affine transformation parameters with size
    dependent on the transformation type, which is coded by the
    integer stamp.
    """
    t = t0

    # Switch on transformation type
    if stamp == RIGID3D:
        t[_rigid3d] = param*precond[_rigid3d]
    elif stamp == SIMILARITY3D:
        t[_similarity3d] = param[[0,1,2,3,4,5,6,6,6]]*precond[_similarity3d]
    elif stamp == AFFINE3D:
        t = param*precond
    elif stamp == RIGID2D:
        t[_rigid2d] = param*precond[_rigid2d]
    elif stamp == SIMILARITY2D:
        t[_similarity2d] = param[[0,1,2,3,3]]*precond[_similarity2d]
    elif stamp == AFFINE2D:
        t[_affine2d] = param*precond[_affine2d]

    return t
    

def matrix44(ndarray t, dtype):
    """
    T = matrix44(t)

    t is a vector of of affine transformation parameters with size at
    least 6.

    size < 6 ==> error
    size == 6 ==> t is interpreted as translation + rotation
    size == 7 ==> t is interpreted as translation + rotation + isotropic scaling
    7 < size < 12 ==> error
    size >= 12 ==> t is interpreted as translation + rotation + scaling + shearing 
    """
    cdef unsigned int size
    size = <unsigned int>PyArray_SIZE(t)
    T = np.eye(4, dtype=dtype)
    R = rotation_vec2mat(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(np.exp(t[6:9])) 
        Q = rotation_vec2mat(t[9:12]) 
        # Beware: R*s*Q
        T[0:3,0:3] = np.dot(R,np.dot(S,Q))
    T[0:3,3] = t[0:3] 
    return T 




def slice_time(Z, double tr_slices, slice_order):
    """
    Fast routine to compute the time when a slice is acquired given its index
    """
    cdef double *z, *t, *s
    cdef double weight, slice_interp
    cdef unsigned int zfloor, cycles, zl, zr
    cdef broadcast multi
    cdef unsigned int nslices

    Za = np.asarray(Z, dtype='double')
    T = np.zeros(Za.shape, dtype='double')
    S = np.ascontiguousarray(slice_order, dtype='double')
    s = <double*>PyArray_DATA(<ndarray>S)
    nslices = S.size
 
    multi = PyArray_MultiIterNew(2, <void*>Za, <void*>T)
    while(multi.index < multi.size):
        z = <double*>PyArray_MultiIter_DATA(multi, 0)
        t = <double*>PyArray_MultiIter_DATA(multi, 1)
        zfloor = <unsigned int>z[0]
        cycles = zfloor / nslices 
        zl = zfloor % nslices
        zr = zl + 1
        weight = z[0]-zfloor
        if zl < (nslices-1):
            slice_interp = (1-weight)*s[zl] + weight*s[zr]
        else:
            slice_interp = (1-weight)*s[zl] + weight*nslices
        t[0] = cycles + tr_slices*slice_interp
        PyArray_MultiIter_NEXT(multi)
    return T


