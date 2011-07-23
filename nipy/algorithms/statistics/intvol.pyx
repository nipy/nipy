"""
The estimators for the intrinsic volumes appearing in this module
were partially supported by NSF grant DMS-0405970.

Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
   with an application to brain mapping."
   Journal of the American Statistical Association, 102(479):913-928.

"""
cimport cython

import numpy as np
cimport numpy as np

from scipy.sparse import dok_matrix

# local imports
from utils import cube_with_strides_center, join_complexes


DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

cdef double PI = np.pi


cdef extern from "math.h" nogil:
    double floor(double x)
    double sqrt(double x)
    double fabs(double x)
    double log2(double x)
    double acos(double x)    
    bint isnan(double x)

    
cpdef double mu3_tet(double D00, double D01, double D02, double D03,
                     double D11, double D12, double D13, 
                     double D22, double D23, 
                     double D33) nogil:
  """
  Compute the 3rd intrinsic volume (just volume in this case) of 
  a tetrahedron with coordinates [coords[v0], coords[v1], coords[v2], coords[v3]].

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Returns
  -------
  mu3 : float
  """
  cdef double C00, C01, C02, C11, C12, C22
  C00 = D00 - 2*D03 + D33
  C01 = D01 - D13 - D03 + D33
  C02 = D02 - D23 - D03 + D33
  C11 = D11 - 2*D13 + D33
  C12 = D12 - D13 - D23 + D33
  C22 = D22 - 2*D23 + D33
  return sqrt((C00 * (C11 * C22 - C12 * C12) -
               C01 * (C01 * C22 - C02 * C12) +
               C02 * (C01 * C12 - C11 * C02))) / 6.


cpdef double mu2_tet(double D00, double D01, double D02, double D03,
                     double D11, double D12, double D13, 
                     double D22, double D23,
                     double D33) nogil:
  """
  Compute the 2nd intrinsic volume (half the surface area) of 
  a tetrahedron with coordinates [coords[v0], coords[v1], coords[v2], coords[v3]].

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Returns
  -------
  mu2 : float
  """
  cdef double mu = 0
  mu += mu2_tri(D00, D01, D02, D11, D12, D22)
  mu += mu2_tri(D00, D02, D03, D22, D23, D33)
  mu += mu2_tri(D11, D12, D13, D22, D23, D33)
  mu += mu2_tri(D00, D01, D03, D11, D13, D33)
  return mu * 0.5

  
cpdef double mu1_tet(double D00, double D01, double D02, double D03,
                     double D11, double D12, double D13, 
                     double D22, double D23,
                     double D33) nogil:
  """ Return 3rd intinsic volume of tetrahedron
  
  Compute the 3rd intrinsic volume (sum of external angles * edge
  lengths) of a tetrahedron with coordinates [coords[v0], coords[v1],
  coords[v2], coords[v3]].

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1, v2, v3 : int
       Indices for vertices of the tetrahedron.

  Outputs:
  --------
  mu1 : float
  """
  cdef double mu
  mu = 0
  mu += _mu1_tetface(D00, D01, D11, D02, D03, D12, D13, D22, D23, D33)
  mu += _mu1_tetface(D00, D02, D22, D01, D03, D12, D23, D11, D13, D33)
  mu += _mu1_tetface(D00, D03, D33, D01, D02, D13, D23, D11, D12, D22)
  mu += _mu1_tetface(D11, D12, D22, D01, D13, D02, D23, D00, D03, D33)
  mu += _mu1_tetface(D11, D13, D33, D01, D12, D03, D23, D00, D02, D22)
  mu += _mu1_tetface(D22, D23, D33, D02, D12, D03, D13, D00, D01, D11)
  return mu


@cython.cdivision(True)
cdef double _mu1_tetface(double Ds0s0,
                         double Ds0s1,
                         double Ds1s1,
                         double Ds0t0,
                         double Ds0t1,
                         double Ds1t0,
                         double Ds1t1,
                         double Dt0t0,
                         double Dt0t1,
                         double Dt1t1) nogil:
    cdef double A00, A01, A02, A11, A12, A22, np_len, a
    cdef double length, norm_proj0, norm_proj1, inner_prod_proj

    A00 = Ds1s1 - 2 * Ds0s1 + Ds0s0
    # all norms divided by this value, leading to NaN value for output
    if A00 == 0:
      return 0
    A11 = Dt0t0 - 2 * Ds0t0 + Ds0s0
    A22 = Dt1t1 - 2 * Ds0t1 + Ds0s0
    A01 = Ds1t0 - Ds0t0 - Ds0s1 + Ds0s0
    A02 = Ds1t1 - Ds0t1 - Ds0s1 + Ds0s0
    A12 = Dt0t1 - Ds0t0 - Ds0t1 + Ds0s0
    length = sqrt(A00)
    norm_proj0 = A11 - A01 * A01 / A00
    norm_proj1 = A22 - A02 * A02 / A00
    inner_prod_proj = A12 - A01 * A02 / A00
    np_len = norm_proj0 * norm_proj1
    if np_len <= 0: # would otherwise lead to NaN return value
      return 0
    a = (PI - acos(inner_prod_proj / sqrt(np_len))) * length / (2 * PI)
    return a


cpdef double mu2_tri(double D00, double D01, double D02,
                     double D11, double D12,
                     double D22) nogil:
  """
  Compute the 2nd intrinsic volume (just area in this case) of
  a triangle with coordinates [coords[v0], coords[v1], coords[v2]].

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1, v2 : int
       Indices for vertices of the tetrahedron.

  Returns
  -------
  mu2 : float
  """
  cdef double C00, C01, C11, L
  C00 = D11 - 2*D01 + D00
  C01 = D12 - D01 - D02 + D00
  C11 = D22 - 2*D02 + D00
  L = C00 * C11 - C01 * C01
  # Negative area appeared to result from floating point errors on PPC
  if L < 0:
      return 0.0
  return sqrt(L) * 0.5


cpdef double mu1_tri(double D00, double D01, double D02,
                     double D11, double D12,
                     double D22) nogil:
  """
  Compute the 1st intrinsic volume (1/2 the perimeter of
  a triangle with coordinates [coords[v0], coords[v1], coords[v2]].

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1, v2 : int
       Indices for vertices of the tetrahedron.

  Returns
  -------
  mu1 : float
  """
  cdef double mu = 0
  mu += mu1_edge(D00, D01, D11)
  mu += mu1_edge(D00, D02, D22)
  mu += mu1_edge(D11, D12, D22)
  return mu * 0.5

  
cpdef double mu1_edge(double D00, double D01, 
                      double D11) nogil:
  """
  Compute the 1st intrinsic volume (length)
  of a line segment with coordinates [coords[v0], coords[v1]]

  Parameters
  ----------
  coords : ndarray((*,2))
       An array of coordinates of vertices of tetrahedron.
  v0, v1 : int
       Indices for vertices of the tetrahedron.

  Returns
  -------
  mu0 : float
  """
  return sqrt(D00 - 2*D01 + D11)


def EC3d(np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a 3d mask, compute the 0th intrinsic volume
    (Euler characteristic)
    of the masked region. The region is broken up into tetrahedra /
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Parameters
    ----------
    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask
    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu0 : int

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume, as
    described in the reference below.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' mask (1d array)
    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra 
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d2
    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d4

    cdef long i, j, k, l, s0, s1, s2, ds2, ds3, ds4, index, m, nvox
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef long l0 = 0

    cdef np.ndarray[DTYPE_int_t, ndim=3] pmask
    pmask = np.zeros((mask.shape[0]+1, mask.shape[1]+1, mask.shape[2]+1), np.int)
    pmask[:-1,:-1,:-1] = mask

    s0, s1, s2 = (pmask.shape[0], pmask.shape[1], pmask.shape[2])

    fmask = pmask.reshape((s0*s1*s2))

    strides = np.empty((s0, s1, s2), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,0,1), strides),
                             cube_with_strides_center((0,1,0), strides),
                             cube_with_strides_center((0,1,1), strides),
                             cube_with_strides_center((1,0,0), strides),
                             cube_with_strides_center((1,0,1), strides),
                             cube_with_strides_center((1,1,0), strides),
                             cube_with_strides_center((1,1,1), strides)])
    c = cube_with_strides_center((0,0,0), strides)

    d4 = np.array(list(c[4].difference(union[4])))
    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds2 = d2.shape[0]
    ds3 = d3.shape[0]
    ds4 = d4.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]
    ss2 = strides[2]

    nvox = s0*s1*s2

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fmask[v1] * fmask[v2] * fmask[v3]
                        l0 = l0 - m

                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fmask[v1] * fmask[v2]
                        l0 = l0 + m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fmask[v1]
                        l0 = l0 - m
    l0 += mask.sum()
    return l0


def Lips3d(np.ndarray[DTYPE_float_t, ndim=4] coords,
           np.ndarray[DTYPE_int_t, ndim=3] mask):
    """
    Given a 3d mask and coordinates, estimate the intrinsic volumes
    of the masked region. The region is broken up into tetrahedra / 
    triangles / edges / vertices, which are included based on whether
    all voxels in the tetrahedron / triangle / edge / vertex are
    in the mask or not.

    Parameters
    ----------
    coords : ndarray((*,i,j,k))
         Coordinates for the voxels in the mask
    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu : ndarray
         Array of intrinsic volumes [mu0, mu1, mu2, mu3]

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume, as
    described in the reference below.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' coords (2d array)
    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords 
    cdef np.ndarray[DTYPE_float_t, ndim=2] D

    # 'flattened' mask (1d array)

    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask
    cdef np.ndarray[DTYPE_int_t, ndim=1] fpmask
    cdef np.ndarray[DTYPE_int_t, ndim=3] pmask

    # d3 and d4 are lists of triangles and tetrahedra 
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d4
    cdef np.ndarray[DTYPE_int_t, ndim=2] m4
    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] m3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d2
    cdef np.ndarray[DTYPE_int_t, ndim=2] m2
    cdef np.ndarray[DTYPE_int_t, ndim=1] cvertices

    cdef long i, j, k, l, s0, s1, s2, ds4, ds3, ds2
    cdef long index, pindex, m, nvox, r, s, rr, ss, mr, ms
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices for mask
    cdef long w0, w1, w2, w3 # vertices for data
    cdef double l0, l1, l2, l3
    cdef double res

    l0 = 0; l1 = 0; l2 = 0; l3 = 0

    pmask = np.zeros((mask.shape[0]+1, mask.shape[1]+1, mask.shape[2]+1), np.int)
    pmask[:-1,:-1,:-1] = mask

    s0, s1, s2 = (pmask.shape[0], pmask.shape[1], pmask.shape[2])

    fpmask = pmask.reshape((s0*s1*s2))
    fmask = mask.reshape((s0-1)*(s1-1)*(s2-1))

    if (mask.shape[0], mask.shape[1], mask.shape[2]) != (coords.shape[1], coords.shape[2], coords.shape[3]):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0-1)*(s1-1)*(s2-1)))
    n = fcoords.shape[0]

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    # The mask is copied into a larger array,
    # hence it will have different strides than the data

    strides = np.empty((s0, s1, s2), np.bool).strides
    dstrides = np.empty((s0-1, s1-1, s2-1), np.bool).strides
    cvertices = np.array([[[dstrides[0]*i+dstrides[1]*j+dstrides[2]*k for i in range(2)] for j in range(2)] for k in range(2)]).ravel()
    cvertices.sort()

    union = join_complexes(*[cube_with_strides_center((0,0,1), strides),
                             cube_with_strides_center((0,1,0), strides),
                             cube_with_strides_center((0,1,1), strides),
                             cube_with_strides_center((1,0,0), strides),
                             cube_with_strides_center((1,0,1), strides),
                             cube_with_strides_center((1,1,0), strides),
                             cube_with_strides_center((1,1,1), strides)])
    c = cube_with_strides_center((0,0,0), strides)
    m4 = np.array(list(c[4].difference(union[4])))
    m3 = np.array(list(c[3].difference(union[3])))
    m2 = np.array(list(c[2].difference(union[2])))

    d4 = np.array([[_convert_stride3(v, strides, (4,2,1)) for v in m4[i]] for i in range(m4.shape[0])])
    d4 = np.hstack([m4, d4])
    ds4 = d4.shape[0]

    d3 = np.array([[_convert_stride3(v, strides, (4,2,1)) for v in m3[i]] for i in range(m3.shape[0])])
    d3 = np.hstack([m3, d3])
    ds3 = d3.shape[0]

    d2 = np.array([[_convert_stride3(v, strides, (4,2,1)) for v in m2[i]] for i in range(m2.shape[0])])
    d2 = np.hstack([m2, d2])
    ds2 = d2.shape[0]

    ss0, ss1, ss2 = strides[0], strides[1], strides[2]
    ss0d, ss1d, ss2d = dstrides[0], dstrides[1], dstrides[2]

    nvox = (s0-1)*(s1-1)*(s2-1)

    D = np.zeros((8,8))

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):

                pindex = i*ss0+j*ss1+k*ss2
                index = i*ss0d+j*ss1d+k*ss2d
                for r in range(8):
                    rr = (index+cvertices[r]) % nvox
                    mr = fmask[rr]
                    for s in range(r+1):
                        res = 0
                        ss = (index+cvertices[s]) % nvox
                        ms = fmask[ss]
                        if mr * ms:
                            for l in range(fcoords.shape[0]):
                                res += fcoords[l,ss] * fcoords[l,rr]
                            D[r,s] = res
                            D[s,r] = res
                        else:
                            D[r,s] = 0
                            D[s,r] = 0

                for l in range(ds4):
                    v0 = pindex + d4[l,0]
                    w0 = d4[l,4]
                    m = fpmask[v0]
                    if m:
                        v1 = pindex + d4[l,1]
                        v2 = pindex + d4[l,2]
                        v3 = pindex + d4[l,3]
                        w1 = d4[l,5]
                        w2 = d4[l,6]
                        w3 = d4[l,7]

                        m = m * fpmask[v1] * fpmask[v2] * fpmask[v3]
                        d = m * mu3_tet(D[w0,w0], D[w0,w1], D[w0,w2], 
                                        D[w0,w3], D[w1,w1], D[w1,w2], 
                                        D[w1,w3], D[w2,w2], D[w2,w3],
                                        D[w3,w3])

                        l3 = l3 + m * mu3_tet(D[w0,w0], D[w0,w1], D[w0,w2], 
                                              D[w0,w3], D[w1,w1], D[w1,w2], 
                                              D[w1,w3], D[w2,w2], D[w2,w3],
                                              D[w3,w3])

                        l2 = l2 - m * mu2_tet(D[w0,w0], D[w0,w1], D[w0,w2], 
                                              D[w0,w3], D[w1,w1], D[w1,w2], 
                                              D[w1,w3], D[w2,w2], D[w2,w3],
                                              D[w3,w3])

                        l1 = l1 + m * mu1_tet(D[w0,w0], D[w0,w1], D[w0,w2], 
                                              D[w0,w3], D[w1,w1], D[w1,w2], 
                                              D[w1,w3], D[w2,w2], D[w2,w3],
                                              D[w3,w3])

                        l0 = l0 - m

                for l in range(ds3):
                    v0 = pindex + d3[l,0]
                    w0 = d3[l,3]
                    m = fpmask[v0]
                    if m:
                        v1 = pindex + d3[l,1]
                        v2 = pindex + d3[l,2]
                        w1 = d3[l,4]
                        w2 = d3[l,5]

                        m = m * fpmask[v1] * fpmask[v2] 
                        l2 = l2 + m * mu2_tri(D[w0,w0], D[w0,w1], D[w0,w2], 
                                              D[w1,w1], D[w1,w2], D[w2,w2]) 

                        l1 = l1 - m * mu1_tri(D[w0,w0], D[w0,w1], D[w0,w2], 
                                              D[w1,w1], D[w1,w2], D[w2,w2]) 

                        l0 = l0 + m

                for l in range(ds2):
                    v0 = pindex + d2[l,0]
                    w0 = d2[l,2]
                    m = fpmask[v0]
                    if m:
                        v1 = pindex + d2[l,1]
                        w1 = d2[l,3]
                        m = m * fpmask[v1]
                        l1 = l1 + m * mu1_edge(D[w0,w0], D[w0,w1], D[w1,w1])

                        l0 = l0 - m

    l0 += mask.sum()
    return np.array([l0, l1, l2, l3])


def _convert_stride3(v, stride1, stride2):
    """
    Take a voxel, expressed as in index in stride1 and
    re-express it as an index in stride2
    """
    v0 = v / stride1[0]
    v -= v0 * stride1[0]
    v1 = v / stride1[1]
    v2 = v - v1 * stride1[1]
    return v0*stride2[0] + v1*stride2[1] + v2*stride2[2]


def _convert_stride2(v, stride1, stride2):
    """
    Take a voxel, expressed as in index in stride1 and
    re-express it as an index in stride2
    """
    v0 = v / stride1[0]
    v1 = v - v0 * stride1[0]
    return v0*stride2[0] + v1*stride2[1]


def _convert_stride1(v, stride1, stride2):
    """
    Take a voxel, expressed as in index in stride1 and
    re-express it as an index in stride2
    """
    v0 = v / stride1[0]
    return v0 * stride2[0]


def Lips2d(np.ndarray[DTYPE_float_t, ndim=3] coords,
           np.ndarray[DTYPE_int_t, ndim=2] mask):
    """
    Given a 2d mask and coordinates, estimate the intrinsic volumes
    of the masked region. The region is broken up into
    triangles / edges / vertices, which are included based on whether
    all voxels in the triangle / edge / vertex are
    in the mask or not.

    Parameters
    ----------
    coords : ndarray((*,i,j))
         Coordinates for the voxels in the mask
    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu : ndarray
         Array of intrinsic volumes [mu0, mu1, mu2]

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' coords (2d array)
    cdef np.ndarray[DTYPE_float_t, ndim=2] fcoords 

    # 'flattened' mask (1d array)
    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask
    cdef np.ndarray[DTYPE_int_t, ndim=1] fpmask
    cdef np.ndarray[DTYPE_int_t, ndim=2] pmask

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef np.ndarray[DTYPE_int_t, ndim=2] d3
    cdef np.ndarray[DTYPE_int_t, ndim=2] d2

    cdef long i, j, k, l, r, s, rr, ss, mr, ms, s0, s1, ds2, ds3, index, m, npix
    cdef long ss0, ss1, ss0d, ss1d # strides
    cdef long v0, v1, v2 # vertices
    cdef double l0, l1, l2
    cdef double res

    l0 = 0; l1 = 0; l2 = 0; l3 = 0

    pmask = np.zeros((mask.shape[0]+1, mask.shape[1]+1), np.int)
    pmask[:-1,:-1] = mask

    s0, s1 = pmask.shape[0], pmask.shape[1]

    if (mask.shape[0], mask.shape[1]) != (coords.shape[1], coords.shape[2]):
        raise ValueError('shape of mask does not match coordinates')

    fcoords = coords.reshape((coords.shape[0], (s0-1)*(s1-1)))
    n = fcoords.shape[0]
    fpmask = pmask.reshape((s0*s1))
    fmask = mask.reshape((s0-1)*(s1-1))

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    strides = np.empty((s0, s1), np.bool).strides
    dstrides = np.empty((s0-1, s1-1), np.bool).strides
    cvertices = np.array([[dstrides[0]*i+dstrides[1]*j for i in range(2)] for j in range(2)]).ravel()
    cvertices.sort()

    union = join_complexes(*[cube_with_strides_center((0,1), strides),
                             cube_with_strides_center((1,0), strides),
                             cube_with_strides_center((1,1), strides)])

    c = cube_with_strides_center((0,0), strides)
    m3 = np.array(list(c[3].difference(union[3])))
    m2 = np.array(list(c[2].difference(union[2])))

    d3 = np.array([[_convert_stride2(v, strides, (2,1)) for v in m3[i]] for i in range(m3.shape[0])])
    d3 = np.hstack([m3, d3])
    ds3 = d3.shape[0]

    d2 = np.array([[_convert_stride2(v, strides, (2,1)) for v in m2[i]] for i in range(m2.shape[0])])
    d2 = np.hstack([m2, d2])
    ds2 = d2.shape[0]

    ss0, ss1 = strides[0], strides[1]
    ss0d, ss1d = dstrides[0], dstrides[1]

    D = np.zeros((4,4))

    npix = (s0-1)*(s1-1)

    for i in range(s0-1):
        for j in range(s1-1):
          pindex = i*ss0+j*ss1
          index = i*ss0d+j*ss1d

          for r in range(4):
            rr = (index+cvertices[r]) % npix
            mr = fmask[rr]
            for s in range(r+1):
              res = 0
              ss = (index+cvertices[s]) % npix
              ms = fmask[ss]
              if mr * ms:
                for l in range(fcoords.shape[0]):
                  res += fcoords[l,ss] * fcoords[l,rr]
                D[r,s] = res
                D[s,r] = res
              else:
                D[r,s] = 0
                D[s,r] = 0

          for l in range(ds3):
            v0 = pindex + d3[l,0]
            w0 = d3[l,3]
            m = fpmask[v0]
            if m:
              v1 = pindex + d3[l,1]
              v2 = pindex + d3[l,2]
              w1 = d3[l,4]
              w2 = d3[l,5]
              m = m * fpmask[v1] * fpmask[v2]
              l2 = l2 + mu2_tri(D[w0,w0], D[w0,w1], D[w0,w2],
                                D[w1,w1], D[w1,w2], D[w2,w2]) * m
              l1 = l1 - mu1_tri(D[w0,w0], D[w0,w1], D[w0,w2],
                                D[w1,w1], D[w1,w2], D[w2,w2]) * m
              l0 = l0 + m

          for l in range(ds2):
            v0 = pindex + d2[l,0]
            w0 = d2[l,2]
            m = fpmask[v0]
            if m:
              v1 = pindex + d2[l,1]
              w1 = d2[l,3]
              m = m * fpmask[v1]
              l1 = l1 + m * mu1_edge(D[w0,w0], D[w0,w1], D[w1,w1])
              l0 = l0 - m

    l0 += mask.sum()
    return np.array([l0,l1,l2])


def EC2d(np.ndarray[DTYPE_int_t, ndim=2] mask):
    """
    Given a 2d mask, compute the 0th intrinsic volume
    (Euler characteristic)
    of the masked region. The region is broken up into 
    triangles / edges / vertices, which are included based on whether
    all voxels in the triangle / edge / vertex are
    in the mask or not.

    Parameters
    ----------
    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu0 : int

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume, as
    described in the reference below.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' mask (1d array)
    cdef np.ndarray[DTYPE_int_t, ndim=1] fmask

    # d3 and d4 are lists of triangles and tetrahedra 
    # associated to particular voxels in the cuve

    cdef np.ndarray[DTYPE_int_t, ndim=2] d2
    cdef np.ndarray[DTYPE_int_t, ndim=2] d3

    cdef long i, j, k, l, s0, s1, s2, ds2, ds3, index, m
    cdef long ss0, ss1, ss2 # strides
    cdef long v0, v1, v2, v3 # vertices
    cdef long l0 = 0

    cdef np.ndarray[DTYPE_int_t, ndim=2] pmask
    pmask = np.zeros((mask.shape[0]+1, mask.shape[1]+1), np.int)
    pmask[:-1,:-1] = mask

    s0, s1 = (pmask.shape[0], pmask.shape[1])

    fmask = pmask.reshape((s0*s1))

    strides = np.empty((s0, s1), np.bool).strides

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    union = join_complexes(*[cube_with_strides_center((0,1), strides),
                             cube_with_strides_center((1,0), strides),
                             cube_with_strides_center((1,1), strides)])
    c = cube_with_strides_center((0,0), strides)

    d3 = np.array(list(c[3].difference(union[3])))
    d2 = np.array(list(c[2].difference(union[2])))

    ds2 = d2.shape[0]
    ds3 = d3.shape[0]

    ss0 = strides[0]
    ss1 = strides[1]

    for i in range(s0-1):
        for j in range(s1-1):
          index = i*ss0+j*ss1

          for l in range(ds3):
            v0 = index + d3[l,0]
            m = fmask[v0]
            if m and v0:
              v1 = index + d3[l,1]
              v2 = index + d3[l,2]
              m = m * fmask[v1] * fmask[v2]
              l0 = l0 + m

          for l in range(ds2):
            v0 = index + d2[l,0]
            m = fmask[v0]
            if m:
              v1 = index + d2[l,1]
              m = m * fmask[v1]
              l0 = l0 - m
    l0 += mask.sum()
    return l0


def Lips1d(np.ndarray[DTYPE_float_t, ndim=2] coords,
           np.ndarray[DTYPE_int_t, ndim=1] mask):
    """
    Given a 1d mask and coordinates, estimate the intrinsic volumes
    of the masked region. The region is broken up into
    edges / vertices, which are included based on whether
    all voxels in the edge / vertex are
    in the mask or not.

    Parameters
    ----------
    coords : ndarray((*,i))
         Coordinates for the voxels in the mask
    mask : ndarray((i,), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu : ndarray
         Array of intrinsic volumes [mu0, mu1]

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' coords (2d array)

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef long i, j, k, l, r, s, rr, ss, mr, ms, s0, ds2, index, m
    cdef long ss0, ss0d # strides
    cdef long v0, v1 # vertices
    cdef double l0, l1
    cdef double res

    l0 = 0; l1 = 0

    s0 = mask.shape[0]

    if mask.shape[0] != coords.shape[1]:
        raise ValueError('shape of mask does not match coordinates')

    n = coords.shape[0]

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    D = np.zeros((2,2))

    for i in range(s0):
      for r in range(2):
        rr = (i+r) % s0
        mr = mask[rr]
        for s in range(r+1):
          res = 0
          ss = (i+s) % s0
          ms = mask[ss]
          if mr * ms * ((i+r) < s0) * ((i+s) < s0):
            for l in range(coords.shape[0]):
              res += coords[l,ss] * coords[l,rr]
            D[r,s] = res
            D[s,r] = res
          else:
            D[r,s] = 0
            D[s,r] = 0

      m = mask[i]
      if m:
        m = m * (mask[(i+1) % s0] * (i < s0))
        l1 = l1 + m * mu1_edge(D[0,0], D[0,1], D[1,1])
        l0 = l0 - m

    l0 += mask.sum()
    return np.array([l0,l1])


def EC1d(np.ndarray[DTYPE_int_t, ndim=1] mask):
    """
    Given a 1d mask, compute the 0th intrinsic volume
    (Euler characteristic)
    of the masked region. The region is broken up into 
    edges / vertices, which are included based on whether
    all voxels in the edge / vertex are
    in the mask or not.

    Parameters
    ----------
    mask : ndarray((i,), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu0 : int

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it
    is not clear how to get cython to use np.bool arrays.

    The 3d cubes are triangulated into 6 tetrahedra of equal volume, as
    described in the reference below.

    References
    ----------
    Taylor, J.E. & Worsley, K.J. (2007). "Detecting sparse signal in random fields,
      with an application to brain mapping."
      Journal of the American Statistical Association, 102(479):913-928.
    """
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    # 'flattened' coords (2d array)

    # d3 and d4 are lists of triangles
    # associated to particular voxels in the square

    cdef long i, m, s0
    cdef double l0 = 0

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    s0 = mask.shape[0]
    for i in range(s0):
      m = mask[i]
      if m:
        m = m * (mask[(i+1) % s0] * (i < s0))
        l0 = l0 - m

    l0 += mask.sum()
    return l0
