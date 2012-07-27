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

# Array helper
from nipy.utils.arrays import strides_from

# local imports
from utils import cube_with_strides_center, join_complexes


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
  """ Compute the 3rd intrinsic volume of a tetrahedron.

  3rd intrinsic volume (just volume in this case) of a tetrahedron with
  coordinates implied by dot products below.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the first vertex, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the second
    vertex.
  D02 : float
    ``cv0.dot(cv2)``
  D03 : float
    ``cv0.dot(cv3)``
  D11 : float
    ``cv1.dot(cv1)``
  D12 : float
    ``cv1.dot(cv2)``
  D13 : float
    ``cv1.dot(cv3)``
  D22 : float
    ``cv2.dot(cv2)``
  D23 : float
    ``cv2.dot(cv2)``
  D33 : float
    ``cv3.dot(cv3)``

  Returns
  -------
  mu3 : float
    volume of tetrahedron
  """
  cdef double C00, C01, C02, C11, C12, C22, v2
  C00 = D00 - 2*D03 + D33
  C01 = D01 - D13 - D03 + D33
  C02 = D02 - D23 - D03 + D33
  C11 = D11 - 2*D13 + D33
  C12 = D12 - D13 - D23 + D33
  C22 = D22 - 2*D23 + D33
  v2 = (C00 * (C11 * C22 - C12 * C12) -
        C01 * (C01 * C22 - C02 * C12) +
        C02 * (C01 * C12 - C11 * C02))
  # Rounding errors near 0 cause NaNs
  if v2 <= 0:
      return 0
  return sqrt(v2) / 6.


cpdef double mu2_tet(double D00, double D01, double D02, double D03,
                     double D11, double D12, double D13,
                     double D22, double D23,
                     double D33) nogil:
  """ Compute the 2nd intrinsic volume of tetrahedron

  2nd intrinsic volume (half the surface area) of a tetrahedron with coordinates
  implied by dot products below.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the first vertex, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the second
    vertex.
  D02 : float
    ``cv0.dot(cv2)``
  D03 : float
    ``cv0.dot(cv3)``
  D11 : float
    ``cv1.dot(cv1)``
  D12 : float
    ``cv1.dot(cv2)``
  D13 : float
    ``cv1.dot(cv3)``
  D22 : float
    ``cv2.dot(cv2)``
  D23 : float
    ``cv2.dot(cv2)``
  D33 : float
    ``cv3.dot(cv3)``

  Returns
  -------
  mu2 : float
    Half tetrahedron surface area
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
  """ Return 3rd intrinsic volume of tetrahedron

  Compute the 3rd intrinsic volume (sum of external angles * edge
  lengths) of a tetrahedron for which the input arguments represent the
  coordinate dot products of the vertices.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the first vertex, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the second
    vertex.
  D02 : float
    ``cv0.dot(cv2)``
  D03 : float
    ``cv0.dot(cv3)``
  D11 : float
    ``cv1.dot(cv1)``
  D12 : float
    ``cv1.dot(cv2)``
  D13 : float
    ``cv1.dot(cv3)``
  D22 : float
    ``cv2.dot(cv2)``
  D23 : float
    ``cv2.dot(cv2)``
  D33 : float
    ``cv3.dot(cv3)``

  Returns
  -------
  mu1 : float
    3rd intrinsic volume of tetrahedron
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


cdef inline double limited_acos(double val) nogil:
    """ Check for -1 <= val <= 1 before returning acos(val)

    Avoids nan values from small rounding errors
    """
    if val >= 1:
        return 0
    elif val <= -1:
        return PI
    return acos(val)


@cython.cdivision(True)
cpdef double _mu1_tetface(double Ds0s0,
                          double Ds0s1,
                          double Ds1s1,
                          double Ds0t0,
                          double Ds0t1,
                          double Ds1t0,
                          double Ds1t1,
                          double Dt0t0,
                          double Dt0t1,
                          double Dt1t1) nogil:
    cdef double A00, A01, A02, A11, A12, A22, np_len, a, acosval
    cdef double length, norm_proj0, norm_proj1, inner_prod_proj

    A00 = Ds1s1 - 2 * Ds0s1 + Ds0s0
    # all norms divided by this value, leading to NaN value for output, for
    # values <= 0
    if A00 <= 0:
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
    # hedge for small rounding errors above 1 and below -1
    acosval = limited_acos(inner_prod_proj / sqrt(np_len))
    a = (PI - acosval) * length / (2 * PI)
    return a


cpdef double mu2_tri(double D00, double D01, double D02,
                     double D11, double D12,
                     double D22) nogil:
  """ Compute the 2nd intrinsic volume of triangle

  2nd intrinsic volume (just area in this case) of a triangle with coordinates
  implied by the dot products below.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the first vertex, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the second
    vertex.
  D02 : float
    ``cv0.dot(cv2)``
  D11 : float
    ``cv1.dot(cv1)``
  D12 : float
    ``cv1.dot(cv2)``
  D22 : float
    ``cv2.dot(cv2)``

  Returns
  -------
  mu2 : float
    area of triangle
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
  """ Compute the 1st intrinsic volume of triangle

  1st intrinsic volume (1/2 the perimeter) of a triangle with coordinates
  implied by the dot products below.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the first vertex, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the second
    vertex.
  D02 : float
    ``cv0.dot(cv2)``
  D11 : float
    ``cv1.dot(cv1)``
  D12 : float
    ``cv1.dot(cv2)``
  D22 : float
    ``cv2.dot(cv2)``

  Returns
  -------
  mu1 : float
    1/2 perimeter of triangle
  """
  cdef double mu = 0
  mu += mu1_edge(D00, D01, D11)
  mu += mu1_edge(D00, D02, D22)
  mu += mu1_edge(D11, D12, D22)
  return mu * 0.5


cpdef double mu1_edge(double D00, double D01, double D11) nogil:
  """ Compute the 1st intrinsic volume (length) of line segment

  Length of a line segment with vertex coordinates implied by dot products
  below.

  Parameters
  ----------
  D00 : float
    If ``cv0`` is a 3-vector of coordinates for the line start, `D00` is
    ``cv0.dot(cv0)``
  D01 : float
    ``cv0.dot(cv1)`` where ``cv1`` is the coordinates for the line end.
  D11 : float
    ``cv1.dot(cv1)``

  Returns
  -------
  mu0 : float
    length of line segment
  """
  return sqrt(D00 - 2*D01 + D11)


def EC3d(mask):
    """ Compute Euler characteristic of region within `mask`

    Given a 3d `mask`, compute the 0th intrinsic volume (Euler characteristic)
    of the masked region. The region is broken up into tetrahedra / triangles /
    edges / vertices, which are included based on whether all voxels in the
    tetrahedron / triangle / edge / vertex are in the mask or not.

    Parameters
    ----------
    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not a voxel is in the mask.

    Returns
    -------
    mu0 : int
        Euler characteristic

    Notes
    -----
    The array mask is assumed to be binary. At the time of writing, it is not
    clear how to get cython to use np.bool arrays.

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
    cdef:
        # c-level versions of the array
        np.ndarray[np.intp_t, ndim=3] mask_c
        # 'flattened' mask (1d array)
        np.ndarray[np.intp_t, ndim=1] fpmask
        # d3 and d4 are lists of triangles and tetrahedra
        # associated to particular voxels in the cuve
        np.ndarray[np.intp_t, ndim=2] d2
        np.ndarray[np.intp_t, ndim=2] d3
        np.ndarray[np.intp_t, ndim=2] d4
        # scalars
        np.npy_intp i, j, k, l, s0, s1, s2, ds2, ds3, ds4, index, m, nvox
        np.npy_intp ss0, ss1, ss2 # strides
        np.npy_intp v0, v1, v2, v3 # vertices
        np.npy_intp l0 = 0

    mask_c = mask

    pmask_shape = np.array(mask.shape) + 1
    pmask = np.zeros(pmask_shape, np.int)
    pmask[:-1,:-1,:-1] = mask_c

    s0, s1, s2 = (pmask.shape[0], pmask.shape[1], pmask.shape[2])

    fpmask = pmask.reshape(-1)
    cdef:
        np.ndarray[np.intp_t, ndim=1] strides
    strides = np.array(strides_from(pmask_shape, np.bool), dtype=np.intp)

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

    nvox = mask.size

    for i in range(s0-1):
        for j in range(s1-1):
            for k in range(s2-1):
                index = i*ss0+j*ss1+k*ss2
                for l in range(ds4):
                    v0 = index + d4[l,0]
                    m = fpmask[v0]
                    if m:
                        v1 = index + d4[l,1]
                        v2 = index + d4[l,2]
                        v3 = index + d4[l,3]
                        m = m * fpmask[v1] * fpmask[v2] * fpmask[v3]
                        l0 = l0 - m

                for l in range(ds3):
                    v0 = index + d3[l,0]
                    m = fpmask[v0]
                    if m:
                        v1 = index + d3[l,1]
                        v2 = index + d3[l,2]
                        m = m * fpmask[v1] * fpmask[v2]
                        l0 = l0 + m

                for l in range(ds2):
                    v0 = index + d2[l,0]
                    m = fpmask[v0]
                    if m:
                        v1 = index + d2[l,1]
                        m = m * fpmask[v1]
                        l0 = l0 - m
    l0 += mask.sum()
    return l0


def Lips3d(coords, mask):
    """ Estimated intrinsic volumes within masked region given coordinates

    Given a 3d `mask` and coordinates `coords`, estimate the intrinsic volumes
    of the masked region. The region is broken up into tetrahedra / triangles /
    edges / vertices, which are included based on whether all voxels in the
    tetrahedron / triangle / edge / vertex are in the mask or not.

    Parameters
    ----------
    coords : ndarray((N,i,j,k))
         Coordinates for the voxels in the mask. ``N`` will often be 3 (for 3
         dimensional coordinates, but can be any integer > 0
    mask : ndarray((i,j,k), np.int)
         Binary mask determining whether or not
         a voxel is in the mask.

    Returns
    -------
    mu : ndarray
        Array of intrinsic volumes [mu0, mu1, mu2, mu3], being, respectively:
        #. Euler characteristic
        #. 2 * mean caliper diameter
        #. 0.5 * surface area
        #. Volume.

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
    if mask.shape != coords.shape[1:]:
        raise ValueError('shape of mask does not match coordinates')
    # if the data can be squeezed, we must use the lower dimensional function
    mask = np.squeeze(mask)
    if mask.ndim < 3:
        value = np.zeros(4)
        coords = coords.reshape((coords.shape[0],) + mask.shape)
        if mask.ndim == 2:
            value[:3] = Lips2d(coords, mask)
        elif mask.ndim == 1:
            value[:2] = Lips1d(coords, mask)
        return value

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    cdef:
        # c-level versions of the arrays
        np.ndarray[np.float_t, ndim=4] coords_c
        np.ndarray[np.intp_t, ndim=3] mask_c
        # 'flattened' coords (2d array)
        np.ndarray[np.float_t, ndim=2] fcoords
        np.ndarray[np.float_t, ndim=2] D
        # 'flattened' mask (1d array)
        np.ndarray[np.intp_t, ndim=1] fmask
        np.ndarray[np.intp_t, ndim=1] fpmask
        np.ndarray[np.intp_t, ndim=3] pmask
        # d3 and d4 are lists of triangles and tetrahedra
        # associated to particular voxels in the cube
        np.ndarray[np.intp_t, ndim=2] d4
        np.ndarray[np.intp_t, ndim=2] m4
        np.ndarray[np.intp_t, ndim=2] d3
        np.ndarray[np.intp_t, ndim=2] m3
        np.ndarray[np.intp_t, ndim=2] d2
        np.ndarray[np.intp_t, ndim=2] m2
        np.ndarray[np.intp_t, ndim=1] cvertices
        # scalars
        np.npy_intp i, j, k, l, s0, s1, s2, ds4, ds3, ds2
        np.npy_intp index, pindex, m, nvox, r, s, rr, ss, mr, ms
        np.npy_intp ss0, ss1, ss2 # strides
        np.npy_intp v0, v1, v2, v3 # vertices for mask
        np.npy_intp w0, w1, w2, w3 # vertices for data
        double l0, l1, l2, l3
        double res

    coords_c = coords
    mask_c = mask
    l0 = 0; l1 = 0; l2 = 0; l3 = 0

    pmask_shape = np.array(mask.shape) + 1
    pmask = np.zeros(pmask_shape, np.int)
    pmask[:-1,:-1,:-1] = mask_c

    s0, s1, s2 = (pmask.shape[0], pmask.shape[1], pmask.shape[2])

    fpmask = pmask.reshape(-1)
    fmask = mask_c.reshape(-1)
    fcoords = coords_c.reshape((coords_c.shape[0], -1))

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    # The mask is copied into a larger array, hence it will have different
    # strides than the data
    cdef:
        np.ndarray[np.intp_t, ndim=1] strides
        np.ndarray[np.intp_t, ndim=1] dstrides
    strides = np.array(strides_from(pmask_shape, np.bool), dtype=np.intp)
    dstrides = np.array(strides_from(mask.shape, np.bool), dtype=np.intp)
    ss0, ss1, ss2 = strides[0], strides[1], strides[2]
    ss0d, ss1d, ss2d = dstrides[0], dstrides[1], dstrides[2]
    verts = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                verts.append(ss0d * i + ss1d * j + ss2d * k)
    cvertices = np.array(sorted(verts), np.intp)

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

    nvox = mask.size

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
    v0 = v // stride1[0]
    v -= v0 * stride1[0]
    v1 = v // stride1[1]
    v2 = v - v1 * stride1[1]
    return v0*stride2[0] + v1*stride2[1] + v2*stride2[2]


def _convert_stride2(v, stride1, stride2):
    """
    Take a voxel, expressed as in index in stride1 and
    re-express it as an index in stride2
    """
    v0 = v // stride1[0]
    v1 = v - v0 * stride1[0]
    return v0*stride2[0] + v1*stride2[1]


def _convert_stride1(v, stride1, stride2):
    """
    Take a voxel, expressed as in index in stride1 and
    re-express it as an index in stride2
    """
    v0 = v // stride1[0]
    return v0 * stride2[0]


def Lips2d(coords, mask):
    """ Estimate intrinsic volumes for 2d region in `mask` given `coords`

    Given a 2d `mask` and coordinates `coords`, estimate the intrinsic volumes
    of the masked region. The region is broken up into triangles / edges /
    vertices, which are included based on whether all voxels in the triangle /
    edge / vertex are in the mask or not.

    Parameters
    ----------
    coords : ndarray((N,i,j,k))
         Coordinates for the voxels in the mask. ``N`` will often be 2 (for 2
         dimensional coordinates, but can be any integer > 0
    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not a voxel is in the mask.

    Returns
    -------
    mu : ndarray
        Array of intrinsic volumes [mu0, mu1, mu2], being, respectively:
        #. Euler characteristic
        #. 2 * mean caliper diameter
        #. Area.

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
    if mask.shape != coords.shape[1:]:
        raise ValueError('shape of mask does not match coordinates')
    # if the data can be squeezed, we must use the lower dimensional function
    mask = np.squeeze(mask)
    if mask.ndim == 1:
        value = np.zeros(3)
        coords = coords.reshape((coords.shape[0],) + mask.shape)
        value[:2] = Lips1d(coords, mask)
        return value

    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')

    cdef:
        # c-level versions of the arrays
        np.ndarray[np.float_t, ndim=3] coords_c
        np.ndarray[np.intp_t, ndim=2] mask_c
        # 'flattened' coords (2d array)
        np.ndarray[np.float_t, ndim=2] fcoords
        np.ndarray[np.float_t, ndim=2] D
        # 'flattened' mask (1d array)
        np.ndarray[np.intp_t, ndim=1] fmask
        np.ndarray[np.intp_t, ndim=1] fpmask
        np.ndarray[np.intp_t, ndim=2] pmask
        # d2 and d3 are lists of triangles associated to particular voxels in
        # the square
        np.ndarray[np.intp_t, ndim=2] d3
        np.ndarray[np.intp_t, ndim=2] d2
        np.ndarray[np.intp_t, ndim=1] cvertices
        # scalars
        np.npy_intp i, j, k, l, r, s, rr, ss, mr, ms, s0, s1
        np.npy_intp ds2, ds3, index, m, npix
        np.npy_intp ss0, ss1, ss0d, ss1d # strides
        np.npy_intp v0, v1, v2 # vertices
        double l0, l1, l2
        double res

    coords_c = coords
    mask_c = mask
    l0 = 0; l1 = 0; l2 = 0

    pmask_shape = np.array(mask.shape) + 1
    pmask = np.zeros(pmask_shape, np.int)
    pmask[:-1,:-1] = mask_c

    s0, s1 = pmask.shape[0], pmask.shape[1]

    fpmask = pmask.reshape(-1)
    fmask = mask_c.reshape(-1)
    fcoords = coords.reshape((coords.shape[0], -1))

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    # The mask is copied into a larger array, hence it will have different
    # strides than the data
    cdef:
        np.ndarray[np.intp_t, ndim=1] strides
        np.ndarray[np.intp_t, ndim=1] dstrides
    strides = np.array(strides_from(pmask_shape, np.bool), dtype=np.intp)
    dstrides = np.array(strides_from(mask.shape, np.bool), dtype=np.intp)
    ss0, ss1 = strides[0], strides[1]
    ss0d, ss1d = dstrides[0], dstrides[1]
    verts = []
    for i in range(2):
        for j in range(2):
            verts.append(ss0d * i + ss1d * j)
    cvertices = np.array(sorted(verts), np.intp)

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

    D = np.zeros((4,4))

    npix = mask.size

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


def EC2d(mask):
    """ Compute Euler characteristic of 2D region in `mask`

    Given a 2d `mask`, compute the 0th intrinsic volume (Euler characteristic)
    of the masked region. The region is broken up into triangles / edges /
    vertices, which are included based on whether all voxels in the triangle /
    edge / vertex are in the mask or not.

    Parameters
    ----------
    mask : ndarray((i,j), np.int)
         Binary mask determining whether or not a voxel is in the mask.

    Returns
    -------
    mu0 : int
        Euler characteristic

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
    cdef:
        # c-level versions of the array
        np.ndarray[np.intp_t, ndim=2] mask_c
        # 'flattened' mask (1d array)
        np.ndarray[np.intp_t, ndim=1] fpmask
        # d2 and d3 are lists of triangles and tetrahedra
        # associated to particular voxels in the cuve
        np.ndarray[np.intp_t, ndim=2] d2
        np.ndarray[np.intp_t, ndim=2] d3
        # scalars
        np.npy_intp i, j, k, l, s0, s1, ds2, ds3, index, m
        np.npy_intp ss0, ss1 # strides
        np.npy_intp v0, v1 # vertices
        long l0 = 0

    mask_c = mask

    pmask_shape = np.array(mask.shape) + 1
    pmask = np.zeros(pmask_shape, np.int)
    pmask[:-1,:-1] = mask_c

    s0, s1 = (pmask.shape[0], pmask.shape[1])

    fpmask = pmask.reshape(-1)

    cdef:
        np.ndarray[np.intp_t, ndim=1] strides
    strides = np.array(strides_from(pmask_shape, np.bool), dtype=np.intp)
    ss0, ss1 = strides[0], strides[1]

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

    for i in range(s0-1):
        for j in range(s1-1):
            index = i*ss0+j*ss1
            for l in range(ds3):
                v0 = index + d3[l,0]
                m = fpmask[v0]
                if m and v0:
                    v1 = index + d3[l,1]
                    v2 = index + d3[l,2]
                    m = m * fpmask[v1] * fpmask[v2]
                    l0 = l0 + m

            for l in range(ds2):
                v0 = index + d2[l,0]
                m = fpmask[v0]
                if m:
                    v1 = index + d2[l,1]
                    m = m * fpmask[v1]
                    l0 = l0 - m

    l0 += mask.sum()
    return l0


def Lips1d(np.ndarray[np.float_t, ndim=2] coords,
           np.ndarray[np.intp_t, ndim=1] mask):
    """ Estimate intrinsic volumes for 1D region in `mask` given `coords`

    Given a 1d `mask` and coordinates `coords`, estimate the intrinsic volumes
    of the masked region. The region is broken up into edges / vertices, which
    are included based on whether all voxels in the edge / vertex are in the
    mask or not.

    Parameters
    ----------
    coords : ndarray((N,i,j,k))
         Coordinates for the voxels in the mask. ``N`` will often be 1 (for 1
         dimensional coordinates, but can be any integer > 0
    mask : ndarray((i,), np.int)
         Binary mask determining whether or not a voxel is in the mask.

    Returns
    -------
    mu : ndarray
        Array of intrinsic volumes [mu0, mu1], being, respectively:
        #. Euler characteristic
        #. Line segment length

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
    if mask.shape[0] != coords.shape[1]:
        raise ValueError('shape of mask does not match coordinates')
    if not set(np.unique(mask)).issubset([0,1]):
      raise ValueError('mask should be filled with 0/1 '
                       'values, but be of type np.int')
    cdef:
        np.npy_intp i, l, r, s, rr, ss, mr, ms, s0, index, m
        double l0, l1
        double res

    l0 = 0; l1 = 0
    s0 = mask.shape[0]
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
            m = m * (mask[(i+1) % s0] * ((i+1) < s0))
            l1 = l1 + m * mu1_edge(D[0,0], D[0,1], D[1,1])
            l0 = l0 - m

    l0 += mask.sum()
    return np.array([l0,l1])


def EC1d(np.ndarray[np.intp_t, ndim=1] mask):
    """ Compute Euler characteristic for 1d `mask`

    Given a 1d mask `mask`, compute the 0th intrinsic volume (Euler
    characteristic) of the masked region. The region is broken up into edges /
    vertices, which are included based on whether all voxels in the edge /
    vertex are in the mask or not.

    Parameters
    ----------
    mask : ndarray((i,), np.int)
         Binary mask determining whether or not a voxel is in the mask.

    Returns
    -------
    mu0 : int
        Euler characteristic

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
    cdef:
        np.npy_intp i, m, s0
        double l0 = 0

    s0 = mask.shape[0]
    for i in range(s0):
        m = mask[i]
        if m:
            m = m * (mask[(i+1) % s0] * ((i+1) < s0))
            l0 = l0 - m

    l0 += mask.sum()
    return l0
