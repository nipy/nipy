# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Copy of nibabel io_orientation function from nibabel > 1.2.0

This copy fixes a bug when there are columns of all zero in the affine.

See https://github.com/nipy/nibabel/pull/128

Remove when we depend on nibabel > 1.2.0
"""
from __future__ import absolute_import

import numpy as np
import numpy.linalg as npl


def io_orientation(affine, tol=None):
    ''' Orientation of input axes in terms of output axes for `affine`

    Valid for an affine transformation from ``p`` dimensions to ``q``
    dimensions (``affine.shape == (q + 1, p + 1)``).

    The calculated orientations can be used to transform associated
    arrays to best match the output orientations. If ``p`` > ``q``, then
    some of the output axes should be considered dropped in this
    orientation.

    Parameters
    ----------
    affine : (q+1, p+1) ndarray-like
       Transformation affine from ``p`` inputs to ``q`` outputs.  Usually this
       will be a shape (4,4) matrix, transforming 3 inputs to 3 outputs, but the
       code also handles the more general case
    tol : {None, float}, optional
       threshold below which SVD values of the affine are considered zero. If
       `tol` is None, and ``S`` is an array with singular values for `affine`,
       and ``eps`` is the epsilon value for datatype of ``S``, then `tol` set to
       ``S.max() * eps``.

    Returns
    -------
    orientations : (p, 2) ndarray
       one row per input axis, where the first value in each row is the closest
       corresponding output axis. The second value in each row is 1 if the input
       axis is in the same direction as the corresponding output axis and -1 if
       it is in the opposite direction.  If a row is [np.nan, np.nan], which can
       happen when p > q, then this row should be considered dropped.
    '''
    affine = np.asarray(affine)
    q, p = affine.shape[0]-1, affine.shape[1]-1
    # extract the underlying rotation, zoom, shear matrix
    RZS = affine[:q, :p]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    # Zooms can be zero, in which case all elements in the column are zero, and
    # we can leave them as they are
    zooms[zooms == 0] = 1
    RS = RZS / zooms
    # Transform below is polar decomposition, returning the closest
    # shearless matrix R to RS
    P, S, Qs = npl.svd(RS)
    # Threshold the singular values to determine the rank.
    if tol is None:
        tol = S.max() * np.finfo(S.dtype).eps
    keep = (S > tol)
    R = np.dot(P[:, keep], Qs[keep])
    # the matrix R is such that np.dot(R,R.T) is projection onto the
    # columns of P[:,keep] and np.dot(R.T,R) is projection onto the rows
    # of Qs[keep].  R (== np.dot(R, np.eye(p))) gives rotation of the
    # unit input vectors to output coordinates.  Therefore, the row
    # index of abs max R[:,N], is the output axis changing most as input
    # axis N changes.  In case there are ties, we choose the axes
    # iteratively, removing used axes from consideration as we go
    ornt = np.ones((p, 2), dtype=np.int8) * np.nan
    for in_ax in range(p):
        col = R[:, in_ax]
        if not np.alltrue(np.equal(col, 0)):
            out_ax = np.argmax(np.abs(col))
            ornt[in_ax, 0] = out_ax
            assert col[out_ax] != 0
            if col[out_ax] < 0:
                ornt[in_ax, 1] = -1
            else:
                ornt[in_ax, 1] = 1
            # remove the identified axis from further consideration, by
            # zeroing out the corresponding row in R
            R[out_ax, :] = 0
    return ornt
