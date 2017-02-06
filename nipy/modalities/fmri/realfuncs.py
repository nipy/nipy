""" Helper functions for constructing design regressors
"""
from __future__ import division

import numpy as np


def dct_ii_basis(volume_times, order=None, normcols=False):
    """ DCT II basis up to order `order`

    See: https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    By default, basis not normalized to length 1, and therefore, basis is not
    orthogonal.  Normalize basis with `normcols` keyword argument.

    Parameters
    ----------
    volume_times : array-like
        Times of acquisition of each volume.  Must be regular and continuous
        otherwise we raise an error.
    order : None or int, optional
        Order of DCT-II basis.  If None, return full basis set.
    normcols : bool, optional
        If True, normalize columns to length 1, so return orthogonal
        `dct_basis`.

    Returns
    -------
    dct_basis : array
        Shape ``(len(volume_times), order)`` array with DCT-II basis up to
        order `order`.

    Raises
    ------
    ValueError
        If difference between successive `volume_times` values is not constant
        over the 1D array.
    """
    N = len(volume_times)
    if order is None:
        order = N
    if not np.allclose(np.diff(np.diff(volume_times)), 0):
        raise ValueError("DCT basis assumes continuous regular sampling")
    n = np.arange(N)
    cycle = np.pi * (n + 0.5) / N
    dct_basis = np.zeros((N, order))
    for k in range(0, order):
        dct_basis[:, k] = np.cos(cycle * k)
    if normcols:  # Set column lengths to 1
        lengths = np.ones(order) * np.sqrt(N / 2.)
        lengths[0:1] = np.sqrt(N)  # Allow order=0
        dct_basis /= lengths
    return dct_basis


def dct_ii_cut_basis(volume_times, cut_period):
    """DCT-II regressors with periods >= `cut_period`

    See: http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    Parameters
    ----------
    volume_times : array-like
        Times of acquisition of each volume.  Must be regular and continuous
        otherwise we raise an error.
    cut_period: float
        Cut period (wavelength) of the low-pass filter (in time units).

    Returns
    -------
    cdrift: array shape (n_scans, n_drifts)
        DCT-II drifts plus a constant regressor in the final column.  Constant
        regressor always present, regardless of `cut_period`.
    """
    N = len(volume_times)
    hfcut = 1./ cut_period
    dt = volume_times[1] - volume_times[0]
    # Such that hfcut = 1/(2*dt) yields N
    order = int(np.floor(2 * N * hfcut * dt))
    # Always return constant column
    if order == 0:
        return np.ones((N, 1))
    basis = np.ones((N, order))
    basis[:, :-1] = dct_ii_basis(volume_times, order, normcols=True)[:, 1:]
    return basis
