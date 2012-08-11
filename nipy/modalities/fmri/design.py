# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convenience functions for specifying a design in the GLM
"""

import numpy as np

from nipy.algorithms.statistics.utils import combinations
from nipy.algorithms.statistics.formula import formulae
from nipy.algorithms.statistics.formula.formulae import (
    Formula, Factor, Term, make_recarray)

from .utils import events, fourier_basis as fourier_basis_sym

from .hrf import glover

def fourier_basis(t, freq):
    """
    Create a design matrix with columns given by the Fourier
    basis with a given set of frequencies.

    Parameters
    ----------
    t : np.ndarray
        An array of np.float values at which to evaluate
        the design. Common examples would be the acquisition
        times of an fMRI image.
    freq : sequence of float
        Frequencies for the terms in the Fourier basis.

    Returns
    -------
    X : np.ndarray

    Examples
    --------
    >>> t = np.linspace(0,50,101)
    >>> drift = fourier_basis(t, np.array([4,6,8]))
    >>> drift.shape
    (101, 6)
    """
    tval = make_recarray(t, ['t'])
    f = fourier_basis_sym(freq)
    return f.design(tval, return_float=True)


def natural_spline(tvals, knots=None, order=3, intercept=True):
    """
    Create a design matrix with columns given by a natural spline of a given
    order and a specified set of knots.

    Parameters
    ----------
    tvals : np.array
        Time values
    knots : None or sequence, optional
       Sequence of float.  Default None (same as empty list)
    order : int, optional
       Order of the spline. Defaults to a cubic (==3)
    intercept : bool, optional
       If True, include a constant function in the natural
       spline. Default is False

    Returns
    -------
    X : np.ndarray

    Examples
    --------
    >>> tvals = np.linspace(0,50,101)
    >>> drift = natural_spline(tvals, knots=[10,20,30,40])
    >>> drift.shape
    (101, 8)
    """
    tvals = make_recarray(tvals, ['t'])
    t = Term('t')
    f = formulae.natural_spline(t, knots=knots, order=order, intercept=intercept)
    return f.design(tvals, return_float=True)


def event_design(event_spec, t, order=2, hrfs=[glover]):
    """
    Create a design matrix for a GLM analysis based
    on an event specification, evaluating
    it a sequence of time values. Each column
    in the design matrix will be convolved with each HRF in hrfs.

    Parameters
    ----------
    event_spec : np.recarray
       A recarray having at least a field named 'time' signifying the
       event time, and all other fields will be treated as factors in an
       ANOVA-type model.
    t : np.ndarray
       An array of np.float values at which to evaluate the
       design. Common examples would be the acquisition times of an fMRI
       image.
    order : int
       The highest order interaction to be considered in constructing
       the contrast matrices.
    hrfs : seq
       A sequence of (symbolic) HRF that will be convolved with each
       event. If empty, glover is used.

    Returns 
    -------
    X : np.ndarray
       The design matrix with X.shape[0] == t.shape[0]. The number of
       columns will depend on the other fields of event_spec.
    contrasts : dict
       Dictionary of contrasts that is expected to be of interest from
       the event specification. For each interaction / effect up to a
       given order will be returned. Also, a contrast is generated for
       each interaction / effect for each HRF specified in hrfs.
    """
    fields = list(event_spec.dtype.names)
    if 'time' not in fields:
        raise ValueError('expecting a field called "time"')
    fields.pop(fields.index('time'))
    e_factors = [Factor(n, np.unique(event_spec[n])) for n in fields]
    e_formula = np.product(e_factors)
    e_contrasts = {}
    if len(e_factors) > 1:
        for i in range(1, order+1):
            for comb in combinations(zip(fields, e_factors), i):
                names = [c[0] for c in comb]
                fs = [c[1].main_effect for c in comb]
                e_contrasts[":".join(names)] = np.product(fs).design(event_spec)

    e_contrasts['constant'] = formulae.I.design(event_spec)

    # Design and contrasts in event space
    # TODO: make it so I don't have to call design twice here
    # to get both the contrasts and the e_X matrix as a recarray

    e_X = e_formula.design(event_spec)
    e_dtype = e_formula.dtype

    # Now construct the design in time space

    t_terms = []
    t_contrasts = {}
    for l, h in enumerate(hrfs):
        t_terms += [events(event_spec['time'], \
            amplitudes=e_X[n], f=h) for i, n in enumerate(e_dtype.names)]
        for n, c in e_contrasts.items():
            t_contrasts["%s_%d" % (n, l)] = Formula([ \
                 events(event_spec['time'], amplitudes=c[nn], f=h)
                 for i, nn in enumerate(c.dtype.names)])
    t_formula = Formula(t_terms)
    
    tval = make_recarray(t, ['t'])
    X_t, c_t = t_formula.design(tval, contrasts=t_contrasts)
    return X_t, c_t


def stack2designs(old_X, new_X, old_contrasts={}, new_contrasts={}):
    """
    Add some columns to a design matrix that has contrasts matrices
    already specified, adding some possibly new contrasts as well.

    This basically performs an np.hstack of old_X, new_X
    and makes sure the contrast matrices are dealt with accordingly.
    
    If two contrasts have the same name, an exception is raised.

    Parameters
    ----------
    old_X : np.ndarray
       A design matrix
    new_X : np.ndarray
       A second design matrix to be stacked with old_X
    old_contrast : dict
       Dictionary of contrasts in the old_X column space
    new_contrasts : dict
       Dictionary of contrasts in the new_X column space
    
    Returns
    -------
    X : np.ndarray
       A new design matrix:  np.hstack([old_X, new_X])
    contrasts : dict
       The new contrast matrices reflecting changes to the columns.
    """
    contrasts = {}

    if old_X.ndim == 1:
        old_X = old_X.reshape((old_X.shape[0], 1))
    if new_X.ndim == 1:
        new_X = new_X.reshape((new_X.shape[0], 1))

    X = np.hstack([old_X, new_X])

    if set(old_contrasts.keys()).intersection(new_contrasts.keys()) != set([]):
        raise ValueError('old and new contrasts must have different names')

    for n, c in old_contrasts.items():
        if c.ndim > 1:
            cm = np.zeros((c.shape[0], X.shape[1]))
            cm[:,:old_X.shape[1]] = c
        else:
            cm = np.zeros(X.shape[1])
            cm[:old_X.shape[1]] = c
        contrasts[n] = cm

    for n, c in new_contrasts.items():
        if c.ndim > 1:
            cm = np.zeros((c.shape[0], X.shape[1]))
            cm[:,old_X.shape[1]:] = c
        else:
            cm = np.zeros(X.shape[1])
            cm[old_X.shape[1]:] = c
        contrasts[n] = cm

    return X, contrasts


def stack_contrasts(contrasts, name, keys):
    """
    Create a new F-contrast matrix called 'name'
    based on a sequence of keys. The contrast
    is added to contrasts, in-place.

    Parameters
    ----------
    contrasts : dict
       Dictionary of contrast matrices
    name : str
       Name of new contrast. Should not already be a key of contrasts.
    keys : sequence of str
       Keys of contrasts that are to be stacked.

    Returns
    -------
    None
    """
    if name in contrasts.keys():
        raise ValueError('contrast "%s" already exists' % name)

    contrasts[name] = np.vstack([contrasts[k] for k in keys])


def stack_designs(*pairs):
    """
    Stack a sequence of design / contrast dictionary
    pairs. Uses multiple calls to stack2designs

    Parameters
    ----------
    pairs : sequence filled with (np.ndarray, dict) or np.ndarray

    Returns
    -------
    X : np.ndarray
       new design matrix:  np.hstack([old_X, new_X])
    contrasts : dict
       The new contrast matrices reflecting changes to the columns.
    """
    p = pairs[0]
    if len(p) == 1:
        X = p[0]; contrasts={}
    else:
        X, contrasts = p

    for q in pairs[1:]:
        if len(q) == 1:
            new_X = q[0]; new_con = {}
        else:
            new_X, new_con = q
        X, contrasts = stack2designs(X, new_X, contrasts, new_con)
    return X, contrasts
