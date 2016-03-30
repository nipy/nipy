# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convenience functions for specifying a design in the GLM
"""
from __future__ import absolute_import

import itertools

import numpy as np

from nipy.algorithms.statistics.utils import combinations
from nipy.algorithms.statistics.formula import formulae
from nipy.algorithms.statistics.formula.formulae import (
    Formula, Factor, Term, make_recarray)

from .utils import (events, blocks, fourier_basis as fourier_basis_sym,
                    convolve_functions, T)

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


def _build_formula_contrasts(spec, fields, order):
    """ Build formula and contrast in event / block space

    Parameters
    ----------
    spec : stuctured array
        Structured array containing at least fields listed in `fields`.
    fields : sequence of str
        Sequence of field names containing names of factors.
    order : int
        Maximum order of interactions between main effects.

    Returns
    -------
    e_factors : :class:`Formula` instance
        Formula for factors given by `fields`
    e_contrasts : dict
        Dictionary containing contrasts of main effects and interactions
        between factors.
    """
    e_factors = [Factor(n, np.unique(spec[n])) for n in fields]
    e_formula = np.product(e_factors)
    e_contrasts = {}
    # Add contrasts for factors and factor interactions
    max_order = min(len(e_factors), order)
    for i in range(1, max_order + 1):
        for comb in combinations(zip(fields, e_factors), i):
            names = [c[0] for c in comb]
            # Collect factors where there is more than one level
            fs = [fc.main_effect for fn, fc in comb if len(fc.levels) > 1]
            if len(fs) > 0:
                e_contrast = np.product(fs).design(spec)
                e_contrasts[":".join(names)] = e_contrast
    e_contrasts['constant'] = formulae.I.design(spec)
    return e_formula, e_contrasts


def event_design(event_spec, t, order=2, hrfs=(glover,),
                 level_contrasts=False):
    """ Create design matrix at times `t` for event specification `event_spec`

    Create a design matrix for linear model based on an event specification
    `event_spec`, evaluating the design rows at a sequence of time values `t`.
    Each column in the design matrix will be convolved with each HRF in `hrfs`.

    Parameters
    ----------
    event_spec : np.recarray
       A recarray having at least a field named 'time' signifying the event
       time, and all other fields will be treated as factors in an ANOVA-type
       model.  If there is no field other than time, add a single-level
       placeholder event type ``_event_``.
    t : np.ndarray
       An array of np.float values at which to evaluate the design. Common
       examples would be the acquisition times of an fMRI image.
    order : int, optional
       The highest order interaction to be considered in constructing
       the contrast matrices.
    hrfs : sequence, optional
       A sequence of (symbolic) HRFs that will be convolved with each event.
       Default is ``(glover,)``.
    level_contrasts : bool, optional
       If True, generate contrasts for each individual level of each factor.

    Returns
    -------
    X : np.ndarray
       The design matrix with ``X.shape[0] == t.shape[0]``. The number of
       columns will depend on the other fields of `event_spec`.
    contrasts : dict
       Dictionary of contrasts that is expected to be of interest from the
       event specification. Each interaction / effect up to a given order will
       be returned. Also, a contrast is generated for each interaction / effect
       for each HRF specified in `hrfs`.
    """
    fields = list(event_spec.dtype.names)
    if 'time' not in fields:
        raise ValueError('expecting a field called "time"')
    fields.pop(fields.index('time'))
    if len(fields) == 0:  # No factors specified, make generic event
        event_spec = make_recarray(zip(event_spec['time'],
                                       itertools.cycle([1])),
                                   ('time', '_event_'))
        fields = ['_event_']
    e_formula, e_contrasts = _build_formula_contrasts(
        event_spec, fields, order)
    # Design and contrasts in block space
    # TODO: make it so I don't have to call design twice here
    # to get both the contrasts and the e_X matrix as a recarray
    e_X = e_formula.design(event_spec)
    e_dtype = e_formula.dtype

    # Now construct the design in time space
    t_terms = []
    t_contrasts = {}
    for l, h in enumerate(hrfs):
        for n in e_dtype.names:
            term = events(event_spec['time'], amplitudes=e_X[n], f=h)
            t_terms  += [term]
            if level_contrasts:
                t_contrasts['%s_%d' % (n, l)] = Formula([term])
        for n, c in e_contrasts.items():
            t_contrasts["%s_%d" % (n, l)] = Formula([ \
                 events(event_spec['time'], amplitudes=c[nn], f=h)
                 for i, nn in enumerate(c.dtype.names)])
    t_formula = Formula(t_terms)

    tval = make_recarray(t, ['t'])
    X_t, c_t = t_formula.design(tval, contrasts=t_contrasts)
    return X_t, c_t


def block_design(block_spec, t, order=2, hrfs=(glover,),
                 convolution_padding=5.,
                 convolution_dt=0.02,
                 hrf_interval=(0.,30.),
                 level_contrasts=False):
    """ Create design matrix at times `t` for blocks specification `block_spec`

    Create design matrix for linear model from a block specification
    `block_spec`,  evaluating design rows at a sequence of time values `t`.
    Each column in the design matrix will be convolved with each HRF in `hrfs`.

    Parameters
    ----------
    block_spec : np.recarray
       A recarray having at least a field named 'start' and a field named 'end'
       signifying the block onset and offset times. All other fields will be
       treated as factors in an ANOVA-type model.  If there is no field other
       than 'start' and 'end', add a single-level placeholder block type
       ``_block_``.
    t : np.ndarray
       An array of np.float values at which to evaluate the design. Common
       examples would be the acquisition times of an fMRI image.
    order : int, optional
       The highest order interaction to be considered in constructing the
       contrast matrices.
    hrfs : sequence, optional
       A sequence of (symbolic) HRFs that will be convolved with each block.
       Default is ``(glover,)``.
    convolution_padding : float, optional
       A padding for the convolution with the HRF. The intervals
       used for the convolution are the smallest 'start' minus this
       padding to the largest 'end' plus this padding.
    convolution_dt : float, optional
       Time step for high-resolution time course for use in convolving the
       blocks with each HRF.
    hrf_interval: length 2 sequence of floats, optional
       Interval over which the HRF is assumed supported, used in the
       convolution.
    level_contrasts : bool, optional
       If true, generate contrasts for each individual level
       of each factor.

    Returns
    -------
    X : np.ndarray
       The design matrix with ``X.shape[0] == t.shape[0]``. The number of
       columns will depend on the other fields of `block_spec`.
    contrasts : dict
       Dictionary of contrasts that are expected to be of interest from the
       block specification. Each interaction / effect up to a given order will
       be returned. Also, a contrast is generated for each interaction / effect
       for each HRF specified in `hrfs`.
    """
    fields = list(block_spec.dtype.names)
    if 'start' not in fields or 'end' not in fields:
        raise ValueError('expecting fields called "start" and "end"')
    fields.pop(fields.index('start'))
    fields.pop(fields.index('end'))
    if len(fields) == 0:  # No factors specified, make generic block
        block_spec = make_recarray(zip(block_spec['start'],
                                       block_spec['end'],
                                       itertools.cycle([1])),
                                   ('start', 'end', '_block_'))
        fields = ['_block_']
    e_formula, e_contrasts = _build_formula_contrasts(
        block_spec, fields, order)
    # Design and contrasts in block space
    # TODO: make it so I don't have to call design twice here
    # to get both the contrasts and the e_X matrix as a recarray
    e_X = e_formula.design(block_spec)
    e_dtype = e_formula.dtype

    # Now construct the design in time space
    block_times = np.array([(s,e) for s, e in zip(block_spec['start'], 
                                                  block_spec['end'])])
    convolution_interval = (block_times.min() - convolution_padding, 
                            block_times.max() + convolution_padding)

    t_terms = []
    t_contrasts = {}
    for l, h in enumerate(hrfs):
        for n in e_dtype.names:
            B = blocks(block_times, amplitudes=e_X[n])
            term = convolve_functions(B, h(T), 
                                      convolution_interval,
                                      hrf_interval,
                                      convolution_dt)
            t_terms += [term]
            if level_contrasts:
                t_contrasts['%s_%d' % (n, l)] = Formula([term])
        for n, c in e_contrasts.items():
            F = []
            for i, nn in enumerate(c.dtype.names):
                B = blocks(block_times, amplitudes=c[nn])
                F.append(convolve_functions(B, h(T),
                                            convolution_interval,
                                            hrf_interval,
                                            convolution_dt))
            t_contrasts["%s_%d" % (n, l)] = Formula(F)
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
    old_X = np.asarray(old_X)
    new_X = np.asarray(new_X)
    if old_X.size == 0:
        return new_X, new_contrasts
    if new_X.size == 0:
        return old_X, old_contrasts

    if old_X.ndim == 1:
        old_X = old_X[:, None]
    if new_X.ndim == 1:
        new_X = new_X[:, None]

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
    """ Stack a sequence of design / contrast dictionary pairs

    Uses multiple calls to :func:`stack2designs`

    Parameters
    ----------
    \*pairs : sequence
        Elements of either (np.ndarray, dict) or (np.ndarray,) or np.ndarray

    Returns
    -------
    X : np.ndarray
       new design matrix:  np.hstack([old_X, new_X])
    contrasts : dict
       The new contrast matrices reflecting changes to the columns.
    """
    X = []
    contrasts = {}
    for p in pairs:
        if isinstance(p, np.ndarray):
            new_X = p; new_con = {}
        elif len(p) == 1:  # Length one sequence
            new_X = p[0]; new_con = {}
        else:  # Length 2 sequence
            new_X, new_con = p
        X, contrasts = stack2designs(X, new_X, contrasts, new_con)
    return X, contrasts
