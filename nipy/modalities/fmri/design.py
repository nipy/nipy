"""
Convenience functions for specifying a design in the GLM
"""

import numpy as np
from scipy.interpolate import interp1d
from string import join as sjoin

from nipy.algorithms.statistics.utils import combinations

import formula
from utils import events

from hrf import glover, dglover

def event_design(event_spec, t, order=2, hrfs=[glover]):
    """
    Create a design matrix for a GLM analysis based
    on an event specification, evaluating
    it a sequence of time values. Each column
    in the design matrix will be convolved with each HRF in hrfs.

    Parameters:
    -----------

    event_spec : np.recarray
        A recarray having at least a field named 'time' signifying
        the event time, and all other fields will be treated as factors
        in an ANOVA-type model.

    t : np.ndarray
        An array of np.float values at which to evaluate
        the design. Common examples would be the acquisition
        times of an fMRI image.

    order : int
        The highest order interaction to be considered in
        constructing the contrast matrices.

    hrfs : seq
        A sequence of (symbolic) HRF that will be convolved
        with each event. If empty, glover is used.

    Outputs: 
    --------
    
    X : np.ndarray
        The design matrix with X.shape[0] == t.shape[0]. The number
        of columns will depend on the other fields of event_spec.

    contrasts : dict
        Dictionary of contrasts that is expected to be of interest
        from the event specification. For each interaction / effect
        up to a given order will be returned. Also, a contrast
        is generated for each interaction / effect for each HRF
        specified in hrfs.
    
    """

    fields = list(event_spec.dtype.names)
    if 'time' not in fields:
        raise ValueError('expecting a field called "time"')

    fields.pop(fields.index('time'))
    e_factors = [formula.Factor(n, np.unique(event_spec[n])) for n in fields]
    
    e_formula = np.product(e_factors)

    e_contrasts = {}
    for i in range(1, order+1):
        for comb in combinations(zip(fields, e_factors), i):
            names = [c[0] for c in comb]
            fs = [c[1].main_effect for c in comb]
            e_contrasts[sjoin(names, ':')] = np.product(fs).design(event_spec)

    e_contrasts['constant'] = formula.I.design(event_spec)

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
            t_contrasts["%s_%d" % (n, l)] = formula.Formula([ \
                 events(event_spec['time'], amplitudes=c[nn], f=h) for i, nn in enumerate(c.dtype.names)])
    t_formula = formula.Formula(t_terms)
    
    tval = formula.make_recarray(t, ['t'])
    X_t, c_t = t_formula.design(tval, contrasts=t_contrasts)
    return X_t, c_t

def stack2designs(old_X, new_X, old_contrasts={}, new_contrasts={}):
    """
    Add some columns to a design matrix that has contrasts matrices
    already specified, adding some possibly new contrasts as well.

    This basically performs an np.hstack of old_X, new_X
    and makes sure the contrast matrices are dealt with accordingly.
    
    If two contrasts have the same name, an exception is raised.

    Parameters:
    -----------

    old_X : np.ndarray
        A design matrix

    new_X : np.ndarray
        A second design matrix to be stacked with old_X

    old_contrast : dict
        Dictionary of contrasts in the old_X column space

    new_contrasts : dict
        Dictionary of contrasts in the new_X column space
    
    Outputs:
    --------

    X : np.ndarray
        A new design matrix:  np.hstack([old_X, new_X])

    contrasts : dict
        The new contrast matrices reflecting changes to the columns.

    """
    contrasts = {}
    X = np.hstack([old_X, new_X])

    if set(old_contrasts.keys()).intersection(new_contrasts.keys()) != set([]):
        raise ValueError('old and new contrasts must have different names')

    for n, c in old_contrasts.items():
        cm = np.zeros((c.shape[0], X.shape[1]))
        cm[:,:old_X.shape[0]] = c
        contrasts[n] = c

    for n, c in new_contrasts.items():
        cm = np.zeros((c.shape[0], X.shape[1]))
        cm[:,old_X.shape[0]:] = c
        contrasts[n] = c

    return X, contrasts

def stack_designs(*pairs):
    """
    Stack a sequence of design / contrast dictionary
    pairs. Uses multiple calls to stack2designs

    Parameters:
    -----------

    pairs : sequence filled with (np.ndarray, dict) or np.ndarray

    Outputs:
    --------

    X : np.ndarray
        A new design matrix:  np.hstack([old_X, new_X])

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
