# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This test ensures that the design matrices of the FIAC dataset match
with fMRIstat's, at least on one block and one event trial.

Taylor, J.E. & Worsley, K.J. (2005). \'Inference for 
    magnitudes and delays of responses in the FIAC data using 
    BRAINSTAT/FMRISTAT\'. Human Brain Mapping, 27,434-441
"""

import numpy as np

from ... import utils, hrf, design
from .. import hrf as delay

from nipy.algorithms.statistics.models.regression import OLSModel
from nipy.algorithms.statistics.formula import formulae

from nipy.utils.compat3 import to_str

# testing imports
from nipy.testing import (dec, assert_true, assert_almost_equal)

# Local imports
from .FIACdesigns import (descriptions, fmristat, altdescr,
                          N_ROWS, time_vector)

t = formulae.make_recarray(time_vector, 't')


def protocol(recarr, design_type, *hrfs):
    """ Create an object that can evaluate the FIAC

    Subclass of formulae.Formula, but not necessary.

    Parameters
    ----------
    recarr : (N,) structured array
       with fields 'time' and 'event'
    design_type : str
       one of ['event', 'block'].  Handles how the 'begin' term is
       handled.  For 'block', the first event of each block is put in
       this group. For the 'event', only the first event is put in this
       group. The 'begin' events are convolved with hrf.glover.
    hrfs: symoblic HRFs
       Each event type ('SSt_SSp','SSt_DSp','DSt_SSp','DSt_DSp') is
       convolved with each of these HRFs in order.

    Returns
    -------
    f: Formula
       Formula for constructing design matrices.
    contrasts : dict
       Dictionary of the contrasts of the experiment.
    """
    event_types = np.unique(recarr['event'])
    N = recarr.size
    if design_type == 'block':
        keep = np.not_equal((np.arange(N)) % 6, 0)
    else:
        keep = np.greater(np.arange(N), 0)
    # This first frame was used to model out a potentially
    # 'bad' first frame....
    _begin = recarr['time'][~keep]

    termdict = {}
    termdict['begin'] = utils.define('begin', utils.events(_begin, f=hrf.glover))
    drift = formulae.natural_spline(utils.T,
                                   knots=[N_ROWS/2.+1.25],
                                   intercept=True)
    for i, t in enumerate(drift.terms):
        termdict['drift%d' % i] = t
    # After removing the first frame, keep the remaining
    # events and times
    times = recarr['time'][keep]
    events = recarr['event'][keep]

    # Now, specify the experimental conditions.  This creates expressions named
    # SSt_SSp0, SSt_SSp1, etc.  with one expression for each (eventtype, hrf)
    # pair
    for v in event_types:
        k = np.array([events[i] == v for i in range(times.shape[0])])
        for l, h in enumerate(hrfs):
            # Make sure event type is a string (not byte string)
            term_name = '%s%d' % (to_str(v), l)
            termdict[term_name] = utils.define(term_name,
                                               utils.events(times[k], f=h))
    f = formulae.Formula(termdict.values())
    Tcontrasts = {}
    Tcontrasts['average'] = (termdict['SSt_SSp0'] + termdict['SSt_DSp0'] +
                             termdict['DSt_SSp0'] + termdict['DSt_DSp0']) / 4.
    Tcontrasts['speaker'] = (termdict['SSt_DSp0'] - termdict['SSt_SSp0'] +
                             termdict['DSt_DSp0'] - termdict['DSt_SSp0']) * 0.5
    Tcontrasts['sentence'] = (termdict['DSt_DSp0'] + termdict['DSt_SSp0'] -
                              termdict['SSt_DSp0'] - termdict['SSt_SSp0']) * 0.5
    Tcontrasts['interaction'] = (termdict['SSt_SSp0'] - termdict['SSt_DSp0'] -
                                 termdict['DSt_SSp0'] + termdict['DSt_DSp0'])
    # Ftest
    Fcontrasts = {}
    Fcontrasts['overall1'] = formulae.Formula(Tcontrasts.values())

    return f, Tcontrasts, Fcontrasts


def altprotocol(d, design_type, *hrfs):
    """ Create an object that can evaluate the FIAC.

    Subclass of formulae.Formula, but not necessary.

    Parameters
    ----------
    d : np.recarray
       recarray defining design in terms of time, sentence speaker

    design_type : str in ['event', 'block']
        Handles how the 'begin' term is handled.
        For 'block', the first event of each block
        is put in this group. For the 'event', 
        only the first event is put in this group.

        The 'begin' events are convolved with hrf.glover.

    hrfs: symoblic HRFs
        Each event type ('SSt_SSp','SSt_DSp','DSt_SSp','DSt_DSp')
        is convolved with each of these HRFs in order.

    """
    if design_type == 'block':
        keep = np.not_equal((np.arange(d.time.shape[0])) % 6, 0)
    else:
        keep = np.greater(np.arange(d.time.shape[0]), 0)

    # This first frame was used to model out a potentially
    # 'bad' first frame....

    _begin = d.time[~keep]
    d = d[keep]

    termdict = {}
    termdict['begin'] = utils.define('begin', utils.events(_begin, f=hrf.glover))
    drift = formulae.natural_spline(utils.T,
                                   knots=[N_ROWS/2.+1.25],
                                   intercept=True)
    for i, t in enumerate(drift.terms):
        termdict['drift%d' % i] = t

    # Now, specify the experimental conditions
    # The elements of termdict are DiracDeltas, rather than HRFs

    st = formulae.Factor('sentence', ['DSt', 'SSt'])
    sp = formulae.Factor('speaker', ['DSp', 'SSp'])

    indic = {}
    indic['sentence'] =  st.main_effect
    indic['speaker'] =  sp.main_effect
    indic['interaction'] = st.main_effect * sp.main_effect
    indic['average'] = formulae.I

    for key in indic.keys():
        # The matrix signs will be populated with +- 1's
        # d is the recarray having fields ('time', 'sentence', 'speaker')
        signs = indic[key].design(d, return_float=True)

        for l, h in enumerate(hrfs):

            # symb is a sympy expression representing a sum
            # of [h(t-_t) for _t in d.time]
            symb = utils.events(d.time, amplitudes=signs, f=h)

            # the values of termdict will have keys like
            # 'average0', 'speaker1'
            # and values  that are sympy expressions like average0(t), 
            # speaker1(t)
            termdict['%s%d' % (key, l)] = utils.define("%s%d" % (key, l), symb)

    f = formulae.Formula(termdict.values())

    Tcontrasts = {}
    Tcontrasts['average'] = termdict['average0']
    Tcontrasts['speaker'] = termdict['speaker0']
    Tcontrasts['sentence'] = termdict['sentence0']
    Tcontrasts['interaction'] = termdict['interaction0']

    # F tests

    Fcontrasts = {}
    Fcontrasts['overall1'] = formulae.Formula(Tcontrasts.values())

    nhrf = len(hrfs)
    Fcontrasts['averageF'] = formulae.Formula([termdict['average%d' % j] for j in range(nhrf)])
    Fcontrasts['speakerF'] = formulae.Formula([termdict['speaker%d' % j] for j in range(nhrf)])
    Fcontrasts['sentenceF'] = formulae.Formula([termdict['sentence%d' % j] for j in range(nhrf)])
    Fcontrasts['interactionF'] = formulae.Formula([termdict['interaction%d' % j] for j in range(nhrf)])

    Fcontrasts['overall2'] = Fcontrasts['averageF'] + Fcontrasts['speakerF'] + Fcontrasts['sentenceF'] + Fcontrasts['interactionF']

    return f, Tcontrasts, Fcontrasts


def create_protocols():
    # block and event protocols
    block, bTcons, bFcons = protocol(descriptions['block'], 'block', *delay.spectral)
    event, eTcons, eFcons = protocol(descriptions['event'], 'event', *delay.spectral)

    # Now create the design matrices and contrasts
    # The 0 indicates that it will be these columns
    # convolved with the first HRF
    X = {}
    c = {}
    D = {}
    for f, cons, design_type in [(block, bTcons, 'block'), (event, eTcons, 'event')]:
        X[design_type], c[design_type] = f.design(t, contrasts=cons)
        D[design_type] = f.design(t, return_float=False)
    return X, c, D


def test_altprotocol():
    block, bT, bF = protocol(descriptions['block'], 'block', *delay.spectral)
    event, eT, eF = protocol(descriptions['event'], 'event', *delay.spectral)

    blocka, baT, baF = altprotocol(altdescr['block'], 'block', *delay.spectral)
    eventa, eaT, eaF = altprotocol(altdescr['event'], 'event', *delay.spectral)

    for c in bT.keys():
        baf = baT[c]
        if not isinstance(baf, formulae.Formula):
            baf = formulae.Formula([baf])

        bf = bT[c]
        if not isinstance(bf, formulae.Formula):
            bf = formulae.Formula([bf])

    X = baf.design(t, return_float=True)
    Y = bf.design(t, return_float=True)
    if X.ndim == 1:
        X.shape = (X.shape[0], 1)
    m = OLSModel(X)
    r = m.fit(Y)
    remaining = (r.resid**2).sum() / (Y**2).sum()
    assert_almost_equal(remaining, 0)

    for c in bF.keys():
        baf = baF[c]
        if not isinstance(baf, formulae.Formula):
            baf = formulae.Formula([baf])

        bf = bF[c]
        if not isinstance(bf, formulae.Formula):
            bf = formulae.Formula([bf])

    X = baf.design(t, return_float=True)
    Y = bf.design(t, return_float=True)
    if X.ndim == 1:
        X.shape = (X.shape[0], 1)
    m = OLSModel(X)
    r = m.fit(Y)
    remaining = (r.resid**2).sum() / (Y**2).sum()
    assert_almost_equal(remaining, 0)


def matchcol(col, X):
    """ Find the row in X with the highest correlation with 1D col.

    Used to find matching columns in fMRIstat's design with the design
    created by Protocol. Not meant as a generic helper function.
    """
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    ind = np.argmax(np.abs(c))
    return ind, c[ind]


def test_agreement():
    # The test: does Protocol manage to recreate the design of fMRIstat?
    X, c, D = create_protocols()
    for design_type in ['event', 'block']:
        dd = D[design_type]
        for i in range(X[design_type].shape[1]):
            _, cmax = matchcol(X[design_type][:,i], fmristat[design_type])
            if not dd.dtype.names[i].startswith('ns'):
                assert_true(np.greater(np.abs(cmax), 0.999))


@dec.slow
def test_event_design():
    block = altdescr['block']
    event = altdescr['event']
    t = time_vector

    bkeep = np.not_equal((np.arange(block.time.shape[0])) % 6, 0)
    ekeep = np.greater(np.arange(event.time.shape[0]), 0)

    # Even though there is a FIAC block experiment
    # the design is represented as an event design
    # with the same event repeated several times in a row...

    Xblock, cblock = design.event_design(block[bkeep], t, hrfs=delay.spectral)
    Xevent, cevent = design.event_design(event[ekeep], t, hrfs=delay.spectral)
