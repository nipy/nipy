"""
This test ensures that the design matrices of the
FIAC dataset match with 
fMRIstat's, at least on one block and one event
trial.

Taylor, J.E. & Worsley, K.J. (2005). \'Inference for 
    magnitudes and delays of responses in the FIAC data using 
    BRAINSTAT/FMRISTAT\'. Human Brain Mapping, 27,434-441

"""

import numpy as np
from scipy.interpolate import interp1d

from FIACdesigns import descriptions, designs, altdescr

import csv
from StringIO import StringIO
import numpy as np
import nipy.testing as niptest
import sympy

from nipy.modalities.fmri import formula, utils, hrf
from nipy.modalities.fmri.fmristat import hrf as delay

from nipy.fixes.scipy.stats.models.regression import OLSModel

def protocol(fh, design_type, *hrfs):
        """
        Create an object that can evaluate the FIAC.
        Subclass of formula.Formula, but not necessary.

        Parameters:
        -----------

        fh : file handler
            File-like object that reads in the FIAC design,
            i.e. like file('subj1_evt_fonc3.txt')

        design_type : str in ['event', 'block']
            Handles how the 'begin' term is handled.
            For 'block', the first event of each block
            is put in this group. For the 'event', 
            only the first event is put in this group.

            The 'begin' events are convolved with hrf.glover.

        hrfs: symoblic HRFs
            Each event type ('SSt_SSp','SSt_DSp','DSt_SSp','DSt_DSp')
            is convolved with each of these HRFs in order.

	Outputs:
	--------

	f: Formula
	     Formula for constructing design matrices.

	contrasts : dict
	     Dictionary of the contrasts of the experiment.

        """
        eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}

        fh = fh.read().strip().splitlines()

        times = []
        events = []

        for row in fh:
            time, eventtype = map(float, row.split())
            times.append(time)
            events.append(eventdict[eventtype])
        if design_type == 'block':
            keep = np.not_equal((np.arange(len(times))) % 6, 0)
        else:
            keep = np.greater(np.arange(len(times)), 0)

        # This first frame was used to model out a potentially
        # 'bad' first frame....

        _begin = np.array(times)[~keep]

        termdict = {}        
        termdict['begin'] = formula.define('begin', utils.events(_begin, f=hrf.glover))

        drift = formula.natural_spline(hrf.t, knots=[191/2.+1.25], intercept=True)
        for i, t in enumerate(drift.terms):
            termdict['drift%d' % i] = t
        # After removing the first frame, keep the remaining
        # events and times

        times = np.array(times)[keep]
        events = np.array(events)[keep]

        # Now, specify the experimental conditions
	# This creates expressions
	# named SSt_SSp0, SSt_SSp1, etc.
	# with one expression for each (eventtype, hrf) pair

        for v in eventdict.values():
            for l, h in enumerate(hrfs):
                k = np.array([events[i] == v for i in 
                              range(times.shape[0])])
                termdict['%s%d' % (v,l)] = formula.define("%s%d" % (v, l), 
							  utils.events(times[k], f=h))

        f = formula.Formula(termdict.values())

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
	Fcontrasts['overall1'] = formula.Formula(Tcontrasts.values())
	
        return f, Tcontrasts, Fcontrasts

def altprotocol(fh, design_type, t, *hrfs):
        """
        Create an object that can evaluate the FIAC.
        Subclass of formula.Formula, but not necessary.

        Parameters:
        -----------

        fh : file handler
            File-like object that reads in the FIAC design,
            but has a different format (test_FIACdata.altdescr)

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

        fh = csv.reader(fh, delimiter=',')

        d = []

	# d = np.loadtxt(fh)??
        for row in fh:
            time, st, sp = row
	    d.append((time, st, sp))
	d = np.array(d, np.dtype([('eventtime', np.float),
				  ('sentence', 'S3'),
				  ('speaker', 'S3')])).view(np.recarray)

        if design_type == 'block':
            keep = np.not_equal((np.arange(d.eventtime.shape[0])) % 6, 0)
        else:
            keep = np.greater(np.arange(d.eventtime.shape[0]), 0)

        # This first frame was used to model out a potentially
        # 'bad' first frame....

        _begin = d.eventtime[~keep]
	d = d[keep]

        termdict = {}        
        termdict['begin'] = formula.define('begin', utils.events(_begin, f=hrf.glover))
        drift = formula.natural_spline(hrf.t, knots=[191/2.+1.25], intercept=True)
        for i, t in enumerate(drift.terms):
            termdict['drift%d' % i] = t

        # Now, specify the experimental conditions
	# The elements of termdict are DiracDeltas, rather than HRFs

	st = formula.Factor('sentence', ['DSt', 'SSt'])
	sp = formula.Factor('speaker', ['DSp', 'SSp'])
	
	indic = {}
	indic['sentence'] =  st.main_effect
	indic['speaker'] =  sp.main_effect
	indic['interaction'] = st.main_effect * sp.main_effect
	indic['average'] = formula.I

	for key in indic.keys():
            # The matrix signs will be populated with +- 1's
            # d is the recarray having fields ('eventtime', 'sentence', 'speaker')
            signs = indic[key].design(d, return_float=True)

            for l, h in enumerate(hrfs):

		# symb is a sympy expression representing a sum
		# of [h(t-_t) for _t in d.eventtime]
		symb = utils.events(d.eventtime, amplitudes=signs, f=h)

		# the values of termdict will have keys like
		# 'average0', 'speaker1'
		# and values  that are sympy expressions like average0(t), 
		# speaker1(t)
		termdict['%s%d' % (key, l)] = formula.define('%s%d' % (key, l), symb)

        f = formula.Formula(termdict.values())

 	Tcontrasts = {}
	Tcontrasts['average'] = termdict['average0']
	Tcontrasts['speaker'] = termdict['speaker0']
	Tcontrasts['sentence'] = termdict['sentence0']
	Tcontrasts['interaction'] = termdict['interaction0']

	# F tests

	Fcontrasts = {}
	Fcontrasts['overall1'] = formula.Formula(Tcontrasts.values())
	
	nhrf = len(hrfs)
	Fcontrasts['averageF'] = formula.Formula([termdict['average%d' % j] for j in range(nhrf)])
	Fcontrasts['speakerF'] = formula.Formula([termdict['speaker%d' % j] for j in range(nhrf)])
	Fcontrasts['sentenceF'] = formula.Formula([termdict['sentence%d' % j] for j in range(nhrf)])
	Fcontrasts['interactionF'] = formula.Formula([termdict['interaction%d' % j] for j in range(nhrf)])

	Fcontrasts['overall2'] = Fcontrasts['averageF'] + Fcontrasts['speakerF'] + Fcontrasts['sentenceF'] + Fcontrasts['interactionF']

        return f, Tcontrasts, Fcontrasts

block, bTcons, bFcons = protocol(StringIO(descriptions['block']), 'block', *delay.spectral)
event, eTcons, eFcons = protocol(StringIO(descriptions['event']), 'event', *delay.spectral)

# Now create the design matrices and contrasts
# The 0 indicates that it will be these columns
# convolved with the first HRF

t = formula.make_recarray(np.arange(191)*2.5+1.25, 't')
X = {}
c = {}
fmristat = {}
D = {}

for f, cons, design_type in [(block, bTcons, 'block'), (event, eTcons, 'event')]:
    X[design_type], c[design_type] = f.design(t, contrasts=cons)
    D[design_type] = f.design(t, return_float=False)
    fstat = np.array([float(x) for x in designs[design_type].strip().split('\t')])
    fmristat[design_type] = fstat.reshape((191, fstat.shape[0]/191)).T

def test_altprotocol():
    block, bT, bF = protocol(StringIO(descriptions['block']), 'block', *delay.spectral)
    event, eT, eF = protocol(StringIO(descriptions['event']), 'event', *delay.spectral)

    blocka, baT, baF = altprotocol(StringIO(altdescr['block']), 'block', *delay.spectral)
    eventa, eaT, eaF = altprotocol(StringIO(altdescr['event']), 'event', *delay.spectral)

    for c in bT.keys():
        baf = baT[c]
        if not isinstance(baf, formula.Formula):
            baf = formula.Formula([baf])

        bf = bT[c]
        if not isinstance(bf, formula.Formula):
            bf = formula.Formula([bf])

	X = baf.design(t, return_float=True)
	Y = bf.design(t, return_float=True)
	if X.ndim == 1:
            X.shape = (X.shape[0], 1)
	m = OLSModel(X)
	r = m.fit(Y)
	remaining = (r.resid**2).sum() / (Y**2).sum()
	yield niptest.assert_almost_equal, remaining, 0

    for c in bF.keys():
        baf = baF[c]
        if not isinstance(baf, formula.Formula):
            baf = formula.Formula([baf])

        bf = bF[c]
        if not isinstance(bf, formula.Formula):
            bf = formula.Formula([bf])

	X = baf.design(t, return_float=True)
	Y = bf.design(t, return_float=True)
	if X.ndim == 1:
            X.shape = (X.shape[0], 1)
	m = OLSModel(X)
	r = m.fit(Y)
	remaining = (r.resid**2).sum() / (Y**2).sum()
	yield niptest.assert_almost_equal, remaining, 0


def matchcol(col, X):
    """
    Find the column in X with the highest correlation with col.

    Used to find matching columns in fMRIstat's design with
    the design created by Protocol. Not meant as a generic
    helper function.
    """
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    return np.argmax(c), c.max()

def interpolate(name, col, t):
    i = interp1d(t, formula.vectorize(col)(t))
    return formula.aliased_function(name, i)(formula.t)

def test_agreement():
    """
    The test: does Protocol manage to recreate the design of fMRIstat?
    """
    for design_type in ['event', 'block']:
        dd = D[design_type]

        for i in range(X[design_type].shape[1]):
            _, cmax = matchcol(X[design_type][:,i], fmristat[design_type])
            if not dd.dtype.names[i].startswith('ns'):
                yield niptest.assert_true, np.greater(cmax, 0.999)

