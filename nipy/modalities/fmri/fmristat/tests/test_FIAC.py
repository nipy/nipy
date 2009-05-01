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

from test_FIACdata import descriptions, designs

from StringIO import StringIO
import numpy as np
import numpy.testing as nptest
import nose.tools
import sympy

from nipy.modalities.fmri import formula, utils, hrf
from nipy.modalities.fmri.fmristat import hrf as delay



class Protocol(formula.Formula):

    """
    This class is meant to create the design matrix for
    the FIAC data, as used in the reference above.
    
    It is not a generic way of specifying a design,
    but probably offers some common elements for 
    many designs....
    """

    def __init__(self, fh, design_type, *hrfs):
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

        """
        eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
        eventdict_r = {}
        for key, value in eventdict.items():
            eventdict_r[value] = key

        fh = fh.read().strip().split('\n')

        times = []
        events = []

        for row in fh:
            time, eventtype = map(float, row.split())
            times.append(time)
            events.append(eventdict[eventtype])

        self.design_type = design_type
        if design_type == 'block':
            keep = np.not_equal((np.arange(len(times))) % 6, 0)
        else:
            keep = np.greater(np.arange(len(times)), 0)

        # This first frame was used to model out a potentially
        # 'bad' first frame....

        _begin = np.array(times)[~keep]

        # We have two representations of the formula

        s = formula.vectorize(utils.events(_begin, f=hrf.glover))
        begin = formula.aliased_function('begin', s)

        self.termdict = {}        
        self.termdict['begin'] = begin(hrf.t)
        drift = formula.natural_spline(hrf.t, knots=[191/2.+1.25], intercept=True)
        for i, t in enumerate(drift.terms):
            self.termdict['drift%d' % i] = t
        # After removing the first frame, keep the remaining
        # events and times

        self.times = np.array(times)[keep]
        self.events = np.array(events)[keep]

        # Now, specify the experimental conditions

        for v in eventdict.values():
            for l, h in enumerate(hrfs):
                k = np.array([self.events[i] == v for i in 
                              range(self.times.shape[0])])
                s = formula.vectorize(utils.events(self.times[k], f=h))
                self.termdict['%s%d' % (v,l)] = formula.aliased_function("%s%d" % (v, l), s)(hrf.t)

        formula.Formula.__init__(self, self.termdict.values())

block = Protocol(StringIO(descriptions['block']), 'block', *delay.spectral)
event = Protocol(StringIO(descriptions['event']), 'event', *delay.spectral)

# Now create the design matrices and contrasts
# The 0 indicates that it will be these columns
# convolved with the first HRF

t = formula.make_recarray(np.arange(191)*2.5+1.25, 't')
X = {}
c = {}
fmristat = {}
D = {}
for p in [block, event]:
    contrasts = {}
    contrasts['average'] = (p.termdict['SSt_SSp0'] + p.termdict['SSt_DSp0'] +
                            p.termdict['DSt_SSp0'] + p.termdict['DSt_DSp0']) / 4.
    contrasts['speaker'] = (p.termdict['SSt_DSp0'] - p.termdict['SSt_SSp0'] +
                            p.termdict['DSt_DSp0'] - p.termdict['DSt_SSp0']) * 0.5
    contrasts['sentence'] = (p.termdict['DSt_DSp0'] + p.termdict['DSt_SSp0'] -
                            p.termdict['SSt_DSp0'] - p.termdict['SSt_SSp0']) * 0.5
    contrasts['interaction'] = (p.termdict['SSt_SSp0'] - p.termdict['SSt_DSp0'] -
                                p.termdict['DSt_SSp0'] + p.termdict['DSt_DSp0'])
    X[p.design_type], c[p.design_type] = p.design(t, contrasts=contrasts)
    D[p.design_type] = p.design(t, return_float=False)
    f = np.array([float(x) for x in designs[p.design_type].strip().split('\t')])
    fmristat[p.design_type] = f.reshape((191, f.shape[0]/191)).T

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
            print cmax
            if not dd.dtype.names[i].startswith('ns'):
                yield nose.tools.assert_true, np.greater(cmax, 0.999)

