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

from test_FIACdata import descriptions, designs
from StringIO import StringIO
import numpy as np
import numpy.testing as nptest
import nose.tools
import sympy

from neuroimaging.modalities.fmri import formula, utils, hrf
from neuroimaging.modalities.fmri.fmristat import hrf as delay

class Protocol(object):

    def __init__(self, design_type):

        self.design_type = design_type
        self._read(StringIO(descriptions[design_type]))

        self.fmristat = np.array([float(x) for x in designs[design_type].strip().split('\t')])
        self.fmristat = self.fmristat.reshape((191, 
                                               self.fmristat.shape[0]/191)).T
        self.nipy = self.design(np.arange(191)*2.5+1.25, 
                                return_float=True).T

    def _read(self, fh):
        """
        Helper function to read protocols as specified in the FIAC
        experiment.
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

        if self.design_type == 'block':
            keep = np.not_equal((np.arange(len(times))) % 6, 0)
        else:
            keep = np.greater(np.arange(len(times)), 0)

        self._begin = np.array(times)[~keep]
        self._times = np.array(times)[keep]
        self._events = np.array(events)[keep]

    def _gettimes(self):
        return self._times
    times = property(_gettimes)

    def _getevents(self):
        return self._events
    events = property(_getevents)

    def _getbegin(self):
        s = utils.events(self._begin, f=hrf.glover_sympy)
        return interpolize(s, 'begin')
    begin = property(_getbegin)

    def effect(self, key, f=hrf.glover_sympy, name=None, **aliases):
        k = np.array([self.events[i] == key for i in 
                      range(self.times.shape[0])])
        s = utils.events(self.times[k], f=f)
        name = name or key
        return interpolize(s, name, **aliases)

    def _getformula(self):
        hrf1 = sympy.Function('hrf0')
        hrf2 = sympy.Function('hrf1')
        if not hasattr(self, "_formula"):
            ff = self.begin
            for e in ['DSt_DSp', 'SSt_DSp', 'DSt_SSp', 'SSt_SSp']:
                for i, f in zip((0,1), (hrf1, 
                                        hrf2)):
                    ff = ff + self.effect(e, name='%s%d' % (e, i), f=f,
                                          hrf0=delay.spectral[0],
                                          hrf1=delay.spectral[1])
            ff = ff + formula.natural_spline(formula.Term('t'), knots=[191/2.+1.25], intercept=True)
            self._formula = ff
        return self._formula
    formula = property(_getformula)

    def design(self, t, return_float=False):
        d = formula.Design(self.formula, return_float=return_float)
        return d(t.view(np.dtype([('t', np.float)])))

def interpolize(s, name, t=np.linspace(0, 500, 5001), **aliases):
    f = formula.Formula([s])
    for key, value in aliases.items():
        f.aliases[key] = value
    d = formula.Design(f, return_float=True)
    l = utils.linear_interp(t, d(t.view(np.dtype([('t', np.float)]))),
                            name=name,
                            bounds_error=False,
                            fill_value=0.)
    return l

def matchcol(col, X):
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    return np.argmax(c), c.max()

def test_agreement():
    for dtype in ['event', 'block']:
        p = Protocol(dtype)
        for i in range(p.nipy.shape[0]):
            _, cmax = matchcol(p.nipy[i], p.fmristat)
            if not p.design(np.arange(191)*2.5+1.25).dtype.names[i].startswith('ns'):
                yield nose.tools.assert_true, np.greater(cmax, 0.999)
