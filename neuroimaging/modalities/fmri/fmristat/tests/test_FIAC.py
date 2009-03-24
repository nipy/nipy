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

    """
    This class is meant to create the design matrix for
    the FIAC data, as used in the reference above.
    
    It is not a generic way of specifying a design,
    but probably offers some common elements for 
    many designs....
    """

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



        # This first frame was used to model out a potentially
        # 'bad' first frame....

        _begin = np.array(times)[~keep]
        s = utils.events(_begin, f=hrf.glover_sympy)
        self.first_frame = interpolize(s, 'first_frame')

        # After removing the first frame, keep the remaining
        # events and times

        self.times = np.array(times)[keep]
        self.events = np.array(events)[keep]

    def effect(self, key, hrf=hrf.glover_sympy, name=None, **aliases):
        """
        For a given event type, specified by key, return the
        column of the design matrix that consists of that
        event paired with a symbolic HRF, specified by hrf.

        This could probably be used in a helper class or function
        to create designs for generic experiments...

        Parameters
        ----------

        key : str
            One of ['SSt_SSp', 'SSt_DSp', 'DSt_SSp', 'DSt_DSp'],
            the event types of the FIAC experiment.

        hrf :  sympy.Function
            A sympy symbolic expression paired with the condition type.

        name : str
            A name for the resulting object in the Formula

        aliases : 
            Aliases to be used in evaluating the Formula.

        Examples
        --------

        TODO

        """
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
        """
        Evaluate the design matrix at some specified values of time.

        """
        d = formula.Design(self.formula, return_float=return_float)
        return d(t.view(np.dtype([('t', np.float)])))

def interpolize(s, name, t=np.linspace(0, 500, 5001), **aliases):
    """
    Take a symbolic function of time and create an interpolated
    version. 
    """
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
    """
    Find the column in X with the highest correlation with col.

    Used to find matching columns in fMRIstat's design with
    the design created by Protocol. Not meant as a generic
    helper function.
    """
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    return np.argmax(c), c.max()

def test_agreement():
    """
    The test: does Protocol manage to recreate the design of fMRIstat?
    """
    for dtype in ['event', 'block']:
        p = Protocol(dtype)
        for i in range(p.nipy.shape[0]):
            _, cmax = matchcol(p.nipy[i], p.fmristat)
            if not p.design(np.arange(191)*2.5+1.25).dtype.names[i].startswith('ns'):
                yield nose.tools.assert_true, np.greater(cmax, 0.999)
