import numpy as np
import sympy as sym

from utils import events
from formula import Term, Formula, Design
import hrf

aliases = {'glover': hrf.glover, 'dglover':hrf.dglover}

def _contrast(cont, terms):
    args = cont.args
    coeffs = []
    indices = []
    for a in args: 
        w = [a.as_coefficient(t) for t in terms]
        wf = filter(None, w)
        if len(wf):
            coeffs += [wf[0]]
            indices += [w.index(wf[0])]
    return coeffs, indices


class LinearModel:
    
    def __init__(self, hrf): 
        self.hrf = hrf 
        self._conditions = {}
        self._extra_regressors = {}
        
    def condition(self, term, onsets, amplitudes=None, durations=None):
        """
        specify a new condition from the 
        NOTE:
        add condition-specific hrf ?
        """
        hrf_sym = [sym.Function(h) for h in self.hrf]
        self._conditions[term] = Formula([events(onsets, amplitudes=amplitudes, f=f) for f in hrf_sym])
        # or blocks(...)

    def polynomial_drift(self,order):
        """
        create the drift terms as polynomials
        """
        t = Term('t')
        monom=t
        aux = [monom**(i+1) for i in range(order)]

    def cosine_drift(self,hfcut,duration):
        """
        create the dirft terms as cosine functiona of time
        """
        t = Term('t')
        # create the DCT basis
        
    def drift(self, order, expression=None):
        """
        Create the drift terms
        seems to be consistent only for polynomial drifts
        -> rename 'polynomial drift ?'
        """
        t = Term('t')
        if isinstance(expression, sym.function.FunctionClass): 
            monom = expression(t)
        else: 
            monom = t
        aux = [monom**(i+1) for i in range(order)]
        self._extra_regressors[Term('drift')] = Formula(aux)

    def regressor(self): 
        """
        Not implemented yet. 
        """
        return
    
    def conditions(self):
        return [term for term in self._conditions]
    
    def extra_regressors(self):
        return [term for term in self._extra_regressors] + [Term('baseline')]

    def terms(self): 
        return self.conditions()+ self.extra_regressors() 

    def formula(self): 
        f = Formula([])
        for term in self._conditions:
            f += self._conditions[term]
        for term in self._extra_regressors:
            f += self._extra_regressors[term]
        # aliases
        for h in self.hrf: 
            f.aliases[h] = aliases[h]
        return f

    def design_matrix(self, timestamps): 
        """
        X = self.design_matrix(timestamps)

        timestamps is a 1d array. 
        """
        tval = np.asarray(timestamps, dtype=np.float).view(np.dtype([('t', np.float)]))
        aux = Design(self.formula(), return_float=True)(tval)
        X = np.zeros([aux.shape[0], aux.shape[1]+1])
        X[:,0:-1] = aux
        X[:,-1] = 1.
        return X

    def contrast(self, cont):
        """
        C = self.contrast(cont)

        cont is a linear combination of terms (a symbolic expression).
        Return a matrix pxq where p is the number of columns of the
        design matrix and q is the 'dimensionality' of the contrast. 
        """
        nregressors = len(self.formula().terms)
        nhrfs = len(self.hrf)
        mat = np.zeros([nregressors, nhrfs])

        # Try contrast on conditions
        coeffs, indices = _contrast(cont, self.conditions())        
        if len(coeffs): 
            J = np.arange(nhrfs)
            for (c,i) in zip(coeffs, indices):
                mat[nhrfs*i+J, J] = c
        
        # Otherwise try contrast on extra regressors
        else:
            coeffs, indices = _contrast(cont, self.extra_regressors()) 
            if len(coeffs):       
                j = nhrfs*len(self.conditions())
                for (c,i) in zip(coeffs, indices):
                    mat[i+j, j] = c
        return mat
