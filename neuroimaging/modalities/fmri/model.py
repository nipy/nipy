import numpy as np
import sympy as sym
import utils, hrf, formula

class LinearModel:
    
    def __init__(self, terms, hrf, baseline=True): 
        self.terms = terms + [formula.Term('drift')]
        self.hrf = hrf
        self.baseline = baseline
        self.formula = formula.Formula([])

    def set_condition(self, term, onsets, amplitudes=None, durations=None): 
        if self.terms.count(term): 
            aux = [utils.events(onsets, amplitudes=amplitudes, f=f) for f in self.hrf]
            # or blocks(...)
            self.formula += formula.Formula(aux)

    def set_drift(self, order, expression=None): 
        t = formula.Term('t')
        if isinstance(expression, sym.function.FunctionClass): 
            monom = expression(t)
        else: 
            monom = t
        aux = [monom**(i+1) for i in range(order)] 
        self.formula += formula.Formula(aux)
        
    def set_regressor(self): 
        return

    def design_matrix(self, timestamps): 
        tval = timestamps.view(np.dtype([('t', np.float)]))
        aux = formula.Design(self.formula, return_float=True)(tval)
        if self.baseline==False:
            X = aux
        else: 
            X = np.zeros([aux.shape[0], aux.shape[1]+1])
            X[:,0] = 1.
            X[:,1:] = aux
        return X


            
