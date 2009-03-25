import numpy as np
import sympy 
import utils, hrf, formula

class LinearModel:
    
    def __init__(self, terms, hrf): 
        self.terms = terms + [formula.Term('drift')]
        self.hrf = hrf
        self.formula = formula.Formula([])

    def set_condition(self, term, onsets, amplitudes=None, durations=None): 
        if self.terms.count(term): 
            aux = [utils.events(onsets, amplitudes=amplitudes, f=f) for f in self.hrf]
            # or blocks(...)
            self.formula += formula.Formula(aux)

    def set_drift(self, order=3, expression=None): 
        if isinstance(expression, sympy.function.FunctionClass): 
            monom = expression(formula.Term('t'))
        else: 
            monom = formula.Term('t')
        aux = [monom**i for f in self.hrf for i in range(order)] 
        self.formula += formula.Formula(aux)

    def set_regressor(self): 
        return

    def design_matrix(self, timestamps): 
        tval = timestamps.view(np.dtype([('t', np.float)]))
        return formula.Design(self.formula, return_float=True)(tval)

    def formula(self): 
        return 

