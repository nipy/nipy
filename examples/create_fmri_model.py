import numpy as np 
import sympy as sym

from neuroimaging.modalities.fmri.model import LinearModel
from neuroimaging.modalities.fmri import formula, hrf


# Define symbolic regressors
m1 = formula.Term('visual')
m2 = formula.Term('audio')
w1 = formula.Term('word1')
w2 = formula.Term('word2')
w3 = formula.Term('word3')

c1 = m1*w1
c2 = m1*w2
c3 = m1*w3
c4 = m2*w1
c5 = m2*w2
c6 = m2*w3

"""
c1 = formula.Term('visual word1')
c2 = formula.Term('visual word2')
c3 = formula.Term('visual word3')
c4 = formula.Term('audio word1')
c5 = formula.Term('audio word2')
c6 = formula.Term('audio word3')
"""

hb = formula.Term('heartbeat')
ux = formula.Term('translation x')
uy = formula.Term('translation y')
uz = formula.Term('translation z')

# The abstract model is essentially a list of symbols
m = [c1, c2, c3, c4, c5, c6, hb, ux, uy, uz]

# We can define contrasts symbolically
con = c1-c2 

# Instantiate a model 
lm = LinearModel(m, hrf=[hrf.glover_sympy, hrf.dglover_sympy])

lm.set_condition(c1, onsets=np.array([3,9,15]))
lm.set_condition(c2, onsets=np.array([6,12,18]))

## This creates a list of symbolic expressions in lm.regressors[c1]

#lm.set_regressor(ux, val=array, timestamps=timestamps, interp=cubic_spline, units='tr')

lm.set_drift(order=3, expression=sym.Function('cos'))
##lm.set_drift(order=3)

timestamps = np.linspace(0, 25)
X = lm.design_matrix(timestamps)

print X

"""
con_vect = lm.contrast(c1-c2, params)
"""

