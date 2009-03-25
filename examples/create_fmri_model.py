import numpy as np 
import sympy as sym

from neuroimaging.modalities.fmri.model import LinearModel
from neuroimaging.modalities.fmri import formula, hrf

"""
Fixes
- Drift model is wrong
- opposite of the derivative
- contrast definition is wrong

toAdd
- deal with block designs
- deal with multiple session
- specification of the time functions (motion etc)
- FIR model has to be done
- columns identifiers
- instantiate from a .csv file

- normalization of the amplitude -> check SPM
"""

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

hb = formula.Term('heartbeat')
ux = formula.Term('translation x')
uy = formula.Term('translation y')
uz = formula.Term('translation z')


# We can define contrasts symbolically
con = c1-c2 

# Instantiate a model 
lm = LinearModel(hrf=['glover','dglover'])

lm.condition(c1, onsets=np.array([3,9,15]))
lm.condition(c2, onsets=np.array([6,12,18]))

## This creates a list of symbolic expressions in lm.regressors[c1]

#lm.set_regressor(ux, val=array, timestamps=timestamps, interp=cubic_spline, units='tr')

lm.drift(order=3, expression=sym.Function('cos'))
##lm.set_drift(order=3)

timestamps = np.linspace(0, 25)
X = lm.design_matrix(timestamps)
X = X/np.sqrt(np.sum(X**2,0))

import matplotlib.pylab as mp
mp.figure()
mp.imshow(X,interpolation='nearest')
mp.colorbar()
mp.show()

"""
con_vect = lm.contrast(c1-c2, params)
"""

