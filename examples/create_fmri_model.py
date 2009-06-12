import numpy as np 
import sympy as sym
from numpy.random import randn

from nipy.modalities.fmri.model import LinearModel
from nipy.modalities.fmri import formula, hrf

"""
todo
- deal with multiple session
- instantiate from a .csv file
- add modulations of the regressors
- add FIR model 
- normalization of the amplitude -> check SPM
- contrasts based on fatcors (lm.contrast(m1))
- convolution of regressors with hrf

question
- First-level (time) versus second-level(no time)  model ?

Fixes
- blocks should be handled more cleanly
- higher-order (spline) interpolation for functiona of time
"""

n=50
tr = 0.6
offset = tr/2
timestamps = np.linspace(offset,offset+tr*(n-1),n)
# caveat : time units can only be seconds 

ux_values = np.cumsum(randn(np.size(timestamps)))

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


# Instantiate a model 
lm = LinearModel(hrf=['glover','dglover'])

# event regressors
lm.add_condition(c1, onsets=np.array([3,9,15]),amplitudes = np.array([1.5,1,2]), hrf=['glover'])
lm.add_condition(c2, onsets=np.array([1,12,18]))

# block regressor
lm.add_condition(c3, onsets=np.array([5,14,22]),durations = np.array([4,4,4]))

## This creates a list of symbolic expressions in lm.regressors
lm.add_regressor(ux,values=ux_values,timestamps=timestamps, order=1)

# add drifts (NB : normally, either polynomial or cosine, not both)
lm.polynomial_drift(order=3)
lm.cosine_drift(duration=30,hfcut=12)

# create the design matrix
X = lm.design_matrix(timestamps)

# We can define contrasts symbolically
# notion of retriction
con_vect = lm.contrast(c2-c3)
con_vect = lm.contrast(c2-c3,'glover')
con_vect = lm.contrast(c1-c3,'glover')
con_vect = lm.contrast(c1-c3)
con_vect = lm.contrast(2*c1-c2-c3)



X = X/np.sqrt(np.sum(X**2,0))
import matplotlib.pylab as mp
mp.figure()
mp.imshow(X,interpolation='nearest')
mp.colorbar()
mp.show()
