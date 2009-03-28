"""
Example of using the 'lme4' R package (mixed effect models) via the
glm class.
"""
import numpy as np
import sys
from fff2 import glm

# Import data
dat = np.load('data/sleepstudy.npz')
X = dat['X']
n = X.shape[0]
y = dat['y']
#y = np.random.rand(n)-.5

axis = 0
formula = 'y ~ x1 + (x1|x2)'
method = 'lme4' ## other choice: 'rpy_lme4'

# Fit model
m = glm.glm(y, X, model='mfx', axis=axis, formula=formula, method=method)
