import numpy as np
from fff2 import glm

# This example routine simulates a number of pure Gaussian white noise 
# signals, then fits each one in terms of two regressors: a constant baseline, 
# and a linear function of time. The voxelwise t statistics associated 
# with the baseline coefficient are then computed. 

dimt = 100
dimx = 10
dimy = 11
dimz = 12 

# axis defines the "time direction" 

y = np.random.randn(dimt, dimx*dimy*dimz)
axis = 0

"""
y = random.randn(dimx, dimt, dimy, dimz)
axis = 1
"""

X = np.array([np.ones(dimt), range(dimt)])
X = X.transpose() ## the design matrix X must have dimt lines

#mod = glm.glm(y, X, axis=axis) ## default is spherical model using OLS 
mod = glm.glm(y, X, axis=axis, model='ar1')
#mod = glm.glm(y, X, formula='y~x1+(x1|x2)', axis=axis, model='mfx')

##mod.save('toto')
##mod = glm.load('toto')

# Define a t contrast
tcon = mod.contrast([1,0]) 

# Compute the t-stat
t = tcon.stat()
## t = tcon.stat(baseline=1) to test effects > 1 

# Compute the p-value
p = tcon.pvalue()

# Compute the z-score
z = tcon.zscore()

# Perform a F test without keeping the F stat
p = mod.contrast([[1,0],[1,-1]]).pvalue()

# Perform a conjunction test similarly 
##p = mod.contrast([[1,0],[1,-1]], type='tmin').pvalue()

print np.shape(y)
print np.shape(X)
print np.shape(z)

