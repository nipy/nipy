import numpy as np
import fff2.glm as glm


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
vy = np.ones([dimt, dimx*dimy*dimz])
e = np.random.randn(dimt, dimx*dimy*dimz)
y = y + e
axis = 0

"""
y = random.randn(dimx, dimt, dimy, dimz)
vy = ones([dimx, dimt, dimy, dimz])
e = random.randn(dimx, dimt, dimy, dimz)
y = y + e
axis = 1
"""

X = np.array([np.ones(dimt), range(dimt)])
X = X.transpose() ## the design matrix X must have dimt lines

"""
pX = linalg.pinv(X)
beta, s2, z, vz = mfx.em(y, vy, X, pX, axis=axis)
##beta, norm_var_beta, s2, dof = glm.kalman.ols(y, X, axis=axis)
##beta, norm_var_beta, s2, dof = glm.ols(y, X, axis=axis)
ll = mfx.log_likelihood(y, vy, X, beta, s2, axis=axis) 
"""

m = glm.glm(y, X, vy=vy, axis=axis)
m.fit() 
lcon = m.contrast([1,0], type='l') 
t, p, z = lcon.test(zscore=True)
F, p = m.contrast([[1,0],[1,-1]], type='L').test() ## recognizes a F contrast


print np.shape(y)
print np.shape(X)
