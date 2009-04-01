import numpy as np
from fff2.glm import glm
from time import time
import sys

def display(t, title=''):
    import pylab
    pylab.figure()
    pylab.imshow(t)
    pylab.xlabel('Time')
    pylab.ylabel('Electrode')
    pylab.title(title)
    pylab.show()
    
# Load data
Y = np.load('bb.npy')
nsubjects = Y.shape[2]
nconditions = Y.shape[3]
ndata = nsubjects*nconditions
conditions = np.asarray(range(nconditions))*50 + 50
saturation = np.array([0,0,1,1,1,1])

# Regressors
baseline = np.ones(ndata)
conditions = np.tile(conditions, nsubjects)
saturation = np.tile(saturation, nsubjects)
subject_factor = np.repeat(np.asarray(range(nsubjects)),6)

## Models
#X = np.asarray([baseline, conditions, saturation]).T 

#X = np.asarray([conditions, saturation, subject_factor]).T
#formula='y~1+x1+x2+(1|x3)+(x1|x3)+(x2|x3)'
#contrasts = ([0,1,0], [0,0,1])

X = np.asarray([conditions, subject_factor]).T 
formula = 'y~x1+(1|x2)'
contrasts = ([1,0], )

# Test: reduce data
y = Y.reshape([Y.shape[0],Y.shape[1],ndata])
y = y[0:3,0:3,:]

# Standard t-stat
print('Starting fitting...')
tic = time()
m = glm(y, X, axis=2, formula=formula, model='mfx')
dt = time()-tic
print('  duration = %d sec' % dt)
m.save('dump')


# Linear contrast
for con in contrasts:
    c = m.contrast(con)
    t = c.stat() 
    display(t, title='Linear contrast')
