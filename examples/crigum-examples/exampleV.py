"""
How to estimate resels, and compute EC densities
"""

import pylab

import numpy as N
import numpy.random as R

from neuroimaging.algorithms.statistics import intrinsic_volumes
from neuroimaging.algorithms.statistics.rft import Gaussian

def box(shape, edges):
    data = N.zeros(shape)
    sl = []
    for i in range(len(shape)):
        sl.append(slice(edges[i][0], edges[i][1],1))
    data[sl] = 1
    return data

shape = (40,40,40)
Z = R.standard_normal((20,)+shape)

X = box(shape, [[10,20],[10,20],[20,30]])
lk =  [intrinsic_volumes.LK(X, 0.5, coords=Z, lk=d) for d in range(4)]
ec = Gaussian(search=lk)

X = N.linspace(4,6,1000)
pylab.plot(X, ec(X))
pylab.plot([0,10],[0.05,0.05])
a = pylab.gca()
a.set_ylim([0,0.5])
a.set_xlim([4,6])
thresh = X[N.argmin((ec(X) - 0.05)**2)]
pylab.title("Threshold is %f" % thresh)

pylab.show()
