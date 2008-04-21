#!/bin/env python
"""
This quickstart example shows you how to setup a model formula
for an fMRI experiment, including drift terms, and interactions

The example only sets up the model, and does not estimate it
"""

import pylab
import numpy as N

from neuroimaging.modalities.fmri import protocol
from neuroimaging.modalities.fmri import hrf
from neuroimaging.modalities.fmri import functions

"""
Experiment: periodic block with two conditions 'hot' and 'warm'.
Blocks are 9 seconds long, with 9 seconds rest in between. The first block
begins at T=9 in experiment time. There are 10 repetitions of each stimulus.
"""

hot = [['hot',t+9.,t+18.] for t in N.arange(0,360,36)]
warm = [['warm',t+27.,t+36.] for t in N.arange(0,360,36)]

pain_factor = protocol.ExperimentalFactor('pain', hot+warm, delta=False)
pain_factor.convolve(hrf.canonical)

drift = functions.SplineConfound(df=7, window=(0,360))

"""
Look at the spline basis. The first four terms (not plotted) are just

[time**i for i in range(4)]

"""

T = N.linspace(0,360,3000)
def normalize(x):
    M = N.fabs(x).max()
    if M != 0:
        return x / M
    else:
        return x

pylab.subplot(311)
for i in range(4,7):
    pylab.plot(T,normalize(drift(T)[i]))

ax = pylab.gca()
ax.set_ylim([-0.1,1.1])
pylab.title('Models in NIPY')

"""
The drift function is not yet a Term in a model. For
fMRI protocols, we refer to the 'Term's as 'ExperimentalQuantitative's.
As for the experiment Term, it needs a name.
"""

drift_term = protocol.ExperimentalQuantitative('spline_drift', drift)

"""
Create a formula with pain_factor and drift.
"""

model = pain_factor + drift_term
print model

"""
Evaluate the design matrix at a given
collection of points.
"""

design = model(time=T)
print design.shape

"""
There is a prenamed ExperimentalQuantitative called Time.
Let's add an interaction between Time and pain_factor.
"""

newmodel = model + protocol.Time * pain_factor
design = newmodel(time=T)
print newmodel, design.shape

"""
Let's look at the interaction term.
"""

interaction = newmodel['pain*time']

pylab.subplot(312)
for i in range(2):
    pylab.plot(T, interaction(time=T)[i], label=interaction.names()[i])
pylab.legend()

"""
How about a cubic.
"""

newmodel = newmodel + protocol.Time**3 * pain_factor
interaction = newmodel['pain*time^3']
print newmodel, interaction

pylab.subplot(313)
for i in range(2):
    pylab.plot(T, interaction(time=T)[i], label=interaction.names()[i])
pylab.legend()
pylab.xlabel('Time')

# And show our nice figure
pylab.show()

