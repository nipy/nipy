#!/bin/env python
"""
This quickstart example shows how you set up regressors for an FMRI
design, and convolve them with basis functions such as a canonical HRF
"""

import pylab
import numpy as N

from neuroimaging.modalities.fmri import protocol
from neuroimaging.modalities.fmri import hrf

"""
Experiment: periodic block with two conditions 'hot' and 'warm'.
Blocks are 9 seconds long, with 9 seconds rest in between. The first block
begins at T=9 in experiment time. There are 10 repetitions of each stimulus.
"""

hot = [['hot',t+9.,t+18.] for t in N.arange(0,360,36)]
warm = [['warm',t+27.,t+36.] for t in N.arange(0,360,36)]

pain_factor = protocol.ExperimentalFactor('pain', hot+warm, delta=False)

hot_events = pain_factor['hot']
warm_events = pain_factor['warm']

"""
Let's assume we have a 3 second TR.
"""

frametimes = N.arange(0,120,3)

"""
Let's see what evaluating the Factor gives us.
"""

print pain_factor(time=frametimes)

"""
Should see:

[[ 0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.
   0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.
   0.  0.  0.  0.]]

Note that the protocol is really right-continuous. This has little effect on
the convolved signal because a much finer time scale is used, but when
evaluating the design on the frametimes, the behaviour is slightly unexpected.

"""

pylab.subplot(411)
pylab.plot(frametimes, hot_events(time=frametimes), label='Hot', linestyle='steps')
pylab.plot(frametimes, warm_events(time=frametimes), label='Warm', linestyle='steps')
axes = pylab.gca()
axes.set_ylim([-0.1,1.1])
pylab.legend()
pylab.title('Regressors in NiPy')

"""
We will convolve the response with a filter, the canonical 'Glover' filter.
Note that hot_events and warm_events are NOT convolved with this filter.
"""

pain_factor.convolve(hrf.canonical)

ptime = N.linspace(0,100,1000)
pain_convolved = pain_factor(time=ptime)

pylab.subplot(412)
pylab.plot(ptime, pain_convolved[0], label='Hot')
pylab.plot(ptime, pain_convolved[1], label='Warm')
pylab.legend()

"""
We can 'turn off' the convolution.
"""

pain_factor.convolved = False
pain_unconvolved = pain_factor(time=ptime)

pylab.subplot(413)
pylab.plot(ptime, pain_unconvolved[0], label='Hot', linestyle='steps')
pylab.plot(ptime, pain_unconvolved[1], label='Warm', linestyle='steps')
axes = pylab.gca()
axes.set_ylim([-0.1,1.1])
pylab.legend()


"""
We can also convolved with a filter that has two basis functions,
say the Glover filter and its derivative. 
"""

pain_factor.convolve(hrf.glover_deriv)
pain_convolved = pain_factor(time=ptime)

names = pain_factor.names()

pylab.subplot(414)
for i in range(4):
    pylab.plot(ptime, pain_convolved[i], label=names[i], linestyle='steps')
pylab.legend()
pylab.xlabel('Time')

"""
See what happens when we turn off the
'convolution'
"""

pain_factor.convolved = False
print pain_factor.names()

# Show our nice figure
pylab.show()


