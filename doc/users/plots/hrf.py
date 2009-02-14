"""
Plot of the canonical Glover HRF
"""

from neuroimaging.modalities.fmri import hrf
import numpy as np
import pylab


from matplotlib import rc
rc('text', usetex=True)

t = np.linspace(0,25,200)
pylab.plot(t, hrf.glover(t))
a=pylab.gca()
a.set_xlabel(r'$t$')
a.set_ylabel(r'$h_{can}(t)$')


