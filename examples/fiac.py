import string
import neuroimaging
import numpy as N
import urllib
import pylab

from neuroimaging.visualization import viewer
from neuroimaging.fmri import protocol, fmristat
from neuroimaging.fmri.hrf import HRF
from neuroimaging.statistics import contrast

from neuroimaging.fmri.plotting import MultiPlot

import time, gc

eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}

def FIACprotocol(subj=3, run=3):
    return FIACblock(subj=subj, run=run)
    try:
        return FIACblock(subj=subj, run=run)
    except:
        try:
            return FIACevent(subj=subj, run=run)
        except:
            raise ValueError, 'error creating design'

def FIACblock(subj=3, run=3):
    url = 'http://kff.stanford.edu/FIAC/fiac%(subj)d/subj%(subj)d_bloc_fonc%(run)d.txt' % {'subj':subj, 'run':run}

    pfile = urllib.urlopen(url)
    pfile = pfile.read().strip().split('\n')

    # start with a "deadtime" interval

    times = []
    events = []

    for row in pfile:
        time, eventtype = map(float, string.split(row))
        times.append(time)
        events.append(eventdict[eventtype])

    # take off the first 3.33 seconds of each eventtype for the block design
    # the blocks lasted 20 seconds with 9 seconds of rest at the end

    keep = range(0, len(events), 6)
    intervals = [[events[keep[i]], times[keep[i]] + 3.33, times[keep[i]]+20.] for i in range(len(keep))]
    
    return protocol.ExperimentalFactor('FIAC_block', intervals)

def FIACevent(subj=3, run=2):
    url = 'http://kff.stanford.edu/FIAC/fiac%(subj)d/subj%(subj)d_evt_fonc%(run)d.txt' % {'subj':subj, 'run':run}

    pfile = urllib.urlopen(url)
    pfile = pfile.read().strip().split('\n')

    # start with a "deadtime" interval

    times = [0.]
    events = ['deadtime']

    for row in pfile:
        time, eventtype = map(float, string.split(row))
        times.append(time)
        events.append(eventdict[eventtype])

    intervals = [[events[i], times[i], times[i]+3.33] for i in range(len(events))]
    
    return protocol.ExperimentalFactor('FIAC_design', intervals)

def FIACplot(subj=3, run=3, tmin=0., tmax=475., dt=0.2):
    experiment = protocol(subj=subj, run=run)

    t = N.arange(tmin,tmax,dt)

    for event in eventdict.values():
        l = pylab.plot(t, experiment[event](t), label=event,
                       linewidth=2, linestyle='steps')

        gca = pylab.axes()
        gca.set_ylim([-0.1,1.1])
    pylab.legend(eventdict.values())

def FIACfmri(subj=3, run=3):
    url = 'http://kff.stanford.edu/FIAC/fiac%(subj)d/fsl%(run)d/filtered_func_data.img' % {'subj':subj, 'run':run}

    return neuroimaging.fmri.fMRIImage(url, usematfile=False)

def FIACmask(subj=3, run=3):
    url = 'http://kff.stanford.edu/FIAC/fiac%(subj)d/fsl%(run)d/mask.img' % {'subj':subj, 'run':run}
    return neuroimaging.image.Image(url)

IRF = HRF(deriv=True)

f = FIACfmri()
m = FIACmask()

p = FIACprotocol()
p.convolve(IRF)

drift_fn = protocol.SplineConfound(window=[0,475], df=10)
drift = protocol.ExperimentalQuantitative('drift', drift_fn)

formula = p + drift

# output some contrasts, here is one from the term "p" in "formula"

task = contrast.Contrast(p, formula, name='task')

# another built by linear combinations of functions of (experiment) time

SSt_SSp = p['SSt_SSp'].astimefn()
DSt_SSp = p['DSt_SSp'].astimefn()
SSt_DSp = p['SSt_DSp'].astimefn()
DSt_DSp = p['DSt_DSp'].astimefn()

overall = (SSt_SSp + DSt_SSp + SSt_DSp + DSt_DSp) * 0.25

# important: overall is NOT convolved with HRF even though p was!!!
# irf here is the canonical Glover HRF (with no derivative in this case)

irf = HRF(deriv=False)
overall = irf.convolve(overall)
overall = contrast.Contrast(overall,
                            formula,
                            name='overall')

# the "FIAC" contrasts
# annoying syntax: floats must be on LHS -- have to fix __add__, __mul__ methods of TimeFunction

sentence = (DSt_SSp + DSt_DSp) * 0.5 - (SSt_SSp + SSt_DSp) * 0.5
sentence = irf.convolve(sentence)
sentence = contrast.Contrast(sentence, formula, name='sentence')

speaker =  (SSt_DSp + DSt_DSp) / 2. - (SSt_SSp + DSt_SSp) / 2.
speaker = irf.convolve(speaker)
speaker = contrast.Contrast(speaker, formula, name='speaker')

interaction = SSt_SSp - SSt_DSp - DSt_SSp + DSt_DSp
interaction = irf.convolve(interaction)
interaction = contrast.Contrast(interaction, formula, name='interaction')

contrasts = [task, overall, sentence, speaker, interaction]

# OLS pass

OLS = fmristat.fMRIStatOLS(f, formula=formula, mask=m)

toc = time.time()
OLS.fit(resid=True)
tic = time.time()

print 'OLS time', `tic-toc`

rho = OLS.rho_estimator.img
rho.tofile('rho.img')

v=viewer.BoxViewer(rho)
v.draw()

# AR pass

toc = time.time()
AR = fmristat.fMRIStatAR(OLS, contrasts=contrasts)
AR.fit()
tic = time.time()

print 'AR time', `tic-toc`

t = neuroimaging.image.Image('fmristat_run/contrasts/speaker/t.img')
v=viewer.BoxViewer(t)
v.draw()
pylab.show()
