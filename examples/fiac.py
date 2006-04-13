import string, os, gc
import neuroimaging
import numpy as N
import urllib
import pylab
from neuroimaging.visualization import viewer

from neuroimaging.fmri import protocol
from neuroimaging.fmri import fmristat
from neuroimaging.statistics import contrast
from neuroimaging.fmri.plotting import MultiPlot
from neuroimaging.fmri.regression import ResidOutput
from neuroimaging.image.fwhm import fastFWHM
from neuroimaging.fmri.fmristat.delay import DelayHRF
from neuroimaging.fmri.hrf import SpectralHRF, canonical, glover_deriv
import time, gc

eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}

def FIACpath(path, subj=3, run=3):
    base = '/home/analysis/FIAC/'
    if run > 0:
        return os.path.join(base, 'fiac%d' % subj, 'fonc%d' % run, path)
    else:
        return os.path.join(base, 'fiac%d' % subj, path)
        
def FIACprotocol(subj=3, run=3):
    good = False

    for ptype in [FIACblock, FIACevent]:
        try:
            v = ptype(subj=subj, run=run)
            good = True
        except:
            pass

    if good:
        return v
    else:
        return None

def FIACblock(subj=3, run=3):

    url = FIACpath('subj%(subj)d_bloc_fonc%(run)d.txt' % {'subj':subj, 'run':run}, subj=subj, run=-1)

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
    intervals = [[events[keep[i]], times[keep[i]] + 3.33, times[keep[i]]+20.]
                 for i in range(len(keep))]
    
    p = protocol.ExperimentalFactor('FIAC_block', intervals)
    p.design_type = 'block'
    return p

def FIACbegin(subj=3, run=3):

    url = FIACpath('subj%(subj)d_bloc_fonc%(run)d.txt' % {'subj':subj, 'run':run}, subj=subj, run=-1)

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
    intervals = [['Begin', times[keep[i]], times[keep[i]] + 3.33] for i in range(len(keep))]
    return protocol.ExperimentalFactor('beginning', intervals)
    

def FIACevent(subj=3, run=2):
    url = FIACpath('subj%(subj)d_evt_fonc%(run)d.txt' % {'subj':subj, 'run':run}, subj=subj, run=-1)

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
    
    p = protocol.ExperimentalFactor('FIAC_design', intervals)
    p.design_type = 'event'

    return p

def FIACplot(subj=3, run=3, tmin=0., tmax=476.25, dt=0.2):
    experiment = FIACprotocol(subj=subj, run=run)

    t = N.arange(tmin,tmax,dt)

    for event in eventdict.values():
#        print N.array(experiment[event](t)).shape, 'now'
        l = pylab.plot(t, experiment[event](t), label=event,
                       linewidth=2, linestyle='steps')

        gca = pylab.axes()
        gca.set_ylim([-0.1,1.1])
    pylab.legend(eventdict.values())
    pylab.savefig(FIACpath('protocol.png', subj=subj, run=run))

def FIACfmri(subj=3, run=3):
    url = FIACpath('fsl/filtered_func_data.img', subj=subj, run=run)
    f = neuroimaging.fmri.fMRIImage(url, usematfile=False)
    return f

def FIACmask(subj=3, run=3):
    url = FIACpath('fsl/mask.img', subj=subj, run=run)
    return neuroimaging.image.Image(url)

