import os

import numpy as N
import pylab

from neuroimaging.modalities.fmri.api import FmriImage
from neuroimaging.modalities.fmri.protocol import ExperimentalFactor
from neuroimaging.core.api import Image
from neuroimaging.data_io.api import DataSource

eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
eventdict_r = {'SSt_SSp':1, 'SSt_DSp':2, 'DSt_SSp':3, 'DSt_DSp':4}

base = 'http://kff.stanford.edu/FIAC'

def FIACpath(path, subj=3, run=3, base=base):

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

    pfile = DataSource().open(url)
    pfile = pfile.read().strip().split('\n')

    # start with a "deadtime" interval

    times = []
    events = []

    for row in pfile:
        time, eventtype = map(float, row.split())
        times.append(time)
        events.append(eventdict[eventtype])

    # take off the first 3.33 seconds of each eventtype for the block design
    # the blocks lasted 20 seconds with 9 seconds of rest at the end

##    keep = range(0, len(events), 6)
##     intervals = [[events[keep[i]], times[keep[i]] + 3.33, times[keep[i]]+20.] for i in range(len(keep))]
##     p = ExperimentalFactor('FIAC_design', intervals, delta=False)

    notkeep = range(0, len(events), 6)
    intervals = [[events[i], times[i]] for i in range(len(events)) if i not in notkeep]
    p = ExperimentalFactor('FIAC_design', intervals)
    p.design_type = 'block'
    return p

def FIACbegin_block(subj=3, run=3):

    url = FIACpath('subj%(subj)d_bloc_fonc%(run)d.txt' % {'subj':subj, 'run':run}, subj=subj, run=-1)

    pfile = DataSource().open(url)
    pfile = pfile.read().strip().split('\n')

    # start with a "deadtime" interval

    times = []
    events = []

    for row in pfile:
        time, eventtype = map(float, row.split())
        times.append(time)
        events.append(eventdict[eventtype])

    # take off the first 3.33 seconds of each eventtype for the block design
    # the blocks lasted 20 seconds with 9 seconds of rest at the end

    keep = range(0, len(events), 6)
    intervals = [['Begin', times[keep[i]]] for i in range(len(keep))]
    return ExperimentalFactor('beginning', intervals)

def FIACbegin_event(subj=3, run=3):

    intervals = [['Begin', 2.]]
    return ExperimentalFactor('beginning', intervals)

def FIACevent(subj=3, run=2):
    url = FIACpath('subj%(subj)d_evt_fonc%(run)d.txt' % {'subj':subj, 'run':run}, subj=subj, run=-1)

    pfile = DataSource().open(url)
    pfile = pfile.read().strip().split('\n')

    events = []; times = []

    for row in pfile:
        time, eventtype = map(float, row.split())
        times.append(time)
        events.append(eventdict[eventtype])

    times.pop(0); events.pop(0) # delete first event as Keith has
    intervals = [[events[i], times[i]] for i in range(len(events))]
    
    p = ExperimentalFactor('FIAC_design', intervals)
    p.design_type = 'event'

    return p

def FIACplot(subj=3, run=3, tmin=1.0, tmax=476.25, dt=0.2, save=False):
    experiment = FIACprotocol(subj=subj, run=run)

    t = N.arange(tmin,tmax,dt)

    for event in eventdict.values():
        pylab.plot(t, experiment[event](t), label=event,
                       linewidth=2, linestyle='steps')

        gca = pylab.axes()
        gca.set_ylim([-0.1,1.1])
    pylab.legend(eventdict.values())
    if save:
        pylab.savefig(FIACpath('protocol.png', subj=subj, run=run))

def FIACfmri(subj=3, run=3, test=False):
    if not test:
        url = FIACpath('fsl/filtered_func_data.img', subj=subj, run=run)
        f = FmriImage(url, usematfile=False)
    else:
        import test
        return test.test()
    return f

def FIACmask(subj=3, run=3):
    url = FIACpath('fsl/mask.img', subj=subj, run=run)
    return Image(url)

