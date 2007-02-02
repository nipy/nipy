import urllib

from neuroimaging.modalities.fmri import protocol, functions

eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
eventdict_r = {}
for key, value in eventdict.items():
    eventdict_r[value] = key

def block_protocol(url):
    """
    From one of the FIAC block design descriptions, return two
    regressors for use in an fMRI formula: one the 'beginning'
    regressor, the other the protocol.
    
    """

    pfile = urllib.urlopen(url)
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

    notkeep = range(0, len(events), 6)
    intervals = [[events[i], times[i]] for i in range(len(events)) if i not in notkeep]
    p = protocol.ExperimentalFactor('FIAC_design', intervals)
    p.design_type = 'block'

    keep = range(0, len(events), 6)
    intervals = [['Begin', times[keep[i]]] for i in range(len(keep))]
    b = protocol.ExperimentalFactor('beginning', intervals)
    return b, p

def event_protocol(url):

    """
    From one of the FIAC event design descriptions, return two
    regressors for use in an fMRI formula: one the 'beginning'
    regressor, the other the protocol.
    
    """

    pfile = urllib.urlopen(url)
    pfile = pfile.read().strip().split('\n')

    events = []
    times = []

    for row in pfile:
        time, eventtype = map(float, row.split())
        times.append(time)
        events.append(eventdict[eventtype])

    times.pop(0)
    events.pop(0) # delete first event as Keith has
    intervals = [[events[i], times[i]] for i in range(len(events))]
    
    p = protocol.ExperimentalFactor('FIAC_design', intervals)
    p.design_type = 'event'

    intervals = [['Begin', 2.]]
    b = protocol.ExperimentalFactor('beginning', intervals)

    return b, p

def drift(df=7, window=[0,477.5]):
    drift_fn = functions.SplineConfound(window=window, df=df)
    return protocol.ExperimentalQuantitative('drift', drift_fn)

