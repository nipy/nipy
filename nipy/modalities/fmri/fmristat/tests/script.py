import numpy as np

from nipy.fixes.scipy.stats.models.regression import OLSModel, ARModel
from nipy.modalities.fmri.fmristat import hrf as delay
from nipy.modalities.fmri import formula 
from nipy.io.files import load
from test_FIAC import protocol

event = [(0,3),(0,4)] # Dictionaries with all the (subj, run) event designs 
block = [(0,1),(0,2)] # Dictionaries with all the (subj, run) block designs 

root = "/home/jtaylo/FIACmiller"

t = formula.make_recarray(np.arange(191)*2.5+1.25, 't')
for subj, r in block + event:
    if (subj, r) in event:
        design = 'evt'
    else:
        design = 'bloc'
    p, tcons, fcons = protocol(file("%(root)s/fiac%(subj)d/subj%(subj)d_%(design)s_fonc%(run)d.txt" % {'root':root, 'subj':subj, 'run':r, 'design':design}), 'event', *delay.spectral)
    X, tcons = p.design(t, contrasts=tcons)
    X, fcons = p.design(t, contrasts=fcons)

    m = OLSModel(X)
    f = np.array(load("%(root)s/fiac%(subj)d/fonc%(run)d/fsl/filtered_func_data.img" % {'root':root, 'subj':subj, 'run':r}))
    mm = np.array(load("%(root)s/fiac%(subj)d/fonc%(run)d/fsl/mask.img" % {'root':root, 'subj':subj, 'run':r}))
    ar = np.zeros(f.shape[1:])

    for s in range(f.shape[1]):
        print s
        d = np.array(f[:,s])
        flatd = d.reshape((d.shape[0], np.product(d.shape[1:])))
        result = m.fit(flatd)
        ar[s] = ((result.resid[1:] * result.resid[:-1]).sum(0) / (result.resid**2).sum(0)).reshape(ar.shape[1:])

    # round AR to nearest one-hundredth

    ar *= 100
    ar = ar.astype(np.int) / 100.
    # smooth here?
    # ar = smooth(ar, 8.0)
    
    output = {}
    for n in tcons.keys():
        output[n] = {}
        output[n]['sd'] = np.zeros(f.shape[1:])
        output[n]['t'] = np.zeros(f.shape[1:])
        output[n]['effect'] = np.zeros(f.shape[1:])

    for n in fcons.keys():
        output[n] = np.zeros(f.shape[1:])

    arvals = np.unique(ar)
    for val in arvals:
        mask = np.equal(ar, val)
        mask *= mm
        m = ARModel(X, val)
        d = f[:,mask]
        results = m.fit(d)
        print val, mask.sum()

        for n in tcons.keys():
            o = output[n]
            resT = results.Tcontrast(tcons[n])
            output[n]['sd'][mask] = resT.sd
            output[n]['t'][mask] = resT.t
            output[n]['effect'][mask] = resT.effect
        # Where should we save these? 

        for n in fcons.keys():
            output[n] = results.Fcontrast(fcons[n])
