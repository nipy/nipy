import pylab
import numpy as np
from StringIO import StringIO
from os.path import exists, abspath, join as pjoin
from os import makedirs, listdir

from matplotlib.mlab import csv2rec, rec2csv

from nipy.fixes.scipy.stats.models.regression import OLSModel, ARModel, isestimable
from nipy.modalities.fmri.fmristat import hrf as delay
from nipy.modalities.fmri import formula, design, hrf
import nipy.testing as nitest
from nipy.io.api import load_image, save_image
from nipy.core import api

# data

datadir = "/home/jtaylo/fiac-curated"
#datadir = '/home/fperez/research/data/fiac'

# For group analysis

from nipy.algorithms.statistics import onesample 

def test_sanity():
    from nipy.modalities.fmri.fmristat.tests import FIACdesigns

    """
    Single subject fitting of FIAC model
    """

    # Based on file
    # subj3_evt_fonc1.txt
    # subj3_bloc_fonc3.txt

    for subj, run, dtype in [(3,1,'event'),
                             (3,3,'block')]:
        nvol = 191
        TR = 2.5 
        Tstart = 1.25

        volume_times = np.arange(nvol)*TR + Tstart
        volume_times_rec = formula.make_recarray(volume_times, 't')

        path_dict = {'subj':subj, 'run':run}
        if exists(pjoin(datadir, "fiac_%(subj)02d",
                        "block", "initial_%(run)02d.csv") % path_dict):
            path_dict['design'] = 'block'
        else:
            path_dict['design'] = 'event'

        experiment = csv2rec(pjoin(datadir, "fiac_%(subj)02d", "%(design)s", "experiment_%(run)02d.csv")
                             % path_dict)
        initial = csv2rec(pjoin(datadir, "fiac_%(subj)02d", "%(design)s", "initial_%(run)02d.csv")
                                % path_dict)

        X_exper, cons_exper = design.event_design(experiment, volume_times_rec, hrfs=delay.spectral)
        X_initial, _ = design.event_design(initial, volume_times_rec, hrfs=[hrf.glover]) 
        X, cons = design.stack_designs((X_exper, cons_exper),
                                       (X_initial, {}))

        Xf = np.loadtxt(StringIO(FIACdesigns.designs[dtype]))
        for i in range(X.shape[1]):
            yield nitest.assert_true, (matchcol(X[:,i], Xf.T)[1] > 0.999)
        
def matchcol(col, X):
    """
    Find the column in X with the highest correlation with col.

    Used to find matching columns in fMRIstat's design with
    the design created by Protocol. Not meant as a generic
    helper function.
    """
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    return np.argmax(c), c.max()

def rewrite_spec(subj, run, root = "/home/jtaylo/FIAC-HBM2009"):
    """
    Take a FIAC specification file and get two specifications
    (experiment, begin).

    This creates two new .csv files, one for the experimental
    conditions, the other for the "initial" confounding trials that
    are to be modelled out. 

    For the block design, the "initial" trials are the first
    trials of each block. For the event designs, the 
    "initial" trials are made up of just the first trial.

    """

    if exists(pjoin("%(root)s", "fiac%(subj)d", "subj%(subj)d_evt_fonc%(run)d.txt") % {'root':root, 'subj':subj, 'run':run}):
        designtype = 'evt'
    else:
        designtype = 'bloc'

    # Fix the format of the specification so it is
    # more in the form of a 2-way ANOVA

    eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
    s = StringIO()
    w = csv.writer(s)
    w.writerow(['time', 'sentence', 'speaker'])

    specfile = pjoin("%(root)s", "fiac%(subj)d", "subj%(subj)d_%(design)s_fonc%(run)d.txt") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    d = np.loadtxt(specfile)
    for row in d:
        w.writerow([row[0]] + eventdict[row[1]].split('_'))
    s.seek(0)
    d = csv2rec(s)

    # Now, take care of the 'begin' event
    # This is due to the FIAC design

    if designtype == 'evt':
        b = np.array([(d[0]['time'], 1)], np.dtype([('time', np.float),
                                                    ('initial', np.int)]))
        d = d[1:]
    else:
        k = np.equal(np.arange(d.shape[0]) % 6, 0)
        b = np.array([(tt, 1) for tt in d[k]['time']], np.dtype([('time', np.float),
                                                                 ('initial', np.int)]))
        d = d[~k]

    designtype = {'bloc':'block', 'evt':'event'}[designtype]

    fname = pjoin(datadir, "fiac_%(subj)02d", "%(design)s", "experiment_%(run)02d.csv") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(d, fname)
    experiment = csv2rec(fname)

    fname = pjoin(datadir, "fiac_%(subj)02d", "%(design)s", "initial_%(run)02d.csv") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(b, fname)
    initial = csv2rec(fname)

    return d, b

event = [(0,3),(0,4)] # Sequences with all the (subj, run) event designs 
block = [(0,1),(0,2)] # Sequences with all the (subj, run) block designs 


def run_model(subj, run):
    """
    Single subject fitting of FIAC model
    """

    # Number of volumes in the fMRI data
    nvol = 191

    # The TR of the experiment
    TR = 2.5 

    # The time of the first volume
    # XXX Matthew: FSL slicetimes the data so that the first
    # time point is in the midpoint [0,TR]. What does SPM do?
    # Answer: Tstart should be 0
    Tstart = 1.25

    # The array of times corresponding to each 
    # volume in the fMRI data

    volume_times = np.arange(nvol)*TR + Tstart

    # This recarray of times has one column named 't'
    # It is used in the function design.event_design
    # to create the design matrices.

    volume_times_rec = formula.make_recarray(volume_times, 't')

    # This few lines find the .csv files that 
    # describe the experiment

    path_dict = {'subj':subj, 'run':run}
    if exists(pjoin(datadir, "fiac_%(subj)02d",
                    "block", "initial_%(run)02d.csv") % path_dict):
        path_dict['design'] = 'block'
    else:
        path_dict['design'] = 'event'

    rootdir = pjoin(datadir, "fiac_%(subj)02d", "%(design)s") % path_dict

    # The following two lines read in the .csv files
    # and return recarrays, with fields
    # experiment: ['time', 'sentence', 'speaker']
    # initial: ['time', 'initial']

    if not exists(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict):
        raise IOError, "can't find design for subject=%d,run=%d" % (subj, run)

    experiment = csv2rec(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict)
    initial = csv2rec(pjoin(rootdir, "initial_%(run)02d.csv") % path_dict)

    # Create design matrices for the "initial" and "experiment" factors,
    # saving the default contrasts. 

    # The function event_design will create
    # design matrices, which in the case of "experiment"
    # will have num_columns = (# levels of speaker) * (# levels of sentence) * len(delay.spectral) = 2 * 2 * 2 = 8
    # For "initial", there will be (# levels of initial) * len([hrf.glover]) = 1 * 1 = 1

    # Here, delay.spectral is a sequence of 2 symbolic HRFs that 
    # are described in 
    # 
    # Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    #    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    #    data.\' NeuroImage, 16:593-606.

    # The contrasts, cons_exper,
    # is a dictionary with keys: ['constant_0', 'constant_1', 'speaker_0', 
    # 'speaker_1',
    # 'sentence_0', 'sentence_1', 'sentence:speaker_0', 'sentence:speaker_1']
    # representing the four default contrasts: constant, main effects + 
    # interactions,
    # each convolved with 2 HRFs in delay.spectral. Its values
    # are matrices with 8 columns.

    # XXX use the hrf __repr__ for naming contrasts

    X_exper, cons_exper = design.event_design(experiment, volume_times_rec, hrfs=delay.spectral)

    # The contrasts for 'initial' are ignored 
    # as they are "uninteresting" and are included
    # in the model as confounds.

    X_initial, _ = design.event_design(initial, volume_times_rec, hrfs=[hrf.glover]) 

    # In addition to factors, there is typically a "drift" term
    # In this case, the drift is a natural cubic spline with
    # a not at the midpoint (volume_times.mean())

    vt = volume_times # shorthand
    drift = np.array([vt**i for i in range(4)] + [(vt-vt.mean())**3 * (np.greater(vt, vt.mean()))])
    for i in range(drift.shape[0]):
        drift[i] /= drift[i].max()

    # We transpose the drift so that its shape is (nvol,5) so that it will have
    # the same number of rows as X_initial and X_exper.
        
    drift = drift.T

    # Stack all the designs, keeping the new contrasts
    # which has the same keys as cons_exper, but its
    # values are arrays with 15 columns, with the non-zero
    # entries matching the columns of X corresponding to X_exper

    X, cons = design.stack_designs((X_exper, cons_exper),
                                   (X_initial, {}),
                                   (drift, {}))

    # The default contrasts are all t-statistics.
    # We may want to output F-statistics for
    # 'speaker', 'sentence', 'speaker:sentence' based
    # on the two coefficients, one for each HRF in delay.spectral

    cons['speaker'] = np.vstack([cons['speaker_0'], cons['speaker_1']])
    cons['sentence'] = np.vstack([cons['sentence_0'], cons['sentence_1']])
    cons['sentence:speaker'] = np.vstack([cons['sentence:speaker_0'], 
                                          cons['sentence:speaker_1']])

    # At this point, we're almost ready to fit a model
    # Load in the fMRI data, saving it as an array
    # It is transposed to have time as the first dimension,
    # i.e. fmri[t] gives the t-th volume.

    fmri = np.array(load_image(pjoin(rootdir, "swafunctional_%(run)02d.nii") % path_dict))
    fmri = np.transpose(fmri, [3,0,1,2])
    anat = load_image(pjoin(datadir, "fiac_%(subj)02d", "wanatomical.nii") % path_dict)
                   
    nvol, volshape = fmri.shape[0], fmri.shape[1:] 
    nslice, sliceshape = volshape[0], volshape[1:]

    # XXX Matthew: can you output a brain mask -- it would save
    # having to fit the model everywhere
    #     mask = np.array(load_image(pjoin(datadir, "fiac_%(subj)02d", "mask.nii") % path_dict))
    #     mask_a = np.transpose(np.array(mask), [2,1,0])
    mask_a = 1 # XXX change this once we have a mask

    # The model is a two-stage model, the first stage being an OLS (ordinary least squares) fit,
    # whose residuals are used to estimate an AR(1) parameter for each voxel.

    m = OLSModel(X)
    ar1 = np.zeros(volshape)

    # Fit the model, storing an estimate of an AR(1) parameter at each voxel

    for k in cons.keys():
        if not isestimable(X, cons[k]):
            del(cons[k])
            warnings.warn("contrast %s not estimable for this run" % k)

    for s in range(nslice):
        d = np.array(fmri[:,s])
        flatd = d.reshape((d.shape[0], -1))
        result = m.fit(flatd)
        ar1[s] = ((result.resid[1:] * result.resid[:-1]).sum(0) / (result.resid**2).sum(0)).reshape(sliceshape)

    # We round ar1 to nearest one-hundredth
    # and group voxels by their rounded ar1 value,
    # fitting an AR(1) model to each batch of voxels.

    # XXX smooth here?
    # ar1 = smooth(ar1, 8.0)

    ar1 *= 100
    ar1 = ar1.astype(np.int) / 100.

    # We split the contrasts into F-tests and t-tests.
    # XXX helper function should do this
    
    fcons = {}; tcons = {}
    for n, v in cons.items():
        v = np.squeeze(v)
        if v.ndim == 1:
            tcons[n] = v
        else:
            fcons[n] = v

    # Setup a dictionary to hold all the output
    # XXX ideally these would be memmap'ed Image instances

    output = {}
    for n in tcons:
        tempdict = {}
        for v in ['sd', 't', 'effect']:
            tempdict[v] = np.zeros(volshape)
        output[n] = tempdict
    
    for n in fcons:
        output[n] = np.zeros(volshape)

    # Loop over the unique values of ar1

    for val in np.unique(ar1):
        armask = np.equal(ar1, val)
        armask *= mask_a
        m = ARModel(X, val)
        d = fmri[:,armask]
        results = m.fit(d)

        # Output the results for each contrast

        for n in tcons:
            resT = results.Tcontrast(tcons[n])
            output[n]['sd'][armask] = resT.sd
            output[n]['t'][armask] = resT.t
            output[n]['effect'][armask] = resT.effect

        for n in fcons:
            output[n][armask] = results.Fcontrast(fcons[n]).F

    # Dump output to disk

    odir = pjoin(rootdir, "results_%(run)02d" % path_dict)
    if not exists(odir): makedirs(odir)

    for n in tcons:
        if not exists(pjoin(odir, n)): makedirs(pjoin(odir, n))
        for v in ['t', 'sd', 'effect']:
            im = api.Image(output[n][v], anat.coordmap.copy())
            save_image(im, pjoin(odir, n, '%s.nii' % v))

    for n in fcons:
        im = api.Image(output[n], anat.coordmap.copy())
        if not exists(pjoin(odir, n)): makedirs(pjoin(odir, n))
        save_image(im, pjoin(odir, n, "F.nii"))

def fixed_effects(subj, design):
    """
    Fixed effects (within subject) for FIAC model
    """

    # First, find all the effect and standard deviation images
    # for the subject and this design type

    path_dict = {'subj':subj, 'design':design}
    rootdir = pjoin(datadir, "fiac_%(subj)02d", "%(design)s") % path_dict

    # Which runs correspond to this design type?

    runs = ['results_%02d' % i for i in range(1,5) if exists(pjoin(rootdir, f))]

    # Find out which contrasts have t-statistics,
    # storing the filenames for reading below

    results = {}

    for rundir in runs:
        rundir = pjoin(rootdir, rundir)
        for condir in listdir(rundir):
            results[condir] = []
            for stat in ['sd', 'effect']:
                fname_effect = abspath(pjoin(rundir, condir, 'effect.nii'))
                fname_sd = abspath(pjoin(rundir, condir, 'sd.nii'))
            if exists(fname_effect) and exists(fname_sd):
                results[condir].append([fname_effect,
                                        fname_sd])

    # Get our hands on the relevant coordmap to
    # save our results

    coordmap = load_image(pjoin(datadir, "fiac_%(subj)02d", "wanatomical.nii") % path_dict).coordmap

    # The output directory

    fixdir = pjoin(rootdir, "fixed")

    # Compute the "fixed" effects for each type of contrast

    for con in results:
        fixed_effect = 0
        fixed_var = 0
        for effect, sd in results[con]:
            effect = load_image(effect); sd = load_image(sd)
            var = np.array(sd)**2

            # The optimal, in terms of minimum variance,
            # combination of the effects has weights 1 / var
            # XXX regions with 0 variance are set to 0
            # XXX do we want this or np.nan?

            ivar = np.nan_to_num(1. / var)
            fixed_effect += effect * ivar
            fixed_var += ivar

        # Now, compute the fixed effects variance
        # and t statistic

        fixed_sd = np.sqrt(fixed_var)
        isd = np.nan_to_num(1. / fixed_sd)
        fixed_t = fixed_effect * isd

        # Save the results

        odir = pjoin(fixdir, con)
        if not exists(odir): makedirs(odir)
        for a, n in zip([fixed_effect, fixed_sd, fixed_t],
                        ['effect', 'sd', 't']):
            im = api.Image(a, coordmap.copy())
            save_image(im, pjoin(odir, '%s.nii' % n))

group_mask = load_image(pjoin(datadir, 'group', 'mask.nii'))

def group_analysis(design, contrast):
    """
    Compute group analysis effect, sd and t
    for a given contrast and design type
    """

    rootdir = datadir
    odir = pjoin(rootdir, 'group', design, contrast)
    if not exists(odir): makedirs(odir)

    # Which subjects have this (contrast, design) pair?

    subjects = filter(lambda f: exists(f), [pjoin(rootdir, "fiac_%02d" % s, design, "fixed", contrast) for s in range(16)])

    sd = np.array([np.array(load_image(pjoin(s, "sd.nii"))) for s in subjects])
    Y = np.array([np.array(load_image(pjoin(s, "effect.nii"))) for s in subjects])

    # This function estimates the ratio of the
    # fixed effects variance (sum(1/sd**2, 0))
    # to the estimated random effects variance
    # (sum(1/(sd+rvar)**2, 0)) where
    # rvar is the random effects variance.

    # The EM algorithm used is described in 
    #
    # Worsley, K.J., Liao, C., Aston, J., Petre, V., Duncan, G.H., 
    #    Morales, F., Evans, A.C. (2002). \'A general statistical 
    #    analysis for fMRI data\'. NeuroImage, 15:1-15

    varest = onesample.estimate_varatio(Y, sd)
    random_var = varest['random']

    # XXX - if we have a smoother, use random_var = varest['fixed'] * smooth(varest['ratio'])

    # Having estimated the random effects variance (and
    # possibly smoothed it), the corresponding
    # estimate of the effect and its variance is
    # computed and saved.

    # This is the coordmap we will use

    coordmap = load_image(pjoin(datadir, "fiac_00", "wanatomical.nii")).coordmap

    adjusted_var = sd**2 + random_var
    adjusted_sd = np.sqrt(adjusted_var)

    results = onesample.estimate_mean(Y, adjusted_sd) 
    for n in ['effect', 'sd', 't']:
        im = api.Image(results[n], coordmap.copy())
        save_image(im, pjoin(odir, "%s.nii" % n))

def group_analysis_signs(design, contrast, signs, mask):
    """
    This function refits the EM model with a vector of signs.
    Used in the permutation tests.

    Returns the maximum of the T-statistic within mask

    Parameters
    ----------

    design: one of 'block', 'event'

    contrast: str

    signs: ndarray

    mask: ndarray

    Returns
    -------

    minT: float, minimum of T statistic within mask

    maxT: float, maximum of T statistic within mask
    
    """

    rootdir = datadir

    # Which subjects have this (contrast, design) pair?

    subjects = filter(lambda f: exists(f), [pjoin(rootdir, "fiac_%02d" % s, design, "fixed", contrast) for s in range(16)])

    sd = np.array([np.array(load_image(pjoin(s, "sd.nii")))[:,mask] for s in subjects])
    Y = np.array([np.array(load_image(pjoin(s, "effect.nii")))[:,mask] for s in subjects])
    signY = signs[:,np.newaxis,np.newaxis,np.newaxis] * Y

    varest = onesample.estimate_varatio(Y, sd)
    random_var = varest['random']

    adjusted_var = sd**2 + random_var
    adjusted_sd = np.sqrt(adjusted_var)

    results = onesample.estimate_mean(Y, adjusted_sd) 
    T = results['t']

    return np.nanmin(T), np.nanmax(T)
    
    
